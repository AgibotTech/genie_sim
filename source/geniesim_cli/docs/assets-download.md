# 按任务下载 benchmark 资产

`geniesim assets` 从 ModelScope 或 HuggingFace 下载指定任务的资产闭包，不要求先安装 Isaac Sim、ROS、USD 或 ModelScope SDK。

## 安装 CLI

在代码仓库根目录运行：

```bash
python3 -m pip install -e source/geniesim_cli
```

## 下载 g2op_if

```bash
# 查看收录任务及下载规模，不下载资产
geniesim assets list g2op_if
geniesim assets plan g2op_if --output /tmp/g2op-if-assets.json

# 下载到独立目录；相同命令可在中断后重新运行
geniesim assets download g2op_if --dest "$HOME/sim/assets-selected" --workers 8

# 离线逐文件复核大小及 SHA256
geniesim assets verify g2op_if --dest "$HOME/sim/assets-selected"
```

`g2op_if` 对应 10 个任务、1137 个文件，逻辑大小约 7.80 GiB，按相同内容去重后约 6.42 GiB。下载前还需至少 256 MiB 余量。首次下载包含全部 73 个已发布 HDR/EXR，以支持背景泛化。

## 其他任务选择方式

```bash
# 按 category 下载；多个选择器取并集，不重复下载公共内容
geniesim assets download manip spatial --dest "$HOME/sim/assets-selected"
geniesim assets download dexmanip finemanip wbc crsemb --dest "$HOME/sim/assets-selected"

# 按机器人和 category 组合筛选
geniesim assets plan g2op_crsemb

# 仅下载一个任务的全部已收录场景实例依赖
geniesim assets download g2crsbwuji_dexmanip_take_the_book --dest "$HOME/sim/assets-selected"

# 查看所有已收录任务
geniesim assets list
```

## 切换下载源

同一份资产同时发布在 ModelScope 和 HuggingFace（`agibot-world/GenieSimAssets`），两者内容逐字节一致。默认使用 ModelScope；用 `--source` 切换：

```bash
# 从 HuggingFace 镜像下载
geniesim assets download g2op_if --source huggingface --dest "$HOME/sim/assets-selected"

# 查看可用源及各自固定的 revision
geniesim assets list --source huggingface
```

索引为每个源固定各自的 commit（两个仓库是独立 git 历史），并共用同一份文件 SHA256 表。因此无论从哪个源下载，每个文件都必须匹配相同的 SHA256，源不会改变内容。已下载的目录可直接换源继续用：相同字节被复用，不会重复下载。

支持任务配置全名（可带 `.yaml`）、下划线边界上的名字前缀和 category。不匹配的选择器返回非零状态，不退回全量下载。

## 运行时使用下载目录

在宿主机启动新的 GenieSim 容器前指定资产来源：

```bash
export GENIESIM_ASSETS_SRC="$HOME/sim/assets-selected"
geniesim docker5.1 up
geniesim docker5.1 into
```

进入已经准备好的运行环境后，仍用正式 benchmark 命令，例如：

```bash
geniesim benchmark run g2op_if_pick_block_color -- \
  --benchmark.preview=true --benchmark.instance_ids=0 \
  --benchmark.record=false --app.headless=true
```

资产挂载在创建容器时确定；已经运行的容器不会因为宿主机修改环境变量而切换挂载。下载 CLI 本身不启动仿真，也不安装引擎、机器人 ROS 包或推理服务。原生 Newton/libuipc/MuJoCo 任务仍需各自的正式运行环境。

## 下载与校验行为

- 下载的是发布索引中逐任务记录的 USD、嵌套引用、前景、背景、机器人、材质、纹理、元数据、行为脚本和 interaction 数据；所有收录实例的依赖取并集。
- 同一目标目录中的相同 SHA256 内容使用硬链接去重；成功文件再次运行只校验和复用。中断文件保留，服务器支持 HTTP Range 时断点续传；不支持时从头下载。
- 每份内容只有在大小和 SHA256 均匹配后才成为最终资产文件。失败命令返回非零状态；重新运行相同下载命令可继续。
- 同一目录只允许一个下载进程。下载输出保存在 `.geniesim-assets.json`，内容缓存保存在 `.geniesim-download/`。
- 默认保护已存在但与发布哈希不同的文件。确需替换时显式加 `--overwrite`。下载不会删除其他文件。
- 不向原来的全量资产目录回退，也不通过外部符号链接补齐文件。建议使用独立下载目录。
- `verify` 只确认所选文件的发布字节完整性，不代表场景初始化、材质质量、策略运行或计分通过。

## 固定版本与已知问题

当前内置索引来自已完成按需下载和初始化预览验证的 150 个任务，覆盖 `g2op_if`、`manip`、`spatial`、`dexmanip`、`finemanip`、`wbc`、`crsemb`、`robust`、`sandbox`。资产 revision 固定为 `dfa5864361ec79188cfcc5795a8ea60f02f7b508`，对应 benchmark 代码基线 `bdecff390b60bfaa187d7a960b7e59b8d9e1e969`；索引记录每个任务配置的 SHA256。

`sandbox` 收录 25 个任务中的 22 个；`sandbox_hotel_store_suitcase`、`sandbox_restaurant_guide_guests`、`sandbox_store_carry_carton` 这 3 个任务的 scene_info 引用了当前发布 revision 中缺失的对象资产，待资产补齐后再纳入索引。

该命令不扫描远程 `master` 的未来改动，不自动推断未收录的新任务。新版代码、资产或额外自定义任务需要重新生成经过审计的版本索引；`--index FILE` 可指定其他相同 schema 的索引。

本轮预览发现发布版本存在问题：魔方原生包的 `MANIFEST.json` 引用了缺失的 `README.md`，导致官方 CLI 初始化失败；还存在部分元数据解析、贴图损坏、脚本路径、SDK 绝对引用、PhysX 和表面外观问题。文件哈希正确不能修复这些源内容问题。下载魔方资产时 CLI 会明确提示已知的包清单问题，不会伪造缺失文件。

## 维护索引

`source/geniesim_cli/scripts/build_asset_index.py` 从固定版本的依赖审计清单编译 `_asset_index.json.gz`。维护操作需要 PyYAML，普通下载命令仅依赖 Python 标准库。输入为审计产生的 `manifest.json`（含递归依赖 edges、资产单元侧车文件、运行时补充根、完整文件哈希）和匹配的 benchmark 源码：

```bash
python3 source/geniesim_cli/scripts/build_asset_index.py \
  --benchmark-source source/geniesim_benchmark/src/geniesim_benchmark \
  --output source/geniesim_cli/src/geniesim_cli/_asset_index.json.gz \
  /path/to/audited-group/manifest.json
```

多个审计清单可一起传入。编译器拒绝版本冲突、文件哈希冲突和未完成的扫描；逐任务计算引用闭包，并要求同一组任务的文件并集与原审计集合完全一致。索引更新后需重新做独立目录下载、原生包清单检查和实际场景初始化验证；编译索引本身不做 GPU 验证。
