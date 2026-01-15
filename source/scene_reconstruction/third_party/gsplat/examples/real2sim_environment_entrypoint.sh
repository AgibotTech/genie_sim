#!/bin/bash

OUTPUT_NAME="gs-output"
GS_ASSET='gs-asset'
INPUT_PATH=$1
NVS_FLAG=$2
echo "Input path for Real2Sim: $INPUT_PATH"

# Is path exist
if [ ! -d "$INPUT_PATH" ]; then
    echo "ERROR: Input path does not exist: $INPUT_PATH"
    exit 1
fi

cd /root/third_party/gsplat/examples

# activate conda env
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        . "/opt/conda/etc/profile.d/conda.sh"
    else
        export PATH="/opt/conda/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate pgsr

# launch
echo "Running Real2Sim processing on: $INPUT_PATH"

python3 batch_rec.py --path "$INPUT_PATH/"

conda activate base

python3 simple_trainer_agi.py default \
    --data_dir "$INPUT_PATH/" \
    --result_dir "${INPUT_PATH}/${OUTPUT_NAME}/"


if [ "$NVS_FLAG" = "1" ]; then
    conda activate difix
    DIFIX3D_PATH='/root/third_party/Difix3D'
    SPARSE_PATH="sparse/0"
    mkdir -p ${INPUT_PATH}/novel
    echo "Python: $(which python3)"
    python3 get_noval_view_and_render.py \
    --dataset_path "${INPUT_PATH}/novel" \
    --images_bin_path "${INPUT_PATH}/${SPARSE_PATH}/images.bin" \
    --checkpt_ply_path "${INPUT_PATH}/${OUTPUT_NAME}/ply/point_cloud_29999.ply" \
    --points3d_ply_path "${INPUT_PATH}/${SPARSE_PATH}/sparse.ply" \
    --batch_size 16

    python3 fix_novel.py \
        --input_dir ${INPUT_PATH}/novel/images_novel_view \
        --output_dir ${INPUT_PATH}/novel/images \
        --difix_src_path ${DIFIX3D_PATH}/src \
        --model_path ${DIFIX3D_PATH}/hf_model \
        --batch_size 1 \
        --device cuda:0
fi

conda activate pgsr
cd /root/third_party/PGSR
mkdir -p ${INPUT_PATH}/${GS_ASSET}
python3 train.py -s ${INPUT_PATH}/novel -m ${INPUT_PATH}/${GS_ASSET} -r1 --ncc_scale 0.5  --exposure_compensation
echo "Real2Sim processing completed"
