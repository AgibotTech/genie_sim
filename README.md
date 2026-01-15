
![image.png](./docs/image.png)
<div align="center">
  <a href="https://arxiv.org/abs/2601.02078" style="text-decoration:none;">
    <img src="https://img.shields.io/badge/arXiv-2601.02078-red.svg?logo=arxiv&logoColor=white" alt="arXiv Paper: 2601.02078">
  </a>
  <a href="https://github.com/AgibotTech/genie_sim">
    <img src="https://img.shields.io/badge/GitHub-grey?logo=GitHub" alt="GitHub">
  </a>
  <a href="http://agibot-world.com/genie-sim">
    <img src="https://img.shields.io/badge/Webpage-GenieSim-green?" alt="Webpage">
  </a>
  <!-- <a href="https://huggingface.co/datasets/agibot-world/GenieSimAssets">
    <img src="https://img.shields.io/badge/HuggingFace-yellow?logo=HuggingFace" alt="HuggingFace">
  </a> -->
  <a href="https://modelscope.cn/datasets/agibot_world/GenieSim3.0-Dataset">
    <img src="https://img.shields.io/badge/Dataset-ModelScope-FF6B35?logo=model&logoColor=white" alt="ModelScope Dataset">
  </a>
  <div align="center">
    <a href="https://agibot-world.com/videos/genieSim/modules/heroFullVideoEn.mp4" target="_blank">
      <img src="./docs/videoPlay.png" alt="Play Video" />
    </a>
  </div>
</div>


# 1. Genie Sim 3.0
Genie Sim is the simulation platform from AgiBot. It provides developers with a complete toolchain for environment reconstruction, scene generalization, data collection, and automated evaluation. Its core module, Genie Sim Benchmark is a standardized tool dedicated to establishing the most accurate and authoritative evaluation for embodied intelligence.

The platform integrates 3D reconstruction with visual generation to create a high-fidelity simulation environment. It pioneers LLM-driven technology to generate vast simulation scenes and evaluation configurations in minutes. The evaluation system covers 200+ tasks across 100,000+ scenarios to establish a comprehensive capability profile for models. Genie Sim also opens over 10,000 hours synthetic dataset including real-world robot operation scenarios.

The platform will significantly accelerate model development, reduce reliance on physical hardware, and empower innovation in embodied intelligence. Simulation assets, dataset, and code are fully open source.

# 2. Features
- **High-Fidelity Sim-Ready Assets**: 5,140 validated 3D assets covering five real-world operation fields: retail, industry, catering, home and office. [ModelScope](https://modelscope.cn/datasets/agibot_world/GenieSimAssets).
- **3DGS-based Reconstruction Pipeline**: Integrate 3DGS-based reconstruction process with visual generative model to synthesize realistic simulation environment with high-precision meshes. [ModelScope](https://modelscope.cn/datasets/agibot_world/GenieSim3.0-Dataset).
- **LLM-Driven Scene Generation**: Natural language-driven generation and generalization which instantly generates diverse simulation scenes through conversational interaction.
- **Large-Scale Synthetic Dataset**: Over 10,000 hours open-source synthetic data across 200+ loco-manipulation tasks with multi-sensor streams, alongside multi-dimensional variations.
- **Synthetic Data Generation**: Efficient toolkit for data collectoin with error-recovery mechanism, supporting both low-latency teleoperation and automated data programming. [ModelScope](https://modelscope.cn/datasets/agibot_world/GenieSim3.0-Dataset).
- **Robust and Diverse Benchmark**: Provide 100,000+ simulation scenarios and use LLM to autonomously generate task instructions and evaluation configurations. Discrepancy between simulation and real-world test results is less than 10%.
- **VLM-based Auto-Evaluation System**: Full-spectrum evaluation criteria to provide model's capability profile covering manipulation skills, cognitive comprehension and task complexity.
- **Zero-Shot Sim-to-Real Transfer**: Model trained with our synthetic data exhibits zero-shot sim-to-real transfer capability with superior task success rate compared to model trained with real data.

# 3. Updates
- [1/7/2026] v3.0
  - Update Isaac Sim to v5.1.0 and support RTX 50series graphic card
  - Provide USD and URDF files of Genie G2 robot and support whole body control
  - Support 3DGS-based scene reconstruction and convert output to USD format for application in Isaac Sim
  - Release synthetic dataset and corresponding data collection pipeline
  - Add LLM-based features to generate scenarios, task instructions and evaluation configurations
- [7/14/2025] v2.2
  - Provide detailed evaluation metrics for all Agibot World Challenge tasks
  - Add automatic evaluation script to run each task multiple times and record score of all steps
- [6/25/2025] v2.1
  - Add 10 more manipulation tasks for Agibot World Challenge 2025 including all simulation assets
  - Open-source synthetic datasets for 10 manipulation tasks on [Huggingface](https://huggingface.co/datasets/agibot-world/AgiBotWorldChallenge-2025/tree/main/Manipulation-SimData)
  - Integrate UniVLA policy and support model inference simulation evaluation
  - Update IK solver sdk which supports cross-embodiment IK solving for other robots
  - Optimize communication framework and improve simulation running speed by 2x
  - Update automatic evaluation framework for more complicated long-range tasks

# 4. Documentation

Please refer to these links to install Genie Sim and download assets and dataset:
- [User Guide](https://agibot-world.com/sim-evaluation/docs/#/v3)
- [Assets](https://modelscope.cn/datasets/agibot_world/GenieSimAssets)
- [Dataset](https://modelscope.cn/datasets/agibot_world/GenieSim3.0-Dataset)

## 4.1 Support
<img src="./docs/wechat.JPEG" width="30%"/>

## 4.2 Roadmap
- [x] Release more long-horizon benchmark mainuplation tasks
- [x] More scenes and assets for each benchmark task
- [x] Support Agibot World Challenge baseline model
- [x] Scenario layout and manipulation trajectory generalization toolkit
- [x] Provide dockfile and tutorial for scene reconstruction pipeline
- [ ] Upload all assets and dataset on Huggingface
- [ ] Update motion control toolkit to support Genie G2 teleoperation in simulation
- [ ] Human-in-the-loop and distributed reinforcement learning pipline

## 4.3 License and Citation
All the data and code within `source/geniesim` and `source/data_collection` are under `Mozilla Public License 2.0`. The `source/scene_reconstruction` project contains code under multiple licenses, for complete and updated licensing details, please see the LICENSE files

Please consider citing our work either way below if it helps your research.

```
@misc{yin2026geniesim30,
  title={Genie Sim 3.0 : A High-Fidelity Comprehensive Simulation Platform for Humanoid Robot},
  author={Chenghao Yin and Da Huang and Di Yang and Jichao Wang and Nanshu Zhao and Chen Xu and Wenjun Sun and Linjie Hou and Zhijun Li and Junhui Wu and Zhaobo Liu and Zhen Xiao and Sheng Zhang and Lei Bao and Rui Feng and Zhenquan Pang and Jiayu Li and Qian Wang and Maoqing Yao},
  year={2026},
  eprint={2601.02078},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2601.02078},
}
```

## 4.4 References
1. PDDL Parser (2020). Version 1.1. [Source code]. https://github.com/pucrs-automated-planning/pddl-parser.
2. BDDL. Version 1.x.x [Source code]. https://github.com/StanfordVL/bddl
3. CUROBO [Source code]. https://github.com/NVlabs/curobo
4. Isaac Lab [Source code]. https://github.com/isaac-sim/IsaacLab
5. Omni Gibson [Source code]. https://github.com/StanfordVL/OmniGibson
6. The Scene Language [Source code]. https://github.com/zzyunzhi/scene-language
