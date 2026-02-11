# Introduction
This is a scene level reconstruction pipeline. It supports automatically reconstructing high-fidelity and high-precision rendering results in accordance with mesh data.

# COLMAP-PCD patch's MD5 message digest
6b93584ad55c017ca335903ef083b084

# PGSR patch's MD5 message digest
# pgsr's patch/pgsr.patch patch
f4e5c2a401dc7350255f9a6de7a51310

# Hierarchical-Localization patch's MD5 message digest
# hloc's patch/hloc.patch patch
37f8b5f5d46940c8995b7e1281546485

# Build Dockerfile
# git clone codebase and cd source/scene_reconstruction
```bash
docker build . -t xxx

# Or use proxy to build
docker build . -t xxx --build-arg http_proxy="http://ip_addr:port" --build-arg https_proxy="https://ip_addr:port"
```

# Download weights
## Either download the weight file in advance and put them into the docker container, or run the code and wait for automatic download
```bash
# Place to  /root/.cache/torch/
hub/
├── checkpoints
│   ├── alexnet-owt-7be5be79.pth
│   ├── alex.pth
│   ├── aliked-n16.pth
│   ├── superpoint_lightglue_v0-1_arxiv.pth
│   ├── superpoint_v1.pth
│   └── vgg16-397923af.pth
└── netvlad
    └── VGG16-NetVLAD-Pitts30K.mat
```

# References
1. The COLMAP-PCD [source code]. https://github.com/XiaoBaiiiiii/colmap-pcd
2. The gsplat [source code]. https://github.com/nerfstudio-project/gsplat
3. The PGSR [Source code]. https://github.com/zju3dv/PGSR
4. The Difix3d [Source code]. https://github.com/nv-tlabs/Difix3D
