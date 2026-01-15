# Real2Sim Environment Asset Flow Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as system-builder

ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_ALWAYS_YES=true
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV MAX_JOBS=4
ENV QT_QPA_PLATFORM=offscreen

RUN apt-get update && apt-get install -y \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    libgmock-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev \
    libpdal-dev \
    pdal \
    libatlas-base-dev \
    libflann-dev \
    libboost-all-dev \
    libopencv-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglm-dev \
    libceres-dev \
    libcurl4-openssl-dev \
    libmkl-full-dev \
    liblaszip-dev \
    python3-pip \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    libpcl-dev \
    pcl-tools \
    qttools5-dev \
    qttools5-dev-tools \
    flatpak \
    ssh \
    cmake \
    xvfb  \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    libgif-dev \
    libwebp-dev \
    zlib1g-dev \
    libxft-dev \
    pkg-config && rm -rf /var/lib/apt/lists/*

WORKDIR /root
COPY . .
# cloudcompare
RUN git clone https://github.com/CloudCompare/CloudCompare.git
RUN cd CloudCompare && git checkout v2.13.2 && git submodule update --init --recursive && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Rlease -DPLUGIN_STANDARD_QPCL=ON \
    -DPLUGIN_IO_QLAS=ON -DPLUGIN_IO_QCORE=ON -DPLUGIN_IO_QPDAL=ON && \
    make -j$(nproc -1) && make install && cd ../.. &&  rm -rf CloudCompare

# Install colmap-pcd
RUN git clone https://github.com/XiaoBaiiiiii/colmap-pcd.git
RUN cd colmap-pcd && git checkout 9cd7d9b7f257306483dc6ecc95d4ef447335888d && git apply ../patch/colmap-pcd.patch \
     && mkdir build && cd build && cmake .. && make -j$(nproc -1) && make install && cd ../.. && rm -rf colmap-pcd

# Install conda env
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py312_24.9.2-0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# config conda env prevent to auto activate base env.
RUN conda config --set auto_activate_base false && \
    conda clean -afy

RUN conda --version && python --version

FROM system-builder as pgsr-builder
# Install PGSR enviroments
RUN cd ./third_party && git clone https://github.com/zju3dv/PGSR.git && cd PGSR && \
    git checkout de24f1a38b350387e8d8fe381b2cd70c1ae946e7 && \
    git apply ../../patch/pgsr.patch && \
    git submodule update --init --recursive && \
    conda init && \
    conda create -n pgsr python=3.8 -y &&  rm -rf /root/.cache/pip
RUN conda run -n pgsr pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    rm -rf /root/.cache/pip
RUN cd ./third_party/PGSR && conda run -n pgsr pip install -r requirements.txt && rm -rf /root/.cache/pip
RUN cd ./third_party/PGSR &&  conda run -n pgsr pip install pycolmap==3.11.1 && \
    conda run -n pgsr pip install submodules/diff-plane-rasterization && \
    conda run -n pgsr pip install submodules/simple-knn rich laspy \
    && rm -rf /root/.cache/pip

FROM pgsr-builder as difix-builder
# install Difix3d
RUN conda create -n difix python=3.8 -y  && rm -rf /root/.cache/pip
RUN cd ./third_party && git clone https://github.com/nv-tlabs/Difix3D.git && \
    conda run -n difix pip3 install -r Difix3D/requirements.txt && rm -rf /root/.cache/pip
RUN conda run -n difix pip3 install matplotlib scikit-learn plyfile rich && rm -rf /root/.cache/pip

FROM difix-builder as gsplat-builder
# base install gsplat
RUN conda run -n base python3 -m pip install plyfile torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    && rm -rf /root/.cache/pip
RUN cd ./third_party/gsplat && conda run -n base pip3 uninstall pycolmap && \
    conda run -n base pip3 install -r examples/requirements.txt && \
    conda run -n base python -m pip install . && rm -rf /root/.cache/pip

RUN cd ./third_party && git clone https://github.com/cvg/Hierarchical-Localization/  && \
    cd Hierarchical-Localization && \
    git checkout e33422019fee354b7887d05cf6deb525cfe47524  && \
    git submodule update --init --recursive && \
    git apply ../../patch/hloc.patch

ENV PYTHONPATH=/root/third_party/Hierarchical-Localization:$PYTHONPATH
RUN /opt/conda/envs/pgsr/bin/python -m pip install -r requirements.txt && \
    cd /root/third_party/Hierarchical-Localization && \
    /opt/conda/envs/pgsr/bin/python -m pip install e . && \
    rm -rf /root/.cache/pip

RUN rm -rf /root/.cache/pip
RUN rm -rf /var/lib/apt/lists/*

CMD ["/bin/bash"]
