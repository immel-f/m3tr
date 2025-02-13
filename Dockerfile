FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

ARG CMD="/bin/bash"

# Manually install old version of yapf to prevent crash in mmcv due to yapf api changes
RUN pip install yapf==0.40.1
RUN pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
RUN pip install mmdet==2.28.0
RUN pip install mmsegmentation==0.30.0

RUN pip install timm

RUN apt-get update ; exit 0
RUN apt-get install git -y
RUN apt-get install git-lfs -y
RUN apt-get install wget -y

RUN git lfs install

# install missing libraries
RUN apt-get install libglvnd-dev libglib2.0-0 -y

COPY . /workspace/m3tr
WORKDIR /workspace/m3tr

WORKDIR /workspace/m3tr/mmdetection3d
RUN pip install -v -e .
WORKDIR /workspace/m3tr/projects/mmdet3d_plugin/m3tr/modules/ops/geometric_kernel_attn
RUN pip install -v .
WORKDIR /workspace/m3tr
RUN pip install -r requirement.txt

RUN mkdir ckpts
# docker run command: mount argoverse into these paths
RUN mkdir -p /datasets/public/argoverse20
WORKDIR /workspace/m3tr/ckpts
RUN wget https://download.pytorch.org/models/resnet50-19c8e357.pth
RUN wget https://download.pytorch.org/models/resnet18-f37072fd.pth
RUN wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth

# fix dependency issues
RUN pip install numpy==1.21.5 shapely==1.8.5

# used to manually verify that mmdet3d is working after docker build
WORKDIR /workspace/m3tr/mmdetection3d
RUN mkdir checkpoints
WORKDIR /workspace/m3tr/mmdetection3d/checkpoints
RUN wget https://download.openmmlab.com/mmdetection3d/v0.1.0_models/second/hv_second_secfpn_6x8_80e_kitti-3d-car/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth
WORKDIR /workspace/m3tr/

#add jupyter for debugging
RUN pip install jupyter

#install newer version of networkx for av2 annotation script
RUN pip install networkx==2.3

# install lanelet2_ml_converter experimental build
WORKDIR /workspace/m3tr/ll2_wheels
RUN pip install ./lanelet2-1.2.1-cp38-cp38-manylinux_2_27_x86_64.whl

# enable SLURM support inside docker / apptainer container for HPC
RUN rm /opt/conda/lib/python3.8/site-packages/mmcv/runner/dist_utils.py
COPY docker_res/dist_utils.py /opt/conda/lib/python3.8/site-packages/mmcv/runner/dist_utils.py


# create an entrypoint so that the workspace is always sourced when starting a container
RUN echo "#!/bin/bash"                          | tee /entrypoint.sh && \
    echo "set -e"                               | tee -a /entrypoint.sh && \
    echo "exec \$@"                             | tee -a /entrypoint.sh && \
    chmod a+x /entrypoint.sh

WORKDIR /workspace/m3tr/
ENTRYPOINT ["/entrypoint.sh"]
ENV CMD="${CMD}"
CMD ${CMD}




