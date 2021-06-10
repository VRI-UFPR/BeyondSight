ARG UBUNTU_VERSION=18.04

ARG ARCH=devel
ARG CUDA=10.2

FROM nvidia/cuda:${CUDA}-${ARCH}-ubuntu${UBUNTU_VERSION} AS cuda_toolkit


FROM fairembodied/habitat-challenge:testing_2021_habitat_base_docker AS base_image
COPY --from=cuda_toolkit /usr/local/cuda-10.2 /usr/local/cuda-10.2
COPY --from=cuda_toolkit /usr/local/cuda /usr/local/cuda

# Configure the build for our CUDA configuration.
ENV PATH $PATH:/usr/local/cuda-10.2/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64

################################################################################
COPY --from=nvidia/cuda:10.2-devel-ubuntu18.04 /usr/include/cublas* /usr/include/
################################################################################

# install dependencies in the habitat conda environmend
#cuda102
RUN /bin/bash -c ". activate habitat; pip install ifcfg tensorboard && pip install torch==1.7.1 torchvision==0.8.2 -f https://download.pytorch.org/whl/torch_stable.html"
RUN /bin/bash -c ". activate habitat; pip uninstall -y habitat-sim; git clone https://github.com/facebookresearch/habitat-sim.git habitat-sim ; cd habitat-sim ; git checkout v0.1.7; git status; cd habitat-sim ; python setup.py build_ext --parallel 5 install --with-cuda --headless; cd .."

LABEL stage=builderBeyondStage0
