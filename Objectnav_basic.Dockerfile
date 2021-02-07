ARG UBUNTU_VERSION=18.04

ARG ARCH=devel
ARG CUDA=10.1

FROM nvidia/cuda:${CUDA}-${ARCH}-ubuntu${UBUNTU_VERSION} AS cuda_toolkit


FROM fairembodied/habitat-challenge:testing_2020_habitat_base_docker AS base_image
COPY --from=cuda_toolkit /usr/local/cuda-10.1 /usr/local/cuda-10.1
COPY --from=cuda_toolkit /usr/local/cuda /usr/local/cuda

# Configure the build for our CUDA configuration.
ENV PATH $PATH:/usr/local/cuda-10.1/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib64


# install dependencies in the habitat conda environment and visualpriors package
RUN /bin/bash -c ". activate habitat; pip install ifcfg tensorboard && pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html && pip install visualpriors"

#update to v0.1.5
RUN /bin/bash -c ". activate habitat ; cd habitat-sim ; git reset --hard; git fetch --all; git checkout v0.1.5; git status; python setup.py build_ext --parallel 5 install --with-cuda --headless; cd .."

#update to v0.1.5
RUN /bin/bash -c ". activate habitat ; rm -rf habitat-api ;git clone https://github.com/facebookresearch/habitat-api.git ; cd habitat-api ; git reset --hard; git fetch --all; git checkout v0.1.5; git status; python setup.py develop build_ext --parallel 5 install; cd .." AS v0.1.5
