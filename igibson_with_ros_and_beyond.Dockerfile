# from nvidia/cudagl:10.0-base-ubuntu18.04
from nvidia/cudagl:11.0-base-ubuntu20.04
LABEL stage=buildingWithRos

ARG CUDA=11.0
# ARG CUDNN=7.6.2.24-1
# ARG CUDNN=8.0.2.39-1
ARG CUDNN=8.0.5.39-1

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update  && apt-get install -y --no-install-recommends \
	curl build-essential git cmake \
	cuda-command-line-tools-11-0 \
    libcublas-11-0 \
    libcufft-11-0 \
    libcurand-11-0 \
    libcusolver-11-0 \
    libcusparse-11-0 \
    libcudnn8=${CUDNN}+cuda${CUDA} \
    vim \
    tmux \
    libhdf5-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
RUN bash Miniconda-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda create -y -n py3-igibson python=3.6.8
# Python packages from conda

ENV PATH /miniconda/envs/py3-igibson/bin:$PATH

################################################################################
# RUN /bin/bash -c ". activate py3-igibson; pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html"
RUN /bin/bash -c ". activate py3-igibson; pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html"

#ROS-NOETIC BECAUSE IT USES PYTHON3
RUN /bin/bash -c ". activate py3-igibson; echo 'deb http://packages.ros.org/ros/ubuntu focal main' > /etc/apt/sources.list.d/ros-latest.list "
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update
RUN apt-get install -y ros-noetic-desktop
# RUN source /opt/ros/noetic/setup.bash


RUN apt-get install -y ros-noetic-vision-opencv \
	ros-noetic-base-local-planner \
	ros-noetic-costmap-converter \
	ros-noetic-move-base-flex \
	ros-noetic-geometry2 \
	libsuitesparse-dev \
	ros-noetic-libg2o \
	ros-noetic-rtabmap \
	ros-noetic-rtabmap-ros

RUN /bin/bash -c ". activate py3-igibson; pip install defusedxml"
RUN apt-get install -y ros-noetic-image-pipeline

RUN git clone https://github.com/catkin/catkin_tools.git /catkin_tools

# RUN /bin/bash -c ". activate py3-igibson; pip install catkin_tools; mkdir -p /miniconda/envs/py3-igibson/etc/conda/activate.d; ln -s /opt/ros/noetic/setup.bash /miniconda/envs/py3-igibson/etc/conda/activate.d/setup_ros.bash"
RUN /bin/bash -c ". activate py3-igibson; cd /catkin_tools; pip install -r requirements.txt ; python setup.py install --record install_manifest.txt ; mkdir -p /miniconda/envs/py3-igibson/etc/conda/activate.d; ln -s /opt/ros/noetic/setup.bash /miniconda/envs/py3-igibson/etc/conda/activate.d/setup_ros.bash"
################################################################################

RUN pip install pytest
# RUN pip install tensorflow-gpu==1.15.0

#should work with iGibson v1.0.1
RUN git clone --branch 1.0.1 https://github.com/StanfordVL/iGibson /opt/igibson --recursive
WORKDIR /opt/igibson

RUN apt-get install -y nano

RUN /bin/bash -c ". activate py3-igibson; pip uninstall -y pybullet; pip install https://github.com/StanfordVL/bullet3/archive/master.zip"
################################################################################
ADD vribot_ros_2021 /vribot
# ADD vriframework /vriframework
ADD iGibson_v101/gibson2/examples/ros/ /opt/igibson/gibson2/examples/ros/
ADD iGibson_v101/gibson2/global_config.yaml /opt/igibson/gibson2/global_config.yaml
ADD iGibson_v101/gibson2/robots/vribot_robot.py /opt/igibson/gibson2/robots/vribot_robot.py
ADD iGibson_v101/gibson2/envs/env_base.py /opt/igibson/gibson2/envs/env_base.py
ADD iGibson_v101/gibson2/scenes/indoor_scene.py /opt/igibson/gibson2/scenes/indoor_scene.py

RUN pip install -e .
################################################################################

################################################################################
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN /bin/bash -c ". /opt/ros/noetic/setup.bash; pip install empy rospkg; ln -sfn /opt/igibson /vribot/src/gibson2-ros; cd /vribot; catkin clean --yes; catkin build --cmake-args -DPYTHON_EXECUTABLE=/miniconda/envs/py3-igibson/bin/python3; ln -s /vribot/devel/setup.bash /miniconda/envs/py3-igibson/etc/conda/activate.d/setup_vribot.bash"
RUN echo "source /vribot/devel/setup.bash" >> ~/.bashrc
################################################################################

ADD compile_yolact.sh /compile_yolact.sh

RUN apt-get install -y ros-noetic-global-planner \
  libcublas-dev-11-0 \
  libcufft-dev-11-0 \
  libcurand-dev-11-0 \
  libcusolver-dev-11-0 \
  libcusparse-dev-11-0 \
  libcudnn8-dev=${CUDNN}+cuda${CUDA}

RUN /bin/bash -c ". activate py3-igibson; pip install numpy-quaternion pycocotools numba yacs;"

RUN /bin/bash -c ". activate py3-igibson; rm -rf /opt/igibson/gibson2/examples/ros/gibson-ros/beyond_agent/yolact/external/DCNv2; git clone --branch pytorch_1.7 https://github.com/lbin/DCNv2.git /opt/igibson/gibson2/examples/ros/gibson-ros/beyond_agent/yolact/external/DCNv2"
