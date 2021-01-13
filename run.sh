#!/usr/bin/env bash

DOCKER_NAME="igibson"

while [[ $# -gt 0 ]]
do
key="${1}"

case $key in
    --docker-name)
      shift
      DOCKER_NAME="${1}"
	    shift
      ;;
    --cuda_visible_devices)
      shift
      CUDA_VISIBLE_DEVICES="${1}"
      shift
      ;;
    --mode)
      shift
      MODE="${1}"
      shift
      ;;
    *) # unknown arg
      echo unkown arg ${1}
      exit
      ;;
esac
done

echo "CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}
echo "MODE" ${MODE}

if [ $MODE = 1 ]; then
  #run standand CMD set on the docker image using old syntax
  docker run \
      -v $(pwd)/gibson_data:/gibson_data \
      -v $(realpath gibson_dataset):/gibson_dataset \
      --runtime nvidia \
      -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
      --rm \
      --name igibson \
      ${DOCKER_NAME}
elif [ $MODE = 2 ]; then
  export QT_X11_NO_MITSHM=1

  X11_PARAMS=""
  if [[ -e /tmp/.X11-unix && ! -z ${DISPLAY+x} ]]; then
    echo "Using local X11 server"
    X11_PARAMS="-e DISPLAY=$DISPLAY  -v /tmp/.X11-unix:/tmp/.X11-unix:rw --env QT_X11_NO_MITSHM=1"
    xhost +local:root
  fi;

  #run standand CMD set on the docker image with Xserver
  docker run \
      -v $(pwd)/gibson_data:/gibson_data \
      -v $(realpath gibson_dataset):/gibson_dataset \
      --gpus all \
      -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
      --privileged $X11_PARAMS -it \
      --rm \
      --name igibson \
      ${DOCKER_NAME}

  if [[ -e /tmp/.X11-unix && ! -z ${DISPLAY+x} ]]; then
      xhost -local:root
  fi;

elif [ $MODE = 3 ]; then
  #run a bash inside the containeir
  docker run \
      -v $(pwd)/gibson_data:/gibson_data \
      -v $(realpath gibson_dataset):/gibson_dataset \
      --gpus all \
      -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
      --rm \
      --name igibson \
      -it \
      ${DOCKER_NAME} \
      /bin/bash

elif [ $MODE = 4  ]; then
  #run a bash inside the container with X
  export QT_X11_NO_MITSHM=1

  X11_PARAMS=""
  if [[ -e /tmp/.X11-unix && ! -z ${DISPLAY+x} ]]; then
    echo "Using local X11 server"
    X11_PARAMS="-e DISPLAY=$DISPLAY  -v /tmp/.X11-unix:/tmp/.X11-unix:rw --env QT_X11_NO_MITSHM=1"
    xhost +local:root
  fi;

  #run standand CMD set on the docker image with Xserver
  docker run \
      -v $(pwd)/gibson_data:/gibson_data \
      -v $(realpath gibson_dataset):/gibson_dataset \
      --gpus all \
      -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
      --privileged $X11_PARAMS -it \
      --rm \
      --name igibson \
      -it \
      ${DOCKER_NAME} \
      /bin/bash

  if [[ -e /tmp/.X11-unix && ! -z ${DISPLAY+x} ]]; then
      xhost -local:root
  fi;

else
  #run standand CMD set on the docker image
  docker run \
      -v $(pwd)/gibson_data:/gibson_data \
      -v $(realpath gibson_dataset):/gibson_dataset \
      --gpus all \
      -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
      --rm \
      --name igibson \
      ${DOCKER_NAME}
  fi
