#!/usr/bin/env bash

DOCKER_NAME="objectnav_submission_beyond"
CONTAINER_NAME="habitat-challenge"

while [[ $# -gt 0 ]]
do
  key="${1}"

  case $key in
      --docker-name)
        shift
        DOCKER_NAME="${1}"
  	    shift
        ;;
      --container-name)
        shift
        CONTAINER_NAME="${1}"
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
      --help)
        shift
        echo "mode 0=default CMD of image, mode=1=legacy gpu docker syntax, mode=2 gui mode, mode=3 open shell in image"
        exit
        shift
        ;;
      -h)
        shift
        echo "mode 0=default CMD of image, mode=1=legacy gpu docker syntax, mode=2 gui mode, mode=3 open shell in image"
        exit
        shift
        ;;
      *) # unknown arg
        echo unkown arg ${1}
        exit
        ;;
  esac
done

echo "CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}
#MODE code for old nvidia docker syntax
# if [ $MODE = 1 ]; then
#   docker run \
#       -v $(pwd)/habitat-challenge-data:/habitat-challenge-data \
#       -v $(realpath habitat-challenge-data/data/scene_datasets/gibson):/habitat-challenge-data/data/scene_datasets/gibson \
#       -v $(realpath habitat-challenge-data/data/scene_datasets/mp3d):/habitat-challenge-data/data/scene_datasets/mp3d \
#       --runtime nvidia \
#       -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
#       -e "AGENT_EVALUATION_TYPE=local" \
#       -e "TRACK_CONFIG_FILE=/challenge_objectnav2021.local.rgbd.yaml" \
#       --user $(id -u):$(id -g) \
#       --rm \
#       --name habitat-challenge \
#       ${DOCKER_NAME}


#MODE code for old nvidia docker syntax
if [ $MODE = 1 ]; then
  docker run \
      -v $(pwd)/habitat-challenge-data:/habitat-challenge-data \
      -v $(realpath habitat-challenge-data/data/scene_datasets/mp3d):/habitat-challenge-data/data/scene_datasets/mp3d \
      --runtime nvidia \
      -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
      -e "AGENT_EVALUATION_TYPE=local" \
      -e "TRACK_CONFIG_FILE=/challenge_objectnav2021.local.rgbd.yaml" \
      --rm \
      --name ${CONTAINER_NAME} \
      ${DOCKER_NAME}

#run with gui mode and new syntax
elif [ $MODE = 2 ]; then

  export QT_X11_NO_MITSHM=1

  X11_PARAMS=""
  if [[ -e /tmp/.X11-unix && ! -z ${DISPLAY+x} ]]; then
    echo "Using local X11 server"
    X11_PARAMS="-e DISPLAY=$DISPLAY  -v /tmp/.X11-unix:/tmp/.X11-unix:rw --env QT_X11_NO_MITSHM=1"
    xhost +local:root
  fi;

  docker run \
      -v $(pwd)/habitat-challenge-data:/habitat-challenge-data \
      -v $(realpath habitat-challenge-data/data/scene_datasets/gibson):/habitat-challenge-data/data/scene_datasets/gibson \
      -v $(realpath habitat-challenge-data/data/scene_datasets/mp3d):/habitat-challenge-data/data/scene_datasets/mp3d \
      --gpus all \
      -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
      -e "AGENT_EVALUATION_TYPE=local" \
      -e "TRACK_CONFIG_FILE=/challenge_objectnav2021.local.rgbd.yaml" \
      --privileged $X11_PARAMS -it \
      --rm \
      --name ${CONTAINER_NAME} \
      ${DOCKER_NAME}

  if [[ -e /tmp/.X11-unix && ! -z ${DISPLAY+x} ]]; then
      xhost -local:root
  fi;

elif [ $MODE = 3 ]; then
  #run a bash inside the container
  docker run \
      -v $(pwd)/habitat-challenge-data:/habitat-challenge-data \
      -v $(realpath habitat-challenge-data/data/scene_datasets/gibson):/habitat-challenge-data/data/scene_datasets/gibson \
      -v $(realpath habitat-challenge-data/data/scene_datasets/mp3d):/habitat-challenge-data/data/scene_datasets/mp3d \
      --gpus all \
      -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
      -e "AGENT_EVALUATION_TYPE=local" \
      -e "TRACK_CONFIG_FILE=/challenge_objectnav2021.local.rgbd.yaml" \
      --name ${CONTAINER_NAME} \
      --rm \
      -it \
      ${DOCKER_NAME} \
      /bin/bash
elif [ $MODE = 4 ]; then

  export QT_X11_NO_MITSHM=1

  X11_PARAMS=""
  if [[ -e /tmp/.X11-unix && ! -z ${DISPLAY+x} ]]; then
    echo "Using local X11 server"
    X11_PARAMS="-e DISPLAY=$DISPLAY  -v /tmp/.X11-unix:/tmp/.X11-unix:rw --env QT_X11_NO_MITSHM=1"
    xhost +local:root
  fi;

  docker run \
      -v $(pwd)/habitat-challenge-data:/habitat-challenge-data \
      -v $(realpath habitat-challenge-data/data/scene_datasets/gibson):/habitat-challenge-data/data/scene_datasets/gibson \
      -v $(realpath habitat-challenge-data/data/scene_datasets/mp3d):/habitat-challenge-data/data/scene_datasets/mp3d \
      --gpus all \
      -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
      -e "AGENT_EVALUATION_TYPE=local" \
      -e "TRACK_CONFIG_FILE=/challenge_objectnav2021.local.rgbd.yaml" \
      --name ${CONTAINER_NAME} \
      --privileged $X11_PARAMS -it \
      --rm \
      -it \
      ${DOCKER_NAME} \
      /bin/bash

  if [[ -e /tmp/.X11-unix && ! -z ${DISPLAY+x} ]]; then
      xhost -local:root
  fi;

#MODE code for old nvidia docker syntax with bash
elif [ $MODE = 5 ]; then
  docker run \
      -v $(pwd)/habitat-challenge-data:/habitat-challenge-data \
      -v $(realpath habitat-challenge-data/data/scene_datasets/gibson):/habitat-challenge-data/data/scene_datasets/gibson \
      -v $(realpath habitat-challenge-data/data/scene_datasets/mp3d):/habitat-challenge-data/data/scene_datasets/mp3d \
      --ipc="host" \
      --runtime nvidia \
      -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
      -e "AGENT_EVALUATION_TYPE=local" \
      -e "TRACK_CONFIG_FILE=/challenge_objectnav2021.local.rgbd.yaml" \
      --name ${CONTAINER_NAME} \
      --rm \
      -it \
      ${DOCKER_NAME} \
      /bin/bash

#run with new syntax
else
  docker run \
      -v $(pwd)/habitat-challenge-data:/habitat-challenge-data \
      -v $(realpath habitat-challenge-data/data/scene_datasets/gibson):/habitat-challenge-data/data/scene_datasets/gibson \
      -v $(realpath habitat-challenge-data/data/scene_datasets/mp3d):/habitat-challenge-data/data/scene_datasets/mp3d \
      --gpus all \
      -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
      -e "AGENT_EVALUATION_TYPE=local" \
      -e "TRACK_CONFIG_FILE=/challenge_objectnav2021.local.rgbd.yaml" \
      --rm \
      --name ${CONTAINER_NAME} \
      ${DOCKER_NAME}
fi
