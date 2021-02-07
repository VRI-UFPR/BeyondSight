#!/usr/bin/env bash

DOCKER_NAME="objectnav_submission_ddppo"

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
    --legacy)
      shift
      LEGACY="${1}"
      shift
      ;;
    *) # unknown arg
      echo unkown arg ${1}
      exit
      ;;
esac
done

echo "CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}

if [ $LEGACY = 1 ]; then
docker run \
    -v $(pwd)/habitat-challenge-data:/habitat-challenge-data \
    -v $(realpath habitat-challenge-data/data/scene_datasets/gibson):/habitat-challenge-data/data/scene_datasets/gibson \
    -v $(realpath habitat-challenge-data/data/scene_datasets/mp3d):/habitat-challenge-data/data/scene_datasets/mp3d \
    --runtime nvidia \
    -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
    -e "AGENT_EVALUATION_TYPE=local" \
    -e "TRACK_CONFIG_FILE=/challenge_objectnav2020.local.rgbd.yaml" \
    --rm \
    --name habitat-challenge2 \
    ${DOCKER_NAME}
elif [ $LEGACY = 2 ]; then

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
    -e "TRACK_CONFIG_FILE=/challenge_objectnav2020.local.rgbd.yaml" \
    --privileged $X11_PARAMS -it \
    --rm \
    --name habitat-challenge \
    ${DOCKER_NAME}

if [[ -e /tmp/.X11-unix && ! -z ${DISPLAY+x} ]]; then
    xhost -local:root
fi;

else
docker run \
    -v $(pwd)/habitat-challenge-data:/habitat-challenge-data \
    -v $(realpath habitat-challenge-data/data/scene_datasets/gibson):/habitat-challenge-data/data/scene_datasets/gibson \
    -v $(realpath habitat-challenge-data/data/scene_datasets/mp3d):/habitat-challenge-data/data/scene_datasets/mp3d \
    --gpus all \
    -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
    -e "AGENT_EVALUATION_TYPE=local" \
    -e "TRACK_CONFIG_FILE=/challenge_objectnav2020.local.rgbd.yaml" \
    --rm \
    --name habitat-challenge \
    ${DOCKER_NAME}
fi
