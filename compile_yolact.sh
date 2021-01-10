#!/usr/bin/env bash

#SETUP
# ln -sfn /opt/igibson /vribot/src/gibson2-ros
# ln -sfn /vribot/src/vribot_description /gibson_dataset/assets/models/vribot/vribot_description
ln -sfn /opt/igibson/gibson2/examples/ros/gibson2-ros/beyond_agent /beyond_agent
#python -m gibson2.utils.assets_utils --change_data_path
####

###############################################
#UNCOMMENT TO COMPILE
#sanity check
if [ -d "/gibson_data/DCNv2" ]; then
  echo "DCNv2 copying then running setup"
  cp -r /gibson_data/DCNv2 /beyond_agent/yolact/external/;
  cd /beyond_agent/yolact/external/DCNv2; export CUDA_HOME=/usr/local/cuda-11.0; python setup.py build develop >/dev/null 2>&1; cd /
else
  echo "compiling DCNv2 for the first time"
  cd /beyond_agent/yolact/external/DCNv2; export CUDA_HOME=/usr/local/cuda-11.0; python setup.py build develop >/dev/null 2>&1; cp -r /beyond_agent/yolact/external/DCNv2 /gibson_data/ ; cd /
fi

echo "compiling DCNv2"
cd /beyond_agent/yolact/external/DCNv2; export CUDA_HOME=/usr/local/cuda-11.0; python setup.py build develop >/dev/null 2>&1; cd /
###############################################
