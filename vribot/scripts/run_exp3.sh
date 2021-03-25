#!/bin/bash

sudo killall rosmaster
sudo killall gzserver
sudo killall gzclient

roslaunch vribot_gazebo exp3.launch
