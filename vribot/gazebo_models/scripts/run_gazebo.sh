#!/bin/bash

sudo killall rosmaster
sudo killall gzserver
sudo killall gzclient

roslaunch vribot_gazebo vribot_world.launch
# roslaunch vribot_gazebo vribot_world.launch gui:=false
# roslaunch vribot_gazebo vribot_world_ramp.launch
