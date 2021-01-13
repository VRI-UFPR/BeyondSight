#!/bin/bash

path_package=$(roscd vribot_description; pwd)
echo $path_package

#rosrun xacro xacro.py ${path_package}/urdf/vribot.xacro > ${path_package}/vribot.urdf
