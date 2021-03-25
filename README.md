# BeyondSight
Code for the Master's thesis: ["Beyond sight: an approach for visual semantic navigation of mobile robots in an indoor environment"](https://www.researchgate.net/publication/350289808_Beyond_sight_an_approach_for_visual_semantic_navigation_of_mobile_robots_in_an_indoor_environment
)

The full text is publicly available at: https://www.researchgate.net/publication/350289808_Beyond_sight_an_approach_for_visual_semantic_navigation_of_mobile_robots_in_an_indoor_environment

# Disclaimer
The code here hosted still need some major clean up. It contains code from another repositories and original code made by me.

I divided the pertinent code per branches, **vribot_gazebo_2020**, **vribot_habitat_2020**, and **vribot_igibson_2020**. 
The **vribot_gazebo_2020** is structured to be akin to a ROS catkin workspace and compatible with Gazebo simulator, no docker or dataset is necessary.

For the habitat and igibson the code is structured to generate a Docker image with all necessary dependencies to run the experiments. Keep in mind that the module for handling GPUs of Docker is necessary. The datasets are not included in this repository nor it shall be included, for access to Gibson dataset refer to https://github.com/StanfordVL/GibsonEnv/blob/master/gibson/data/README.md and for access to Matterport3D refer to https://niessner.github.io/Matterport/ The COCO classes notations of https://github.com/StanfordVL/3DSceneGraph are necessary for semantics of Gibson.

For code related to YOLACT++ please refer to https://github.com/dbolya/yolact
For code related to Habitat https://github.com/facebookresearch/habitat-lab
For code related to iGibson https://github.com/StanfordVL/iGibson/

# Videos
* Early Experiments - Real World Zero https://youtu.be/ojXLEmJdIbk 
[![](http://img.youtube.com/vi/ojXLEmJdIbk/0.jpg)](http://www.youtube.com/watch?v=ojXLEmJdIbk "")

* Early Experiments - Real World One https://youtu.be/qXM1nOoHq1M 
[![](http://img.youtube.com/vi/qXM1nOoHq1M/0.jpg)](http://www.youtube.com/watch?v=qXM1nOoHq1M "")

* VRI's Lab Indoor Mapping https://youtu.be/txAJuoB2kUE 
[![](http://img.youtube.com/vi/txAJuoB2kUE/0.jpg)](http://www.youtube.com/watch?v=txAJuoB2kUE "")

* VRI's Lab Semantic Segmentation https://youtu.be/Nr8SjarAUA4 
[![](http://img.youtube.com/vi/Nr8SjarAUA4/0.jpg)](http://www.youtube.com/watch?v=Nr8SjarAUA4 "")

* Qualitative results of BEyond Complete https://youtu.be/qt5hHwG4ZAM 
[![](http://img.youtube.com/vi/qt5hHwG4ZAM/0.jpg)](http://www.youtube.com/watch?v=qt5hHwG4ZAM "")



