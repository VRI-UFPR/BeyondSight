FROM igibson_with_ros_and_beyond AS sim_image
LABEL stage=builderConfigIgibson



# apt-get install -y ros-noetic-global-planner libcublas-dev-11-0 libcufft-dev-11-0 libcurand-dev-11-0 libcusolver-dev-11-0 libcusparse-dev-11-0
################################################################################
ADD vribot_ros_2021 /vribot
ADD iGibson_v101/gibson2/examples/ros/ /opt/igibson/gibson2/examples/ros/
ADD iGibson_v101/gibson2/global_config.yaml /opt/igibson/gibson2/global_config.yaml
ADD iGibson_v101/gibson2/robots/vribot_robot.py /opt/igibson/gibson2/robots/vribot_robot.py
ADD iGibson_v101/gibson2/envs/env_base.py /opt/igibson/gibson2/envs/env_base.py
ADD iGibson_v101/gibson2/scenes/indoor_scene.py /opt/igibson/gibson2/scenes/indoor_scene.py

RUN ln -sfn /opt/igibson /vribot/src/gibson2-ros
# RUN ln -sfn /vribot/src/vribot_description /gibson_dataset/assets/models/vribot/vribot_description
RUN echo "ln -sfn /vribot/src/vribot_description /gibson_dataset/assets/models/vribot/vribot_descriptionh" >> ~/.bashrc
RUN ln -sfn /opt/igibson/gibson2/examples/ros/gibson2-ros/beyond_agent /beyond_agent
################################################################################

#SETUP DATASET PATH
# pip uninstall pybullet
# pip install https://github.com/StanfordVL/bullet3/archive/master.zip
# RUN /bin/bash -c ". activate py3-igibson; python -m gibson2.utils.assets_utils --change_data_path";

#assets
#/gibson_dataset/assets
#dataset
#/gibson_dataset/dataset
#other don't care

#SETUP VRIBOT SYM LINK
#RUN ln -sfn /vribot/src/vribot_description /gibson_dataset/assets/models/vribot_description
#RUN ln -sfn /opt/igibson/gibson2/examples/ros/gibson2-ros/beyond_agent /beyond_agent
#ADD compile_yolact.sh /compile_yolact.sh

# pip install PyQt5 PySide2

#SETUP ENTRYPOINT
CMD ["/bin/bash", "-c", "source activate py3-igibson && sh"]
