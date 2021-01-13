FROM igibson_with_ros_and_beyond AS sim_image
LABEL stage=builderConfigIgibson

################################################################################
ADD vribot_ros_2021 /vribot
ADD iGibson_v101/gibson2/examples/ros/ /opt/igibson/gibson2/examples/ros/
ADD iGibson_v101/gibson2/global_config.yaml /opt/igibson/gibson2/global_config.yaml
ADD iGibson_v101/gibson2/robots/vribot_robot.py /opt/igibson/gibson2/robots/vribot_robot.py
ADD iGibson_v101/gibson2/envs/env_base.py /opt/igibson/gibson2/envs/env_base.py
ADD iGibson_v101/gibson2/scenes/indoor_scene.py /opt/igibson/gibson2/scenes/indoor_scene.py

RUN ln -sfn /opt/igibson /vribot/src/gibson2-ros
# RUN ln -sfn /vribot/src/vribot_description /gibson_dataset/assets/models/vribot/vribot_description
RUN echo "ln -sfn /vribot/src/vribot_description /gibson_dataset/assets/models/vribot/vribot_description" >> ~/.bashrc
RUN ln -sfn /opt/igibson/gibson2/examples/ros/gibson2-ros/beyond_agent /beyond_agent
################################################################################

#TODO: Remove once base image is updated.
# RUN /bin/bash -c ". activate py3-igibson; git clone --branch v0.1.5 http://github.com/facebookresearch/habitat-lab.git /habitat-lab;"
# #RUN /bin/bash -c ". activate habitat; mv habitat-api habitat-lab ; cp -r habitat-lab2/habitat_baselines habitat-lab/.; cp -r habitat-lab2/habitat/tasks habitat-lab/habitat/.;cp -r habitat-lab2/habitat/core/embodied_task.py habitat-lab/habitat/core/.; cp -r habitat-lab2/habitat/core/dataset.py habitat-lab/habitat/core/.; cp -r habitat-lab2/habitat/sims habitat-lab/habitat/."
# RUN /bin/bash -c ". activate py3-igibson ; cd /habitat-lab ; python setup.py develop build_ext --parallel 5 install; cd .." AS v0.1.5
################################################################################


#SETUP ENTRYPOINT
CMD ["/bin/bash", "-c", "source activate py3-igibson && sh"]
