# docker rmi -f $(docker images -f "label=stage=buildingWithRos" -q)
#docker build . --file igibson_with_ros_and_beyond.Dockerfile -t igibson_with_ros_and_beyond --rm
docker rmi -f $(docker images -f "label=stage=builderConfigIgibson" -q)
docker build . --file igibson_config_and_run.Dockerfile -t igibson_config_and_run_with_ros_and_beyond --rm
