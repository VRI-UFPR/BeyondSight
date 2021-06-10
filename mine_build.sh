
######################################################
# HABITAT challenge_objectnav2021
# #remove stopped containers
# docker rm $(docker ps -a -q)

# #remove old image to avoid caching useless images
# docker rmi -f $(docker images -f "label=stage=builderBarebones" -q)
# docker build . --file Objectnav.Dockerfile -t objectnav_submission_beyond --rm
######################################################

#stage 0
docker rmi -f $(docker images -f "label=stage=builderBeyondStage2" -q)
docker rmi -f $(docker images -f "label=stage=builderBeyondStage1" -q)
docker rmi -f $(docker images -f "label=stage=builderBeyondStage0" -q)
docker build . --file mine_Objectnav_beyond_stage0.Dockerfile -t mine_objectnav_beyond_stage0 --rm
#
# stage 1
docker rmi -f $(docker images -f "label=stage=builderBeyondStage2" -q)
docker rmi -f $(docker images -f "label=stage=builderBeyondStage1" -q)
docker build . --file mine_Objectnav_beyond_stage1.Dockerfile -t mine_objectnav_beyond_stage1 --rm

# stage 2
docker rmi -f $(docker images -f "label=stage=builderBeyondStage2" -q)
docker build . --file mine_Objectnav_beyond_stage2.Dockerfile -t objectnav_submission_beyond_v003 --rm
