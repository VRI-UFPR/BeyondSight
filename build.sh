#docker build . --file Objectnav_DDPPO_baseline_basic.Dockerfile -t objectnav_base --rm

# #remove old image to avoid caching useless images
# docker rmi -f $(docker images -f "label=stage=builderddppo" -q)
# docker build . --file Objectnav_DDPPO_baseline.Dockerfile -t objectnav_submission_ddppo --rm

#docker image prune

# docker rmi -f $(docker images -f "label=stage=builder" -q)
# docker build . --file Objectnav.Dockerfile -t objectnav_submission2


# docker rmi -f $(docker images -f "label=stage=builder" -q)
# docker rmi -f $(docker images -f "label=stage=builderddppo" -q)
# docker rmi -f $(docker images -f "label=stage=builderbeyond" -q)

#remove old image to avoid caching useless images



#docker build . --file Objectnav_basic.Dockerfile -t objectnav_base --rm
docker rmi -f $(docker images -f "label=stage=builderBeyondPre" -q)
docker build . --file Objectnav_beyond_pre.Dockerfile -t objectnav_submission_beyond_pre --rm


docker rmi -f $(docker images -f "label=stage=builderBeyond" -q)
docker build . --file Objectnav_beyond.Dockerfile -t objectnav_submission_beyond --rm

# docker build . --file Objectnav_beyond.Dockerfile -t objectnav_submission_beyond2 --rm
