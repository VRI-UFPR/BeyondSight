bash build.sh
# nohup ./test_locally_objectnav_rgbd_gibson.sh --docker-name objectnav_submission_ddppo --cuda_visible_devices 1 --legacy 1 &

# ./test_locally_objectnav_rgbd_gibson.sh --docker-name objectnav_submission_ddppo --cuda_visible_devices 0 --legacy 0
# ./test_locally_objectnav_rgbd_gibson.sh --docker-name objectnav_submission_ddppo --cuda_visible_devices 0 --legacy 2

# nohup ./test_locally_objectnav_rgbd_gibson.sh --docker-name objectnav_submission_beyond --cuda_visible_devices 0 --legacy 1 &
# nohup ./test_locally_objectnav_rgbd_gibson.sh --docker-name objectnav_submission_beyond2 --cuda_visible_devices 0 --legacy 1 &
# nohup ./test_locally_objectnav_rgbd_gibson.sh --docker-name objectnav_submission_beyond --cuda_visible_devices 1 --legacy 1 &
# ./test_locally_objectnav_rgbd_gibson.sh --docker-name objectnav_submission_beyond --cuda_visible_devices 1 --legacy 1

./test_locally_objectnav_rgbd_gibson.sh --docker-name objectnav_submission_beyond --cuda_visible_devices 0 --legacy 0
