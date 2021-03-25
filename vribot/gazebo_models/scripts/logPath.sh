# rosbag record -O session2_090210.bag /odom_pose /gt_pose
#python publisherOdomToPose.py &
rosbag record -O $1 /odom_pose /gt_pose
