MYTIME=$1

../rpg_trajectory_evaluation/scripts/dataset_tools/./bag_to_pose.py --output stamped_groundtruth.txt ../experiments/$MYTIME/$MYTIME.bag /gt_pose
../rpg_trajectory_evaluation/scripts/dataset_tools/./bag_to_pose.py --output stamped_traj_estimate.txt ../experiments/$MYTIME/$MYTIME.bag /odom_pose

python ../rpg_trajectory_evaluation/scripts/analyze_trajectory_single.py  ../experiments/$MYTIME/
