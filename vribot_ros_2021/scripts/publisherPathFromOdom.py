#!/usr/bin/env python
import rospy

from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

path = Path()

def odom_cb_predict(data):
    global path, path_pub
    path.header = data.header
    pose = PoseStamped()
    pose.header = data.header
    pose.pose = data.pose.pose
    path.poses.append(pose)
    path_pub.publish(path)

path_gt = Path()
def odom_cb_gt(data):
    global path_gt, path_gt_pub
    path_gt.header = data.header
    pose = PoseStamped()
    pose.header = data.header
    pose.pose = data.pose.pose
    path_gt.poses.append(pose)
    path_gt_pub.publish(path_gt)

rospy.init_node('path_node')

odom_sub = rospy.Subscriber('/odom', Odometry, odom_cb_predict)
path_pub = rospy.Publisher('/path_predict', Path, queue_size=10)

odom_gt_sub = rospy.Subscriber('/vribot/base_pose_ground_truth', Odometry, odom_cb_gt)
path_gt_pub = rospy.Publisher('/path_gt', Path, queue_size=10)

if __name__ == '__main__':
    rospy.spin()
