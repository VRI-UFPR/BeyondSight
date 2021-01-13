#!/usr/bin/env python
import rospy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

def odom_cb(data):
    global path_pub
    pose = PoseStamped()
    pose.header = data.header
    pose.pose = data.pose.pose
    path_pub.publish(pose)

def odom_cb_gt(data):
    global path_pub_gt
    pose = PoseStamped()
    pose.header = data.header
    pose.pose = data.pose.pose
    path_pub_gt.publish(pose)

rospy.init_node('log_path_node')

# odom_sub = rospy.Subscriber('/odom', Odometry, odom_cb)
odom_sub = rospy.Subscriber('/RosAria/pose', Odometry, odom_cb)
# odom_sub = rospy.Subscriber('/scanmatch_odom', Odometry, odom_cb)
path_pub = rospy.Publisher('/odom_pose', PoseStamped, queue_size=10)

# odom_sub_gt = rospy.Subscriber('/vribot/base_pose_ground_truth', Odometry, odom_cb_gt)
# path_pub_gt = rospy.Publisher('/gt_pose', PoseStamped, queue_size=10)


if __name__ == '__main__':
    rospy.spin()
