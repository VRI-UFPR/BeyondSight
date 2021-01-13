#!/usr/bin/env python
import rospy
import numpy as np

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

initTime = rospy.Time.from_sec(0)
i=0
ids=0
lastPose = PoseStamped()
lastTime = rospy.Time.from_sec(0)
step = rospy.Duration.from_sec(np.float32(1./60.))

# ids=data.header.seq
# if((ids>78)and(ids<1319)):




def odom_cb(data):
    global path_pub, i, initTime, lastPose, lastTime,ids


    pose = PoseStamped()
    pose.header = data.header
    pose.header.stamp = pose.header.stamp - initTime
    pose.pose = data.pose.pose


    # if(ids==76):#last initial
    #     print(pose)
    # if(ids==77):
    #     print(pose)

    if((ids>75)and(ids<1315)):
        
        if(i>0):#init
            # print("TODO")
            diff=pose.header.stamp - lastTime
            diff=rospy.Duration(diff.secs,diff.nsecs)
            if( diff > step):#more than 1/60 of sec
                # print("TODO")
                for j in range(1,6):
                    i+= 1
                    lastPose.header.seq = i
                    lastPose.header.stamp = lastTime + j*rospy.Duration.from_sec(np.float32(1./60.))#repeat with a updated time
                    path_pub.publish(lastPose)
                i+= 1
                pose.header.stamp = lastTime + 6*rospy.Duration.from_sec(np.float32(1./60.))
                pose.header.seq = i
        else:
            initTime = pose.header.stamp
            pose.header.stamp = pose.header.stamp - initTime
            i = pose.header.seq

        lastPose = pose
        lastTime = pose.header.stamp
        path_pub.publish(pose)

    ids+=1




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

# print(step.secs,step.nsecs)

# odom_sub_gt = rospy.Subscriber('/vribot/base_pose_ground_truth', Odometry, odom_cb_gt)
# path_pub_gt = rospy.Publisher('/gt_pose', PoseStamped, queue_size=10)


if __name__ == '__main__':
    rospy.spin()
