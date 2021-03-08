#!/home/dvruiz/miniconda3/envs/ros/bin/python
import argparse
import os
import rospy
from std_msgs.msg import Float32, Int64, Header
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import CameraInfo, PointCloud2
from sensor_msgs.msg import Image as ImageMsg
from nav_msgs.msg import Odometry
import rospkg
import numpy as np
from cv_bridge import CvBridge
import tf
from gibson2.envs.locomotor_env import NavigateEnv, NavigateRandomEnv
import matplotlib.pyplot as plt

import yaml
from sensor_msgs.msg import LaserScan
import cv2

from gibson2.core.render.profiler import Profiler

class SimNode:
    def __init__(self, gui_mode='headless'):
        rospy.init_node('gibson2_sim')
        rospack = rospkg.RosPack()
        path = rospack.get_path('gibson2-ros')

        config_filename = os.path.join(path, 'turtlebot_rgbd_mine.yaml')
        with open(config_filename) as f:
            self.config_file = yaml.load(f, Loader=yaml.FullLoader)

        self.cmdx = 0.0
        self.cmdy = 0.0

        self.image_pub = rospy.Publisher(
            "/gibson_ros/camera/rgb/image", ImageMsg, queue_size=10)
        self.depth_pub = rospy.Publisher(
            "/gibson_ros/camera/depth/image", ImageMsg, queue_size=10)

        self.lidar_pub = rospy.Publisher(
            "/gibson_ros/lidar/scan", LaserScan, queue_size=10)

        self.depth_raw_pub = rospy.Publisher("/gibson_ros/camera/depth/image_raw",
                                             ImageMsg,
                                             queue_size=10)

        self.odom_pub = rospy.Publisher("/odom", Odometry, queue_size=10)
        self.gt_odom_pub = rospy.Publisher(
            "/ground_truth_odom", Odometry, queue_size=10)

        self.camera_info_pub = rospy.Publisher("/gibson_ros/camera/depth/camera_info",
                                               CameraInfo,
                                               queue_size=10)
        self.bridge = CvBridge()
        self.br = tf.TransformBroadcaster()

        # self.env = NavigateEnv(config_file=config_filename,
        #                        mode=gui_mode,
        #                        action_timestep=1 / 30.0)    # assume a 30Hz simulation

        self.env = NavigateRandomEnv(config_file=config_filename,
                       mode=gui_mode,
                       action_timestep=1 / 30.0)    # assume a 30Hz simulation


        obs = self.env.reset()
        rospy.Subscriber("/mobile_base/commands/velocity",
                         Twist, self.cmd_callback)
        rospy.Subscriber("/reset_pose", PoseStamped, self.tp_robot_callback)

        self.tp_time = None

    @staticmethod
    def add_objects(env):
        from gibson2.core.physics.interactive_objects import ShapeNetObject
        # obj_path = '/cvgl/group/ShapeNetCore.v2/03001627/1b05971a4373c7d2463600025db2266/models/model_normalized.obj'
        obj_path = '/cvgl/group/ShapeNetCore.v2/03001627/60b3d70238246b3e408442c6701ebe92/models/model_normalized.obj'
        cur_obj = ShapeNetObject(obj_path,
                                 scale=1.0,
                                 position=[0, -2.0, 0.5],
                                 orientation=[0, 0, np.pi])
        env.simulator.import_object(cur_obj)

    def ros_loop(self):
        while not rospy.is_shutdown():
            obs, _, _, _ = self.env.step([self.cmdx, self.cmdy])

            rgb = (obs["rgb"] * 255).astype(np.uint8)
            image_message = self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8")

            depth = obs["depth"].astype(np.float32)
            depth_raw_image = (obs["depth"] * 1000 * self.config_file['depth_high']).astype(np.uint16)# mm to meters and depth_upper_range

            depth_raw_message = self.bridge.cv2_to_imgmsg(
                depth_raw_image, encoding="passthrough")
            depth_message = self.bridge.cv2_to_imgmsg(
                depth, encoding="passthrough")


            now = rospy.Time.now()

            image_message.header.stamp = now
            depth_message.header.stamp = now
            depth_raw_message.header.stamp = now
            image_message.header.frame_id = "camera_depth_optical_frame"
            depth_message.header.frame_id = "camera_depth_optical_frame"
            depth_raw_message.header.frame_id = "camera_depth_optical_frame"


            self.image_pub.publish(image_message)
            self.depth_pub.publish(depth_message)
            self.depth_raw_pub.publish(depth_raw_message)

            cam_y = self.config_file['image_height']
            cam_x = self.config_file['image_width']
            msg = CameraInfo(height=self.config_file['image_height'],
                             width=self.config_file['image_width'],
                             distortion_model="plumb_bob", D=[1e-08, 1e-08, 1e-08, 1e-08, 1e-08], K=[554.254691191187, 0.0, 320.5, 0.0, 554.254691191187, 240.5, 0.0, 0.0, 1.0], R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], P=[554.254691191187, 0.0, 320.5, -0.0, 0.0, 554.254691191187, 240.5, 0.0, 0.0, 0.0, 1.0, 0.0])

            msg.header.stamp = now
            msg.header.frame_id = "camera_depth_optical_frame"
            self.camera_info_pub.publish(msg)

            ###################################################################
            if ((self.tp_time is None) or ((self.tp_time is not None) and
                                           ((rospy.Time.now() - self.tp_time).to_sec() > 1.))):
                lidar_points = obs['scan']
                lidar_points = lidar_points.reshape(-1)
                lidar_header = Header()
                lidar_header.stamp = now
                lidar_header.frame_id = 'laser_frame'

                # num_readings = 228#1/3 of 0.52 angle resolution

                # print("DEBUG:",self.config_file)
                # 1/3 of 0.52 angle resolution
                num_readings = self.config_file['n_horizontal_rays']
                laser_frequency = 4  # ydlidar

                lidar_message = LaserScan()
                lidar_message.header = lidar_header
                lidar_message.angle_min = -1.57  # rad
                lidar_message.angle_max = 1.57  # rad
                lidar_message.angle_increment = 3.14 / num_readings  # rad
                lidar_message.time_increment = (
                    1 / laser_frequency) / (num_readings)  # sec
                lidar_message.range_min = self.config_file['min_laser_dist']
                lidar_message.range_max = self.config_file['laser_linear_range']

                lidar_message.ranges = lidar_points * lidar_message.range_max
                lidar_message.intensities = lidar_points

                self.lidar_pub.publish(lidar_message)
            ###################################################################

            # odometry
            self.env.robots[0].calc_state()

            odom = [
                np.array(self.env.robots[0].get_position()) -
                np.array(self.env.config["initial_pos"]),
                np.array(self.env.robots[0].get_rpy())
            ]

            self.br.sendTransform((odom[0][0], odom[0][1], 0),
                                  tf.transformations.quaternion_from_euler(
                                      0, 0, odom[-1][-1]),
                                  rospy.Time.now(), 'base_footprint', "odom")
            odom_msg = Odometry()
            odom_msg.header.stamp = rospy.Time.now()
            odom_msg.header.frame_id = 'odom'
            odom_msg.child_frame_id = 'base_footprint'

            odom_msg.pose.pose.position.x = odom[0][0]
            odom_msg.pose.pose.position.y = odom[0][1]
            odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, \
                odom_msg.pose.pose.orientation.w = tf.transformations.quaternion_from_euler(
                    0, 0, odom[-1][-1])

            odom_msg.twist.twist.linear.x = (self.cmdx + self.cmdy) * 5
            odom_msg.twist.twist.angular.z = (
                self.cmdy - self.cmdx) * 5 * 8.695652173913043
            self.odom_pub.publish(odom_msg)

            # Ground truth pose
            gt_odom_msg = Odometry()
            gt_odom_msg.header.stamp = rospy.Time.now()
            gt_odom_msg.header.frame_id = 'ground_truth_odom'
            gt_odom_msg.child_frame_id = 'base_footprint'

            xyz = self.env.robots[0].get_position()
            rpy = self.env.robots[0].get_rpy()

            gt_odom_msg.pose.pose.position.x = xyz[0]
            gt_odom_msg.pose.pose.position.y = xyz[1]
            gt_odom_msg.pose.pose.position.z = xyz[2]
            gt_odom_msg.pose.pose.orientation.x, gt_odom_msg.pose.pose.orientation.y, gt_odom_msg.pose.pose.orientation.z, \
                gt_odom_msg.pose.pose.orientation.w = tf.transformations.quaternion_from_euler(
                    rpy[0],
                    rpy[1],
                    rpy[2])

            gt_odom_msg.twist.twist.linear.x = (self.cmdx + self.cmdy) * 5
            gt_odom_msg.twist.twist.angular.z = (
                self.cmdy - self.cmdx) * 5 * 8.695652173913043
            self.gt_odom_pub.publish(gt_odom_msg)
        #when stops

    def run(self):
        # ros_loop()

        #Reinforcement learning logic 10 episodes of 100 actions
        for j in range(10):
        # for j in range(1):
            self.env.reset()
            # for i in range(100):
            for i in range(10):
                with Profiler('Env action step'):
                    # action = self.env.action_space.sample()
                    action = np.array([0.1,0.0])
                    print("action=",action)
                    state, reward, done, info = self.env.step(action)
                    # print("state=",state)
                    # print("reward=",reward)
                    # print("done=",done)
                    # print("info=",info)
                    if done:
                        print("Episode finished after {} timesteps".format(i + 1))
                        break


    def cmd_callback(self, data):
        self.cmdx = data.linear.x / 10.0 - \
            data.angular.z / (10 * 8.695652173913043)
        self.cmdy = data.linear.x / 10.0 + \
            data.angular.z / (10 * 8.695652173913043)

    def tp_robot_callback(self, data):
        rospy.loginfo('Teleporting robot')
        position = [data.pose.position.x,
                    data.pose.position.y, data.pose.position.z]
        orientation = [
            data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z,
            data.pose.orientation.w
        ]
        self.env.robots[0].reset_new_pose(position, orientation)
        self.tp_time = rospy.Time.now()


# import sys
if __name__ == '__main__':
    # gui_mode = rospy.get_param('~gui_mode')
    gui_mode = rospy.get_param('/vribot_gibson_sim/gui_mode')
    # print(gui_mode)

    # node = SimNode()
    node = SimNode(gui_mode)
    node.run()
