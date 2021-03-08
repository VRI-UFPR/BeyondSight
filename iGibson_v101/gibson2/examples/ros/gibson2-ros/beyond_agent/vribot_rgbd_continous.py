#!/miniconda/envs/py3-igibson/bin/python
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

from gibson2.envs.igibson_env import iGibsonEnv

# from gibson2.envs.locomotor_env import NavigateEnv, NavigateRandomEnv
import matplotlib.pyplot as plt

import yaml
from sensor_msgs.msg import LaserScan
import cv2

import torch
# print("!!!!!!!!!!!!!!TORCH SUCCESSS!!!!!!!!!!!!!!!!!!!")
#import pybullet as p
#print(p.getEulerFromQuaternion([0.923399984836578, 0, 0.383830010890961, 0]))

###################################################################
# from beyond_agent.beyond_agent_without_habitat import BeyondAgent
# from beyond_agent.config_default import get_config
from beyond_agent_without_habitat import BeyondAgent
from config_default import get_config
###################################################################

class SimNode:
    def __init__(self, gui_mode='headless'):
        rospy.init_node('gibson2_sim')
        rospack = rospkg.RosPack()
        path = rospack.get_path('gibson2-ros')

        # config_filename = os.path.join(path, 'turtlebot_rgbd_mine.yaml')
        config_filename = os.path.join(path, '/beyond_agent/vribot_rgbd_micanopy.yaml')
        with open(config_filename) as f:
            self.config_file = yaml.load(f, Loader=yaml.FullLoader)

        self.cmdx = 0.0
        self.cmdy = 0.0

        self.image_pub = rospy.Publisher(
            "/gibson_ros/camera/rgb/image", ImageMsg, queue_size=10)
        self.depth_pub = rospy.Publisher(
            "/gibson_ros/camera/depth/image", ImageMsg, queue_size=10)


        # self.sseg_pub = rospy.Publisher(
        #     "/gibson_ros/camera/sseg/image", ImageMsg, queue_size=10)

        self.lidar_pub = rospy.Publisher(
            "/gibson_ros/lidar/scan", LaserScan, queue_size=10)
        # self.lidar_pub = rospy.Publisher("/gibson_ros/lidar/points", PointCloud2, queue_size=10)

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
        #                        mode='headless',
        #                        action_timestep=1 / 30.0)    # assume a 30Hz simulation

        # self.env = NavigateEnv(config_file=config_filename,
        #                        mode='gui',
        #                        action_timestep=1 / 30.0)    # assume a 30Hz simulation

        # render_to_tensor

        # self.env = iGibsonEnv(config_file=config_filename,
        #                        mode=gui_mode,
        #                        render_to_tensor=True,
        #                        action_timestep=1 / 30.0)    # assume a 30Hz simulation

        self.env = iGibsonEnv(config_file=config_filename,
                               mode=gui_mode,
                               render_to_tensor=True,
                               action_timestep=1 / 30.0,
                               physics_timestep=1 / 60.0)    # assume a 5Hz simulation

        # self.env = NavigateEnv(config_file=config_filename,
        #                        mode=gui_mode,
        #                        render_to_tensor=False,
        #                        action_timestep=1 / 30.0)    # assume a 30Hz simulation


        # self.env = NavigateRandomEnv(config_file=config_filename,
        #                        mode=gui_mode,
        #                        action_timestep=1 / 30.0)    # assume a 30Hz simulation

        # print(self.env.config)

        obs = self.env.reset()
        rospy.Subscriber("/mobile_base/commands/velocity",
                         Twist, self.cmd_callback)
        rospy.Subscriber("/reset_pose", PoseStamped, self.tp_robot_callback)

        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)


        self.tp_time = None

        # self.rgb_video = cv2.VideoWriter('video/rgb_video_stream.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (self.config_file['image_width'],self.config_file['image_height']))
        # self.rgb_video = cv2.VideoWriter('/home/dvruiz/pos/codeForFinalThesis/vribot/src/gibson2-ros/examples/ros/gibson2-ros/video/rgb_video_stream.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30.0, (self.config_file['image_width'],self.config_file['image_height']))
        # self.rgb_video = cv2.VideoWriter('/home/dvruiz/pos/codeForFinalThesis/vribot/src/gibson2-ros/examples/ros/gibson2-ros/video/rgb_video_stream.mp4', 0x00000021, 30.0, (self.config_file['image_width'],self.config_file['image_height']))

        # self.rgb_video = cv2.VideoWriter('/home/dvruiz/pos/codeForFinalThesis/vribot/src/gibson2-ros/examples/ros/gibson2-ros/video/rgb_video_stream.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (self.config_file['image_width'],self.config_file['image_height']))

        # self.add_objects(self.env)

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

    def run(self):
        global_goal = 0
        # perm = torch.LongTensor([2,1,0])
        print("will start")

        for key in self.env.robots[0].parts:
            print(  key, self.env.robots[0].parts[key].body_index, self.env.robots[0].parts[key].body_part_index )

        step_counter = 0
        ###################################################################
        '''
        BEyond INIT Logic here
        '''
        # config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
        fullpath = "/vribot/src/gibson2-ros/gibson2/examples/ros/gibson2-ros/beyond_agent/"
        config_paths = fullpath+"configs/challenge_objectnav2020.local.rgbd.yaml"
        config = get_config(
            [fullpath+"configs/beyond.yaml",fullpath+"configs/ddppo_pointnav.yaml"], ["BASE_TASK_CONFIG_PATH", config_paths]
        ).clone()

        agent = BeyondAgent(device=config.BEYOND.DEVICE, config=config, batch_size=1)
        agent.reset()

        print("objectgoal:", agent.m3pd_to_gibson_inverted[self.config_file['objectgoal_id']] )
        ###################################################################
        while not rospy.is_shutdown():
            obs, reward, done, info = self.env.step([self.cmdx, self.cmdy])
            # print("reward",reward,"done",done,"info",info)
            if(done):
                print("SUCCESS!")
                rospy.signal_shutdown("SUCCESS!")
                exit(0)
            step_counter=(step_counter+1)%30 #every 3 seconds
            # step_counter=(step_counter+1)%300 #every 3 seconds
            now = rospy.Time.now()
            ###################################################################
            '''
            BEyond STEP Logic here
            '''
            if(step_counter==0):
                # odometry
                self.env.robots[0].calc_state()

                # odom = [
                #     np.array(self.env.robots[0].get_position()) -
                #     np.array(self.env.config["initial_pos"]),
                #     np.array(self.env.robots[0].get_rpy())
                # ]
                #episodic location
                gps = np.array(self.env.robots[0].get_position()) - np.array(self.env.config["initial_pos"])
                orientation = np.array(self.env.robots[0].get_rpy()) - np.array(self.env.config["initial_orn"])
                #compass is independent from episode
                # orientation = np.array(self.env.robots[0].get_rpy() )
                #yaw angle
                compass = np.array([orientation[2]])

                obs['gps'] = torch.from_numpy(gps[0:2]).to("cuda")
                obs['compass'] = torch.from_numpy(compass).to("cuda")

                '''
                WARNING I AM HARDCODING A OBJECT GOAL FOR TESTING BED 6
                "couch": 5,
                "refrigerator": 3,
                '''
                # objectgoal = 3
                obs['objectgoal'] = torch.from_numpy(np.array([self.config_file['objectgoal_id']])).long().to("cuda")
                # print("obs['objectgoal']",objectgoal)

                pointgoal = agent.act(obs)
                #extract from batch
                pointgoal = pointgoal[0].cpu()
                print("pointgoal",pointgoal)
                '''
                So far target is in polar coordinates and without orientation to publish to ros must convert it
                to cartesian and set a orientation [rho,phi]
                '''
                pointgoal_x = pointgoal[0]*np.cos(pointgoal[1])
                pointgoal_y = pointgoal[0]*np.sin(pointgoal[1])

                '''
                the point goal is in relation to current position so we must update it to episodic coordinates
                '''
                pointgoal_x += gps[0]
                pointgoal_y += gps[1]

                goal_message = PoseStamped()
                goal_message.header.stamp = now
                goal_message.header.frame_id = "map"
                # goal_msg = pointgoal
                goal_message.pose.position.x = pointgoal_x
                goal_message.pose.position.y = pointgoal_y
                goal_message.pose.position.z = 0.0 #floor_height
                goal_message.pose.orientation.w = 1.0 #quaternion don't care x,y,z in this case due to framework
                self.goal_pub.publish(goal_message)
            ###################################################################
            #ros needs numpy arrays to broadcast
            if( torch.is_tensor(obs['rgb']) ):
                obs['rgb'] = torch.flip(obs['rgb'], [0])
                obs['rgb'] = obs['rgb'].byte().cpu().numpy()
                rgb = obs['rgb']
            else:
                rgb = (obs["rgb"] * 255).astype(np.uint8)
            # obs['rgb'] = obs['rgb'][:,:,perm].cpu().numpy()
            if( torch.is_tensor(obs['depth']) ):
                obs['depth'] = torch.flip(obs['depth'], [0])
                obs['depth'] = obs['depth'].cpu().numpy()
            ###################################################################



            # self.rgb_video.write(rgb)
            image_message = self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8")

            depth = obs["depth"].astype(np.float32)

            # depth_raw_image = (obs["depth"] * 100).astype(np.uint16)
            # depthInMeters = 1.0 / (rawDepth * -0.0030711016 + 3.3309495161)

            # depth_raw_image = (obs["depth"] * 1000).astype(np.uint16)

            # depth_raw_image = (obs["depth"] * 1000 * 4).astype(np.uint16)# to match ros scale and lidar
            depth_raw_image = (obs["depth"] * 1000 * self.config_file['depth_high']).astype(np.uint16)# mm to meters and depth_upper_range

            # depth_raw_image = (1.0 / (depth_raw_image * -0.0030711016 + 3.3309495161)).astype(np.uint16)


            depth_raw_message = self.bridge.cv2_to_imgmsg(
                depth_raw_image, encoding="passthrough")
            depth_message = self.bridge.cv2_to_imgmsg(
                depth, encoding="passthrough")

            # sseg = (obs["seg"] * 255).astype(np.uint8)
            # sseg_message = self.bridge.cv2_to_imgmsg(sseg, encoding="rgb8")

            # now = rospy.Time.now()

            image_message.header.stamp = now
            depth_message.header.stamp = now
            depth_raw_message.header.stamp = now
            image_message.header.frame_id = "camera_depth_optical_frame"
            depth_message.header.frame_id = "camera_depth_optical_frame"
            depth_raw_message.header.frame_id = "camera_depth_optical_frame"

            # sseg_message.header.stamp = now
            # sseg_message.header.frame_id = "camera_depth_optical_frame"

            # image_message.header.frame_id = "camera_rgb_frame"
            # depth_message.header.frame_id = "camera_rgb_frame"
            # depth_raw_message.header.frame_id = "camera_rgb_frame"

            # camera_rgb_frame

            self.image_pub.publish(image_message)
            self.depth_pub.publish(depth_message)
            self.depth_raw_pub.publish(depth_raw_message)

            '''
            Uncomment this block later!!!!
            '''
            # if(global_goal<1):
            #     goal_message = PoseStamped()
            #     goal_message.header.stamp = now
            #     goal_message.header.frame_id = "map"
            #     goal_msg = obs["task_obs"][:2]
            #     goal_message.pose.position.x = goal_msg[0]
            #     goal_message.pose.position.y = goal_msg[1]
            #     goal_message.pose.position.z = 0.0 #floor_height
            #     goal_message.pose.orientation.w = 1.0 #quaternion don't care x,y,z in this case due to framework
            #     self.goal_pub.publish(goal_message)
            #     global_goal = 1
            # else:
            #     self.goal_pub.publish(goal_message)


            # self.sseg_pub.publish(sseg_message)

            # msg = CameraInfo(height=256,
            #                  width=256,
            #                  distortion_model="plumb_bob",
            #                  D=[0.0, 0.0, 0.0, 0.0, 0.0],
            #                  K=[128, 0.0, 128, 0.0, 128, 128, 0.0, 0.0, 1.0],
            #                  R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            #                  P=[128, 0.0, 128, 0.0, 0.0, 128, 128, 0.0, 0.0, 0.0, 1.0, 0.0])

            # cam_y = self.config_file['image_height']
            # cam_x = self.config_file['image_width']
            # msg = CameraInfo(height=self.config_file['image_height'],
            #                  width=self.config_file['image_width'],
            #                  distortion_model="plumb_bob",
            #                  D=[0.0, 0.0, 0.0, 0.0, 0.0],
            #                  K=[cam_x, 0.0, cam_x, 0.0, cam_y, cam_y, 0.0, 0.0, 1.0],
            #                  R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            #                  P=[cam_x, 0.0, cam_x, 0.0, 0.0, cam_y, cam_y, 0.0, 0.0, 0.0, 1.0, 0.0])

            cam_y = self.config_file['image_height']
            cam_x = self.config_file['image_width']
            msg = CameraInfo(height=self.config_file['image_height'],
                             width=self.config_file['image_width'],
                             distortion_model="plumb_bob", D=[1e-08, 1e-08, 1e-08, 1e-08, 1e-08], K=[554.254691191187, 0.0, 320.5, 0.0, 554.254691191187, 240.5, 0.0, 0.0, 1.0], R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], P=[554.254691191187, 0.0, 320.5, -0.0, 0.0, 554.254691191187, 240.5, 0.0, 0.0, 0.0, 1.0, 0.0])

            # height: 480
            # width: 640
            # distortion_model: "plumb_bob"
            # D: [1e-08, 1e-08, 1e-08, 1e-08, 1e-08]
            # K: [554.254691191187, 0.0, 320.5, 0.0, 554.254691191187, 240.5, 0.0, 0.0, 1.0]
            # R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            # P: [554.254691191187, 0.0, 320.5, -0.0, 0.0, 554.254691191187, 240.5, 0.0, 0.0, 0.0, 1.0, 0.0]

            # msg = CameraInfo(height=480,
            #                  width=640,
            #                  distortion_model="plumb_bob",
            #                  D=[-2.6389095690190378e-01, 9.9983033880181316e-01, -
            #                      7.6323952014484080e-04, 5.0337278410637169e-03, -1.3056496956879815e+00],
            #                  K= [ 5.9421480358642339e+02, 0., 3.3930546187516956e+02, 0.,
            #                   5.9104092248505947e+02, 2.4273843891390746e+02, 0., 0., 1. ],
            #                  R=[ 1., 0., 0., 0., 1., 0., 0., 0., 1. ],
            #                  P=[ 5.94214355e+02, 0., 3.39307800e+02, 0., 0., 5.91040527e+02,
            #                   2.42739136e+02, 0., 0., 0., 1., 0. ])
            msg.header.stamp = now
            msg.header.frame_id = "camera_depth_optical_frame"
            self.camera_info_pub.publish(msg)

            ###################################################################
            # if ((self.tp_time is None) or ((self.tp_time is not None) and
            #                                ((rospy.Time.now() - self.tp_time).to_sec() > 1.))):
            #     lidar_points = obs['scan']
            #     lidar_header = Header()
            #     lidar_header.stamp = now
            #     lidar_header.frame_id = 'laser_frame'
            #
            #     #lidar msg must have the shape [ [x,y,z], ]
            #
            #     # lidar_msg = lidar_points.tolist(
            #     # print("DEBUG:",lidar_header,"DEBUG lidar_points:",lidar_points, lidar_points.tolist())
            #
            #     # lidar_message = pc2.create_cloud_xyz32(lidar_header, lidar_points.tolist())
            #     lidar_message = pc2.create_cloud_xyz32(lidar_header, lidar_points.tolist())
            #     self.lidar_pub.publish(lidar_message)

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

                # print("DEBUG lidar_points.shape:",lidar_points.shape)

                self.lidar_pub.publish(lidar_message)
            ###################################################################

            # odometry
            self.env.robots[0].calc_state()

            odom = [
                np.array(self.env.robots[0].get_position()) -
                np.array(self.env.config["initial_pos"]),
                np.array(self.env.robots[0].get_rpy() - np.array(self.env.config["initial_orn"]) )
            ]

            self.br.sendTransform((odom[0][0], odom[0][1], 0),
                                  tf.transformations.quaternion_from_euler(
                                      0, 0, odom[-1][-1]),
                                  rospy.Time.now(), 'base_footprint', "odom")
            odom_msg = Odometry()
            odom_msg.header.stamp = rospy.Time.now()
            odom_msg.header.frame_id = 'odom'
            odom_msg.child_frame_id = 'base_footprint'
            # odom_msg.child_frame_id = 'base_link'

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
            # gt_odom_msg.child_frame_id = 'base_link'

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
        # self.rgb_video.release()

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
    print(gui_mode)

    # node = SimNode()
    node = SimNode(gui_mode)
    node.run()
