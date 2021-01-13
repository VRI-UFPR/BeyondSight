import gym
import numpy as np

from gibson2.robots.robot_locomotor import LocomotorRobot

# class VRIBot(LocomotorRobot):
#     """
#     VRIBot robot
#     Uses differentiable_drive / twist command control
#     """
#
#     def __init__(self, config):
#         self.config = config
#         # https://www.trossenrobotics.com/locobot-pyrobot-ros-rover.aspx
#         # Maximum translational velocity: 70 cm/s
#         # Maximum rotational velocity: 180 deg/s (>110 deg/s gyro performance will degrade)
#         self.linear_velocity = config.get('linear_velocity', 0.5)
#         self.angular_velocity = config.get('angular_velocity', np.pi / 2.0)
#         self.wheel_dim = 2
#         # self.wheel_axle_half = 0.115  # half of the distance between the wheels
#         # self.wheel_radius = 0.038  # radius of the wheels
#         LocomotorRobot.__init__(self,
#                                 "vribot/vribot.urdf",
#                                 base_name="base_link",
#                                 action_dim=self.wheel_dim,
#                                 scale=config.get("robot_scale", 1.0),
#                                 is_discrete=config.get("is_discrete", False),
#                                 control="differential_drive")
#
#     def set_up_continuous_action_space(self):
#         """
#         Set up continuous action space
#         """
#         self.action_high = np.zeros(self.wheel_dim)
#         self.action_high[0] = self.linear_velocity
#         self.action_high[1] = self.angular_velocity
#         self.action_low = -self.action_high
#         self.action_space = gym.spaces.Box(shape=(self.action_dim,),
#                                            low=-1.0,
#                                            high=1.0,
#                                            dtype=np.float32)
#
#     def set_up_discrete_action_space(self):
#         """
#         Set up discrete action space
#         """
#         assert False, "Locobot does not support discrete actions"
#
#     def get_end_effector_position(self):
#         """
#         Get end-effector position
#         """
#         return self.parts['gripper_link'].get_position()

###########################################
class VRIBot(LocomotorRobot):
    def __init__(self, config):
        self.config = config
        self.velocity = config.get("velocity", 1.0)
        LocomotorRobot.__init__(self,
                                "vribot/vribot.urdf",
                                action_dim=2,
                                scale=config.get("robot_scale", 1.0),
                                is_discrete=config.get("is_discrete", False),
                                control="velocity")

    def set_up_continuous_action_space(self):
        """
        Set up continuous action space
        """
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)
        self.action_high = self.velocity * np.ones([self.action_dim])
        self.action_low = -self.action_high

    def set_up_discrete_action_space(self):
        """
        Set up discrete action space
        """
        self.action_list = [[self.velocity, self.velocity], [-self.velocity, -self.velocity],
                            [self.velocity * 0.5, -self.velocity * 0.5],
                            [-self.velocity * 0.5, self.velocity * 0.5], [0, 0]]
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.setup_keys_to_action()

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'),): 0,  # forward
            (ord('s'),): 1,  # backward
            (ord('d'),): 2,  # turn right
            (ord('a'),): 3,  # turn left
            (): 4  # stay still
        }
        
    def apply_robot_action(self, action):
        """
        Apply robot action.
        Support joint torque, velocity, position control and
        differentiable drive / twist command control

        :param action: robot action
        """
        if self.control == 'torque':
            for n, j in enumerate(self.ordered_joints):
                j.set_motor_torque(
                    self.torque_coef * j.max_torque * float(np.clip(action[n], -1, +1)))
        # elif self.control == 'velocity':
        #     for n, j in enumerate(self.ordered_joints):
        #         j.set_motor_velocity(
        #             self.velocity_coef * j.max_velocity * float(np.clip(action[n], -1, +1)))
        ########################################################################
        elif self.control == 'velocity':
            for n, j in enumerate(self.ordered_joints[:2]):
                j.set_motor_velocity(self.velocity_coef * j.max_velocity * float(np.clip(action[n], -1, +1)))
        ########################################################################
        elif self.control == 'position':
            for n, j in enumerate(self.ordered_joints):
                j.set_motor_position(action[n])
        elif self.control == 'differential_drive':
            # assume self.ordered_joints = [left_wheel, right_wheel]
            assert action.shape[0] == 2 and len(
                self.ordered_joints) == 2, 'differential drive requires the first two joints to be two wheels'
            lin_vel, ang_vel = action
            if not hasattr(self, 'wheel_axle_half') or not hasattr(self, 'wheel_radius'):
                raise Exception(
                    'Trying to use differential drive, but wheel_axle_half and wheel_radius are not specified.')
            left_wheel_ang_vel = (lin_vel - ang_vel *
                                  self.wheel_axle_half) / self.wheel_radius
            right_wheel_ang_vel = (lin_vel + ang_vel *
                                   self.wheel_axle_half) / self.wheel_radius
            self.ordered_joints[0].set_motor_velocity(left_wheel_ang_vel)
            self.ordered_joints[1].set_motor_velocity(right_wheel_ang_vel)
        elif type(self.control) is list or type(self.control) is tuple:
            # if control is a tuple, set different control type for each joint
            if 'differential_drive' in self.control:
                # Assume the first two joints are wheels using differntiable drive control, and the rest use joint control
                # assume self.ordered_joints = [left_wheel, right_wheel, joint_1, joint_2, ...]
                assert action.shape[0] >= 2 and len(
                    self.ordered_joints) >= 2, 'differential drive requires the first two joints to be two wheels'
                assert self.control[0] == self.control[1] == 'differential_drive', 'differential drive requires the first two joints to be two wheels'
                lin_vel, ang_vel = action[:2]
                if not hasattr(self, 'wheel_axle_half') or not hasattr(self, 'wheel_radius'):
                    raise Exception(
                        'Trying to use differential drive, but wheel_axle_half and wheel_radius are not specified.')
                left_wheel_ang_vel = (
                    lin_vel - ang_vel * self.wheel_axle_half) / self.wheel_radius
                right_wheel_ang_vel = (
                    lin_vel + ang_vel * self.wheel_axle_half) / self.wheel_radius
                self.ordered_joints[0].set_motor_velocity(left_wheel_ang_vel)
                self.ordered_joints[1].set_motor_velocity(right_wheel_ang_vel)

            for n, j in enumerate(self.ordered_joints):
                if self.control[n] == 'torque':
                    j.set_motor_torque(
                        self.torque_coef * j.max_torque * float(np.clip(action[n], -1, +1)))
                elif self.control[n] == 'velocity':
                    j.set_motor_velocity(
                        self.velocity_coef * j.max_velocity * float(np.clip(action[n], -1, +1)))
                elif self.control[n] == 'position':
                    j.set_motor_position(action[n])
        else:
            raise Exception('unknown control type: {}'.format(self.control))
###########################################
