import torch
import numpy as np
from gym.spaces import Box, Discrete

from beyond_agent import BeyondAgent, DDPPOAgent_PointNav
from beyond_agent import batch_obs

from actor_critic import ActorCriticPolicy
import mapper

# from gym_env_semantic_map import rotate_image

# import torchvision.transforms.functional as TF
# from PIL import Image
# def rotate_image(image, angle):
#     result = TF.rotate(img=image, angle=angle, resample=Image.NEAREST)
#     return result

import cv2
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST)
  return result

# def RMSELoss(yhat,y):
#     return torch.sqrt(torch.mean((yhat-y)**2))

def RMSELoss(yhat,y):
    return np.sqrt(np.mean((yhat-y)**2))

class HeuristLocalPlanner():
    def __init__(self, config, batch_size):
        print("HeuristLocalPlanner")
        self.batch_size = batch_size
        self.angle_in_deg = config.TASK_CONFIG.SIMULATOR.TURN_ANGLE

    def reset_batch(self):
        pass
    def reset_idx(self,idx):
        pass
    def eval(self):
        pass
    def act(self, observations):
        #placeholder
        # action = torch.from_numpy(np.array([0]))
        # action = torch.from_numpy(np.array([1]))
        # action = torch.from_numpy(np.array([2]))
        # action = torch.from_numpy(np.array([3]))

        angle = np.deg2rad(self.angle_in_deg)

        #rho, phi
        # print("observations['pointgoal_with_gps_compass']",observations['pointgoal_with_gps_compass'])

        #necessary for train
        # observations['pointgoal_with_gps_compass'] = observations['pointgoal_with_gps_compass'].squeeze(0)

        #test if agent is close enough on the map to observed target then force stop
        #-2 is current location
        # if(observations['map'][-2] ):
        #     pass

        # print("observations['pointgoal_with_gps_compass']",observations['pointgoal_with_gps_compass'])

        actions = []
        for env_idx in range(self.batch_size):
            if( observations['pointgoal_with_gps_compass'][env_idx][0] <= 0.25 ):
            # if( observations['pointgoal_with_gps_compass'][env_idx][0] < 0.15 ):
                # action = torch.from_numpy(np.array([0]))
                # action = 0
                action = 1
            else:
                if( observations['pointgoal_with_gps_compass'][env_idx][1] > angle ):
                    # action = torch.from_numpy(np.array([2]))
                    action = 2
                else:
                    if( observations['pointgoal_with_gps_compass'][env_idx][1] < -angle ):
                        # action = torch.from_numpy(np.array([3]))
                        action = 3
                    else:
                        #dont move 1 if make it farther
                        # action = torch.from_numpy(np.array([1]))
                        action = 1
            actions.append(action)

        actions = torch.from_numpy(np.array(actions)).unsqueeze(-1)

        return actions
    ############################################################################
    # def act(self, observations):
    #     '''
    #     esc = 27
    #     Upkey : 2490368
    #     DownKey : 2621440
    #     LeftKey : 2424832
    #     RightKey: 2555904
    #     '''
    #
    #     k = cv2.waitKey()
    #     if k==ord('w'):#  up
    #         action=1
    #     elif k==ord('a'):#  left
    #         action=2
    #     elif k==ord('d'):#  right
    #         action=3
    #     elif k==ord('s'):# esc
    #         action=0
    #     else:
    #         action=0
    #
    #     action = torch.from_numpy(np.array([action]))
    #     return action

class BeyondAgentWithoutInternalMapper(BeyondAgent):
    def __init__(self, device, config, batch_size, is_batch_internal=False):
        # super(BeyondAgentWithoutInternalMapper, self).__init__(device, config, batch_size, is_batch_internal)

        self.device=device
        self.config=config
        self.batch_size=batch_size
        self.ep_iteration=torch.zeros(self.batch_size)
        ########################################################################
        '''
        Load DDPPO with handle local policy
        '''
        # self.planner = DDPPOAgent_PointNav(config,self.batch_size)
        self.planner = HeuristLocalPlanner(config,self.batch_size)
        self.planner.reset_batch()
        ########################################################################

        ########################################################################
        '''
        Load Global_Policy
        '''
        self.action_space = Box( np.array([0.0,0.0],dtype=np.float32), np.array([1.0,1.0],dtype=np.float32), dtype=np.float32)
        self.actor_critic = ActorCriticPolicy(
            action_space=self.action_space,
            use_sde=False,
            device=self.device,
            config=self.config,
        )
        self.actor_critic.to(self.device)
        # ######################################################################

        if(self.config.BEYOND.GLOBAL_POLICY.LOAD_CHECKPOINT):
            '''
            load policy
            '''
            print("loading",self.config.BEYOND.GLOBAL_POLICY.CHECKPOINT)
            ckpt_dict = torch.load(self.config.BEYOND.GLOBAL_POLICY.CHECKPOINT, map_location="cpu")
            self.actor_critic.load_actor_critic_weights(ckpt_dict["state_dict"])
            del ckpt_dict
        # ######################################################################

        ########################################################################
        '''
        Initialize internal map per env
        '''
        # self.recreate_map_array(self.batch_size)

        self.not_done_masks = torch.zeros(
            self.batch_size, 1, device=self.device, dtype=torch.bool
        )
        self.test_recurrent_hidden_states = torch.zeros(
            self.batch_size,
            self.actor_critic.feature_net.rnn_layers,
            self.actor_critic.feature_net.feature_out_size,
            device=self.device,
        )
        self.y_pred_map_coo_local = np.zeros((self.batch_size,2))
        self.y_pred_map_coo_long = np.zeros((self.batch_size,2))
        ########################################################################

        print("Agent Loaded",flush=True)
    ############################################################################
    ############################################################################
    def compute_pointgoal_with_pred(self,observations, y_pred_reprojected_goal,env_idx):
        '''
        CURRENTLY THERE IS A ISSUE WITH CENTERED CROPED PointGoal COMPUTATION
        '''
        ###############################################
        '''
        Convert the point goal prediction to pointgoal_with_gps_compass

        Sensor that integrates PointGoals observations (which are used PointGoal Navigation) and GPS+Compass.
        For the agent in simulator the forward direction is along negative-z.
        In polar coordinate format the angle returned is azimuth to the goal.
        '''

        '''
        I just need to convert from cartesian_to_polar
        '''

        '''
        This is important gps is in episodic -zx notation

        pointgoal_with_gps_compass should be in world xyz, we will attempt to use episodic xyz
        '''

        ########################################################################
        # '''
        # this block only works if the agent is not centered_cropped
        # '''
        # #so -zx to xyz, y=0 since it is episodic,( -x, 0, -z as in pose? )
        # position = observations['gps'].cpu()
        # # position = position[env_idx]+self.mapper_wrapper_array[env_idx].shift_origin_gps
        # position = position[env_idx]
        # agent_position = torch.zeros(3)
        # agent_position[0] = -position[1]
        # agent_position[2] = position[0]
        ########################################################################
        '''
        this block is to centered_cropped
        '''
        agent_position = torch.zeros(3)

        #prediction is in episodic ZX meters so do the same
        goal_position = torch.zeros(3)
        goal_position[0] = y_pred_reprojected_goal[1]
        goal_position[2] = y_pred_reprojected_goal[0]

        # print("agent_position",agent_position)
        # print("goal_position",goal_position)

        '''
        Seems correct so far, I still unsure about the sign of the coordinates, needs further # DEBUG:
        '''

        '''
        Here this is important as well, compass represent the agent angle. The angle is 0 at state t=0.
        rotation_world_agent should be a quaternion representing the the true rotation betwen the agent
        and the world. Since we will attempt to use episodic coo instead of world coo.

        We will adapt this. We will convert the angle to a quaternion
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)
        '''

        q_r,q_i,q_j,q_k = mapper.angle_to_quaternion_coeff(observations['compass'][env_idx].cpu(), [0.,1.,0.])
        rotation_world_agent = np.quaternion(q_r,q_i,q_j,q_k)

        pointgoal_with_gps_compass = torch.from_numpy(mapper._compute_pointgoal(agent_position.detach().numpy(), rotation_world_agent, goal_position.detach().numpy()))
        #needs a better solution
        pointgoal_with_gps_compass[1] -= np.pi
        #ensure -pi,pi range
        if pointgoal_with_gps_compass[1] < -np.pi:
            pointgoal_with_gps_compass[1] =  (pointgoal_with_gps_compass[1] + np.pi) + np.pi

        observations['pointgoal_with_gps_compass'][env_idx]=pointgoal_with_gps_compass

        return observations
    ############################################################################
    def reset(self, env_idx=0):
        '''
        reset is per env
        '''
        self.ep_iteration[env_idx]=0
        self.planner.reset_idx(env_idx)
        self.not_done_masks[env_idx] = False
        self.test_recurrent_hidden_states[env_idx,:,:] = 0
        self.y_pred_map_coo_local = np.zeros((self.batch_size,2),dtype=np.int32)
        self.y_pred_map_coo_long = np.zeros((self.batch_size,2),dtype=np.int32)
    ############################################################################
    def act(self, observations):
        '''
        act should used for single env only, batch_size==1
        '''

        #test if the observations has a batch dim or not
        # print("observations",observations)
        shape = observations['objectgoal'].shape

        if(len(shape)>1):
            batch = observations
        else:
            batch = batch_obs([observations], device=self.device)

        value, action, action_log_probs, local_planner_actions, main_inputs, computed_reward, rnn_states = self.act_with_batch(batch)
        local_planner_actions = local_planner_actions.item()

        return local_planner_actions, computed_reward
    ############################################################################
    def update_pred(self,map,pred_local,pred_coo):
        for env_idx in range(self.batch_size):
            if(self.ep_iteration[env_idx]>0):
                map[env_idx][-1].fill_(0)
                map[env_idx, -1, pred_local[env_idx][0]-1:pred_local[env_idx][0]+2, pred_local[env_idx][1]-1:pred_local[env_idx][1]+2  ] = 0.5

                map[env_idx, -1, pred_coo[env_idx][0]-1:pred_coo[env_idx][0]+2, pred_coo[env_idx][1]-1:pred_coo[env_idx][1]+2  ] = 1.0
        return map
    ############################################################################
    def act_with_batch(self, observations, deterministic=False):
        with torch.no_grad():
            main_inputs = observations['map']
            main_inputs = self.update_pred(main_inputs,self.y_pred_map_coo_local,self.y_pred_map_coo_long)
            # main_inputs[-1] = self.y_pred_map_coo_local

            features, self.test_recurrent_hidden_states = self.actor_critic.feature_net(
                main_inputs, observations['objectgoal'], mapper.compass_to_orientation(observations['compass']), self.test_recurrent_hidden_states, self.not_done_masks
            )

            #  Make masks not done till reset (end of episode) will be called
            self.not_done_masks.fill_(True)

            value = self.actor_critic.critic(features)
            distribution = self.actor_critic._get_action_dist_from_latent(features)
            action = distribution.get_actions(deterministic=deterministic)

            action_log_probs = distribution.log_prob(action).unsqueeze(-1)

            observations['pointgoal_with_gps_compass'] = torch.zeros(self.batch_size,2,device=self.device)
            #ensure that the prediction stays on the [0,1] range
            y_pred_perc = torch.sigmoid(action)
            y_pred_perc = y_pred_perc.cpu()
            # print("y_pred_perc",y_pred_perc)

            # computed_reward = np.zeros((self.batch_size))
            # computed_reward = np.zeros((self.batch_size))
            computed_reward = [{'rmse':0,'y_pred_map_coo_long':[0,0]} for i in range(self.batch_size)]

            map_size=mapper.get_map_size_in_cells(12.8, 0.05) -1
            y_pred_map_coo = y_pred_perc*map_size

            for env_idx in range(self.batch_size):
                ################################################################
                '''
                post cnn
                '''
                #0 to 511
                '''
                local map shall be 0,255
                '''

                ####################################################################
                targets = torch.nonzero( main_inputs[env_idx][observations['objectgoal'][env_idx]+1].squeeze(0) ).cpu().numpy()
                #target shape is first dim is the point ,second dim is an array of coordinates, batch, coox, cooy
                if(targets.shape[0]>0):
                    # print("activate heuristic")
                    # print("main_inputs",main_inputs.shape)
                    # print("main_inputs2",main_inputs[env_idx].shape)
                    # print("main_inputs3",main_inputs[env_idx][observations['objectgoal'][env_idx]+1].shape)
                    # print("targets",targets)
                    # print("observations['objectgoal'][env_idx]+1",observations['objectgoal'][env_idx]+1)
                    diff_in_cells = np.array([128,128]) - targets
                    euclid_dist_in_cells = np.linalg.norm(diff_in_cells, ord=2, axis=-1)
                    closest_target_idx=np.argmin(euclid_dist_in_cells)
                    # print("euclid_dist_in_cells[closest_target_idx]",euclid_dist_in_cells[closest_target_idx])

                    y_pred_map_coo_long = targets[closest_target_idx]
                    # computed_reward[env_idx] = -1

                    '''
                    computed_reward should be the rmse between
                    '''

                    y_pred_map_coo_long_reward = torch.round(y_pred_map_coo[env_idx]).long().cpu()
                    if(y_pred_map_coo_long_reward[0] > map_size):
                        y_pred_map_coo_long_reward[0]=map_size
                    if(y_pred_map_coo_long_reward[1] > map_size):
                        y_pred_map_coo_long_reward[1]=map_size


                    # print("y_pred_map_coo_long_reward",y_pred_map_coo_long_reward,"y_pred_map_coo_long", y_pred_map_coo_long)
                    rmse = RMSELoss(y_pred_map_coo_long_reward.float().numpy(), y_pred_map_coo_long.astype(np.float32))

                    computed_reward[env_idx] = {'rmse':rmse,'y_pred_map_coo_long':y_pred_map_coo_long_reward}

                    # y_pred_map_coo_long = y_pred_map_coo_long_reward
                ####################################################################
                else:
                    # map_size=mapper.get_map_size_in_cells(self.mapper_wrapper_array[env_idx].map_size_meters/2, self.mapper_wrapper_array[env_idx].map_cell_size) -1
                    # map_size=mapper.get_map_size_in_cells(25.6, 0.1) -1

                    y_pred_map_coo_long = torch.round(y_pred_map_coo[env_idx]).long().cpu()
                    if(y_pred_map_coo_long[0] > map_size):
                        y_pred_map_coo_long[0]=map_size
                    if(y_pred_map_coo_long[1] > map_size):
                        y_pred_map_coo_long[1]=map_size

                    computed_reward[env_idx] = {'rmse':0.0,'y_pred_map_coo_long':y_pred_map_coo_long}
                    '''
                    update pred
                    '''
                    # pred_y_pred_map_coo_long = y_pred_map_coo_long.clone().detach()
                    # computed_reward[env_idx] = self.mapper_wrapper_array[env_idx].update_prev_pred_in_map(pred_y_pred_map_coo_long, observations['objectgoal'][env_idx][0])
                '''
                HERE WE WILL USE A* TO FIND THE CLOSEST FREE CELL WITHIN SENSORING
                RANGE THAT STEER TOWARDS THE PREDICTED OBJECTIVE,

                HAVING ALWAYS A CLOSE POINT GOAL FOR THE DDPPO PLANNER ENSURES A
                BETTER CONVERGENCE
                '''
                ################################################################
                '''
                for pred to be where we expect to make it correct
                '''
                #up
                # y_pred_map_coo_long[0] = 100
                # y_pred_map_coo_long[1] = 128
                ################################################################

                #22 is the map_aux[0] which is the occupied map
                src_pt_value = int(np.floor(main_inputs.shape[-1]/2))
                src_pt = (src_pt_value,src_pt_value)
                dst_pt = (y_pred_map_coo_long[0].item(),y_pred_map_coo_long[1].item())

                #main_inputs is centered cropped so the middle of the map is where the agent is
                grid = main_inputs[env_idx][22].clone().detach()
                y_pred_map_coo_local = self.get_free_cell_on_path(grid=grid.cpu().numpy(), src_pt=src_pt, dst_pt=dst_pt)
                if(y_pred_map_coo_local is None):
                    '''
                    could not find a path
                    '''
                    y_pred_map_coo_local = y_pred_map_coo_long

                # print("y_pred_map_coo_long",y_pred_map_coo_long)
                # print("y_pred_map_coo_local",y_pred_map_coo_local)
                self.y_pred_map_coo_local[env_idx] = y_pred_map_coo_local
                self.y_pred_map_coo_long[env_idx] = y_pred_map_coo_long
                # y_pred_reprojected_goal = mapper.map_to_2dworld(y_pred_map_coo_local, self.mapper_wrapper_array[env_idx].map_size_meters/2, self.mapper_wrapper_array[env_idx].map_cell_size)
                # y_pred_reprojected_goal = mapper.map_to_2dworld(y_pred_map_coo_local, 25.6, 0.1)

                y_pred_reprojected_goal = mapper.map_to_2dworld(y_pred_map_coo_local, 12.8, 0.05)
                # y_pred_reprojected_goal = mapper.map_to_2dworld(y_pred_map_coo_long, 12.8, 0.05)

                # print("y_pred_reprojected_goal",y_pred_reprojected_goal)
                observations = self.compute_pointgoal_with_pred(observations, y_pred_reprojected_goal,env_idx)
                ################################################################

            # # ########################################################################
            # # '''
            # # FOR # DEBUG:
            # # '''
            # # print("self.y_pred_map_coo_local",self.y_pred_map_coo_local)
            # # print("self.y_pred_map_coo_long",self.y_pred_map_coo_long)
            # main_inputs = self.update_pred(main_inputs,self.y_pred_map_coo_local,self.y_pred_map_coo_long)
            # # print("observations['pointgoal_with_gps_compass']",observations['pointgoal_with_gps_compass'])
            # trash = (((main_inputs[0][-4]+main_inputs[0][-3])*255)+(main_inputs[0][-1]*128)).byte().cpu().numpy()
            # # trash = trash+(main_inputs[0][-1]*128.byte().cpu().numpy())
            #
            # cv2.imshow("map-3", trash )
            # cv2.imshow("map-2", (((main_inputs[0][-4]+main_inputs[0][-2])*255)+(main_inputs[0][-1]*128)).byte().cpu().numpy() )
            #
            # arrow = cv2.imread("extras/arrow.png")
            # # arrow = rotate_image(torch.from_numpy(arrow), np.rad2deg(observations['compass'].item()) )
            # # print("arrow.shape",arrow.shape)
            # arrow = rotate_image(arrow, np.rad2deg(observations['compass'].item()) )
            # # cv2.imshow("arrow",arrow.numpy())
            # cv2.imshow("arrow",arrow)
            #
            # cv2.waitKey()
            # ########################################################################

            # computed_reward = self.y_pred_map_coo_long
            '''
            feed ddppo
            '''
            del observations['objectgoal']
            del observations['compass']
            del observations['gps']
            del observations['map']

            # print("observations",observations)
            local_planner_actions = self.planner.act(observations)

            #all envs
            self.ep_iteration+=1

            return value, action, action_log_probs, local_planner_actions, main_inputs, computed_reward, self.test_recurrent_hidden_states
    ############################################################################
    # def act_with_batch(self, observations, deterministic=False):
    #     with torch.no_grad():
    #         '''
    #         22 classes + 4 aux (occupied, past_locations, current_location, previous_prediction)
    #         global map is 51.2 meters with 0.1 cell size
    #         local_map is cropped 256,256 around the agent
    #         '''
    #         main_inputs = torch.zeros((self.batch_size,26,256,256),device=self.device)
    #         # phi = torch.zeros((self.batch_size,1),device=self.device).long()
    #         ####################################################################
    #         '''
    #             First call yolact and segment the rgb
    #         '''
    #         ####################################################################
    #         # y_pred_scores = self.evalimage(observations['rgb'])
    #         y_pred_scores = torch.zeros((self.batch_size,480,640,22),device=self.device)
    #         y_pred_scores[:,:,0]=1.0
    #         # print(y_pred_scores)
    #         # print("y_pred_scores",y_pred_scores.shape)
    #
    #         for env_idx in range(self.batch_size):
    #             '''
    #                 pre cnn
    #             '''
    #             ################################################################
    #             '''
    #                 First call yolact and segment the rgb
    #             '''
    #             ################################################################
    #             # print("observations['semantic']",observations['semantic'].shape)
    #             # print("observations['semantic'][env_idx]",observations['semantic'][env_idx].shape)
    #
    #             # y_pred_scores = self.split_into_channels(observations['semantic'][env_idx])
    #             # print("y_pred_scores",y_pred_scores.shape)
    #             # exit()
    #             ################################################################
    #             '''
    #                 Resize depth from [0,1] to the actual distance for correct projection
    #             '''
    #             resized_depth = observations['depth'][env_idx]
    #             resized_depth = (resized_depth.squeeze(-1)*(self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH-self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH))+self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
    #             ################################################################
    #             '''
    #                 Then do the map projection
    #             '''
    #             ################################################################
    #             self.mapper_wrapper_array[env_idx].update_current_step(self.ep_iteration[env_idx])
    #             self.mapper_wrapper_array[env_idx].update_pose(observations['compass'][env_idx].cpu(),observations['gps'][env_idx].cpu())
    #             #sseg yolact
    #             main_inputs[env_idx] = self.mapper_wrapper_array[env_idx].update_map(resized_depth, y_pred_scores[env_idx])
    #             #sseg gt
    #             # main_inputs[env_idx] = self.mapper_wrapper_array[env_idx].update_map(resized_depth, y_pred_scores)
    #
    #             # trash = (main_inputs[env_idx][-2]*255).byte().cpu().numpy()
    #             print(torch.nonzero(main_inputs[0][0]))
    #             print("resized_depth",resized_depth)
    #             trash = (main_inputs[env_idx][0]*255).byte().cpu().numpy()
    #             # print(trash.shape)
    #             cv2.imshow("map",trash)
    #             cv2.waitKey()
    #             print("EXIT")
    #             exit()
    #
    #             # phi[env_idx] =  int(np.floor((self.mapper_wrapper_array[env_idx].orientation*360)/30))
    #             ################################################################
    #
    #         #end for
    #         # print("main_inputs",main_inputs.shape,main_inputs)
    #
    #         features, self.test_recurrent_hidden_states = self.actor_critic.feature_net(
    #             main_inputs, observations['objectgoal'], mapper.compass_to_orientation(observations['compass']), self.test_recurrent_hidden_states, self.not_done_masks
    #         )
    #
    #         #  Make masks not done till reset (end of episode) will be called
    #         self.not_done_masks.fill_(True)
    #
    #         # print("features",features.shape,features)
    #         value = self.actor_critic.critic(features)
    #         # distribution = self.actor_critic.action_net(features)
    #         # distribution = self.actor_critic._get_action_dist_from_latent(features, latent_sde=latent_sde)
    #         distribution = self.actor_critic._get_action_dist_from_latent(features)
    #         action = distribution.get_actions(deterministic=deterministic)
    #
    #         action_log_probs = distribution.log_prob(action).unsqueeze(-1)
    #
    #         # return value, action, action_log_probs
    #
    #
    #         observations['pointgoal_with_gps_compass'] = torch.zeros(self.batch_size,2,device=self.device)
    #         # x = nn.Sigmoid()(self.out(x))
    #         #ensure that the prediction stays on the [0,1] range
    #
    #         y_pred_perc = torch.sigmoid(action)
    #
    #
    #         y_pred_perc = y_pred_perc.cpu()
    #         # print("y_pred_perc",y_pred_perc)
    #         computed_reward = np.zeros((self.batch_size))
    #         for env_idx in range(self.batch_size):
    #             ################################################################
    #             '''
    #             post cnn
    #             '''
    #             #0 to 511
    #             '''
    #             local map shall be 0,255
    #             '''
    #
    #             map_size=mapper.get_map_size_in_cells(self.mapper_wrapper_array[env_idx].map_size_meters/2, self.mapper_wrapper_array[env_idx].map_cell_size) -1
    #             # print("map_size",map_size)
    #             y_pred_map_coo = y_pred_perc*map_size
    #             # print("y_pred_perc",y_pred_perc)
    #             # print("y_pred_map_coo",y_pred_map_coo)
    #             # self.mapper_wrapper_array[env_idx].update_prev_pred_in_map(torch.floor(y_pred_map_coo[env_idx]).long())
    #
    #             y_pred_map_coo_long = torch.round(y_pred_map_coo[env_idx]).long().cpu()
    #             if(y_pred_map_coo_long[0] > map_size):
    #                 y_pred_map_coo_long[0]=map_size
    #             if(y_pred_map_coo_long[1] > map_size):
    #                 y_pred_map_coo_long[1]=map_size
    #
    #             # print("y_pred_map_coo_long",y_pred_map_coo_long)
    #             '''
    #             update pred
    #             '''
    #             pred_y_pred_map_coo_long = y_pred_map_coo_long.clone().detach()
    #             computed_reward[env_idx] = self.mapper_wrapper_array[env_idx].update_prev_pred_in_map(pred_y_pred_map_coo_long, observations['objectgoal'][env_idx][0])
    #             # print("computed_reward[env_idx]",computed_reward[env_idx])
    #             # print("EXITING")
    #             # exit()
    #             '''
    #             HERE WE WILL USE A* TO FIND THE CLOSEST FREE CELL WITHIN SENSORING
    #             RANGE THAT STEER TOWARDS THE PREDICTED OBJECTIVE,
    #
    #             HAVING ALWAYS A CLOSE POINT GOAL FOR THE DDPPO PLANNER ENSURES A
    #             BETTER CONVERGENCE
    #             '''
    #             ################################################################
    #             #22 is the map_aux[0] which is the occupied map
    #             src_pt_value = int(np.floor(main_inputs.shape[-1]/2))
    #             src_pt = (src_pt_value,src_pt_value)
    #             dst_pt = (y_pred_map_coo_long[0].item(),y_pred_map_coo_long[1].item())
    #             # if(self.mapper_wrapper_array[env_idx].target_astar is None):
    #             #     # print("dst_pt pred",dst_pt)
    #             # else:
    #             #     dst_pt = self.mapper_wrapper_array[env_idx].target_astar
    #             #     # print("dst_pt target_astar",dst_pt)
    #
    #             #main_inputs is centered cropped so the middle of the map is where the agent is
    #             grid = main_inputs[env_idx][22].clone().detach()
    #             # print("y_pred_map_coo_long",y_pred_map_coo_long)
    #             y_pred_map_coo_local = self.get_free_cell_on_path(grid=grid.cpu().numpy(), src_pt=src_pt, dst_pt=dst_pt)
    #             if(y_pred_map_coo_local is None):
    #                 '''
    #                 could not find a path
    #                 '''
    #                 y_pred_map_coo_local = y_pred_map_coo_long
    #             # print("y_pred_map_coo_local",y_pred_map_coo_local)
    #             y_pred_reprojected_goal = mapper.map_to_2dworld(y_pred_map_coo_local, self.mapper_wrapper_array[env_idx].map_size_meters/2, self.mapper_wrapper_array[env_idx].map_cell_size)
    #             # print("y_pred_reprojected_goal",y_pred_reprojected_goal)
    #
    #             ################################################################
    #             # y_pred_reprojected_goal = mapper.map_to_2dworld(y_pred_map_coo, self.mapper_wrapper_array[env_idx].map_size_meters, self.mapper_wrapper_array[env_idx].map_cell_size)
    #             ################################################################
    #
    #             observations = self.compute_pointgoal_with_pred(observations, y_pred_reprojected_goal,env_idx)
    #             ################################################################
    #
    #         ####################################################################
    #         '''
    #         DEBUG
    #         '''
    #         # self.visualize_semantic_map(main_inputs,y_pred_map_coo_local, observations['objectgoal'][0]+1)
    #         # local_planner_actions = self.visualize_observations(observations)
    #         ####################################################################
    #
    #         '''
    #         feed ddppo
    #         '''
    #         del observations['objectgoal']
    #         del observations['compass']
    #         del observations['gps']
    #         # del observations['semantic']
    #         # print("observations['pointgoal_with_gps_compass']",observations['pointgoal_with_gps_compass'])
    #
    #
    #         '''
    #         COMMENT FOR DEBUG ONLY
    #         '''
    #         local_planner_actions = self.planner.act(observations)
    #         # local_planner_actions_debug = self.planner.act(observations)
    #         # print("local_planner_actions_debug",local_planner_actions_debug)
    #         # print("local_planner_actions",local_planner_actions,"\n")
    #
    #
    #         # print("local_planner_actions",local_planner_actions)
    #
    #         # print(value, action, action_log_probs, local_planner_actions, main_inputs, computed_reward)
    #         # print("EXITING")
    #         # exit()
    #
    #         #all envs
    #         self.ep_iteration+=1
    #
    #
    #         return value, action, action_log_probs, local_planner_actions, main_inputs, computed_reward, self.test_recurrent_hidden_states,
    ############################################################################
