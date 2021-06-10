'''
Major refactor of the mapper
'''
################################################################################
import numpy as np
import quaternion
import torch
import torch.nn as nn
import time
from model import ChannelPool
################################################################################

# def get_padding_size(image, height, width):
#     # _, h, w, = image.shape
#     h = image.shape[-2]
#     w = image.shape[-1]
#     top, bottom, left, right = (0, 0, 0, 0)
#     if h < height:
#         dh = height - h
#         top = dh // 2
#         bottom = dh - top
#     if w < width:
#         dw = width - w
#         left = dw // 2
#         right = dw - left
#     else:
#         pass
#     return top, bottom, left, right
#
# def centered_crop_tensor():
#     top, bottom, left, right = get_padding_size(image, height, width)
#     torch.nn.functional.pad(input, pad, mode='constant', value=0)


def cartesian_to_polar(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi

def quaternion_rotate_vector(quat: np.quaternion, v: np.array) -> np.array:
    r"""Rotates a vector by a quaternion
    Args:
        quaternion: The quaternion to rotate by
        v: The vector to rotate
    Returns:
        np.array: The rotated vector
    """
    vq = np.quaternion(0, 0, 0, 0)
    vq.imag = v
    return (quat * vq * quat.inverse()).imag

def _compute_pointgoal(source_position, source_rotation, goal_position):
    '''
    based on https://github.com/facebookresearch/habitat-lab/blob/aa24551fd9942b11b7af4a076d9c05e080594d08/habitat/tasks/nav/nav.py#L163
    '''

    direction_vector = goal_position - source_position
    # print("direction_vector",direction_vector.shape,direction_vector)
    direction_vector_agent = quaternion_rotate_vector(
        source_rotation.inverse(), direction_vector
    )
    #POLAR 2D
    rho, phi = cartesian_to_polar(
        -direction_vector_agent[2], direction_vector_agent[0]
    )
    return np.array([rho, -phi], dtype=np.float32)

def map_to_2dworld(coo, map_size_meters, map_cell_size):
    shift = int(np.floor(get_map_size_in_cells(map_size_meters, map_cell_size) / 2.0))
    return (coo - shift)*map_cell_size

def quaternion_to_rotation(q_r, q_i, q_j, q_k):
    r"""
    ref: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    """
    s = 1  # unit quaternion
    rotation_mat = np.array(
        [
            [
                1 - 2 * s * (q_j ** 2 + q_k ** 2),
                2 * s * (q_i * q_j - q_k * q_r),
                2 * s * (q_i * q_k + q_j * q_r),
            ],
            [
                2 * s * (q_i * q_j + q_k * q_r),
                1 - 2 * s * (q_i ** 2 + q_k ** 2),
                2 * s * (q_j * q_k - q_i * q_r),
            ],
            [
                2 * s * (q_i * q_k - q_j * q_r),
                2 * s * (q_j * q_k + q_i * q_r),
                1 - 2 * s * (q_i ** 2 + q_j ** 2),
            ],
        ],
        dtype=np.float32,
    )
    return rotation_mat

def angle_to_quaternion_coeff(angle, axis):
    qw = np.cos(angle/2)
    qx = axis[0] * np.sin(angle/2)
    qy = axis[1] * np.sin(angle/2)
    qz = axis[2] * np.sin(angle/2)

    # return np.quaternion(qw,qx,qy,qz)
    return qw,qx,qy,qz

def compass_to_orientation(angle):
    """
    convert rad angle to [0,1]

    -pi,pi

    value*(max-min)+min
    """

    max = np.pi
    min = -np.pi
    phi = (angle-min)/(max-min)
    phi = torch.clamp(phi, min=0.0, max=1.0)
    phi = torch.floor((phi*360)/5)
    phi = torch.clamp(phi, min=0.0, max=72.0)
    phi = phi.long()
    return phi

def pose_from_angle_and_position(angle, position):
    '''
    r"""The agents heading in the coordinate frame defined by the epiosde,
    theta=0 is defined by the agents state at t=0
    """

    compass sensor range is -pi to +pi

    based on the top down frame of reference compass is

                     pi
                     ___
                    / | \
                -   |_|_|  +
                    \_|_/
    turn right -      0   turn left +

    Y+ is upwards (0,1,0)

    GPS IS -Z,X  and contation rotation of initial state and the shift from initial pos
    '''
    #necessary flip
    angle = angle*-1
    # print("angle",angle)

    # angle -= np.pi
    # #ensure -pi,pi range
    # if angle < -np.pi:
    #     angle =  (angle + np.pi) + np.pi
    # print("angle",angle)

    q_r,q_i,q_j,q_k = angle_to_quaternion_coeff(angle, [0.,1.,0.])
    matrix_rot = quaternion_to_rotation(q_r,q_i,q_j,q_k)

    gps = position

    translation_vec = torch.zeros(3)
    # translation_vec[0] = -gps[1]
    translation_vec[0] = gps[1]
    #+Y height     [1] = 0
    translation_vec[2] = gps[0]

    translation = torch.eye(4)
    translation[0:3,3] = translation_vec

    rotation = torch.eye(4)
    rotation[0:3,0:3] = torch.from_numpy(matrix_rot)

    #l = T*R*S
    pose = torch.matmul(translation,rotation)
    # print("pose",pose)

    return pose

def depth2local3d(depth, fx, fy, cx, cy):
    r"""Projects depth map to 3d point cloud
    with origin in the camera focus
    """
    device = depth.device
    h, w = depth.size()
    x = torch.linspace(0, w - 1, w).to(device)
    y = torch.linspace(0, h - 1, h).to(device)
    xv, yv = torch.meshgrid([x, y])
    dfl = depth.t().flatten()
    return torch.cat(
        [
            (dfl * (xv.flatten() - cx) / fx).unsqueeze(-1),  # x
            (dfl * (yv.flatten() - cy) / fy).unsqueeze(-1),  # y
            dfl.unsqueeze(-1),
        ],
        dim=1,
    )  # z

def get_map_size_in_cells(map_size_in_meters, cell_size_in_meters):
    # return int(np.ceil(map_size_in_meters / cell_size_in_meters)) + 1
    return int(np.ceil(map_size_in_meters / cell_size_in_meters))

def reproject_local_to_global(xyz_local, p):
    device = xyz_local.device
    num, dim = xyz_local.size()
    # print("xyz_local.size()", xyz_local.size())
    if dim == 3:
        xyz = torch.cat(
            [
                xyz_local,
                torch.ones((num, 1), dtype=torch.float32, device=device),
            ],
            dim=1,
        )
    elif dim == 4:
        xyz = xyz_local
    else:
        raise ValueError(
            "3d point cloud dim is neighter 3, or 4 (homogenious)"
        )
    # print("p",p.squeeze().shape, "xyz",xyz.t().shape)
    # print("p",p.shape, "xyz",xyz.t().shape)
    xyz_global = torch.mm(p, xyz.t())
    return xyz_global.t()

# def treat_boundaries(data_idxs,min_possible=0,max_possible=512,quarter_of_max_possible=128):
#     #the centered_cropped is half the resolution of map
#     min_x = max(data_idxs[0]-quarter_of_max_possible, min_possible)
#     max_x = min(data_idxs[0]+quarter_of_max_possible, max_possible)
#
#     min_y = max(data_idxs[1]-quarter_of_max_possible, min_possible)
#     max_y = min(data_idxs[1]+quarter_of_max_possible, max_possible)
#
#     ########################################################################
#     '''
#     treat special boundary cases
#     '''
#     if(data_idxs[0]-quarter_of_max_possible<min_possible):
#         min_x = 0
#         max_x = (data_idxs[0]+128)+(-1*(data_idxs[0]-128))
#
#     if(data_idxs[1]-quarter_of_max_possible<min_possible):
#         min_y = 0
#         max_y = (data_idxs[1]+128)+(-1*(data_idxs[1]-128))
#     ########################################################################
#
#     if(data_idxs[0]+quarter_of_max_possible>max_possible):
#         min_x = (data_idxs[0]-128)+(512-(data_idxs[0]+128))
#         max_x = 512
#
#     if(data_idxs[1]+quarter_of_max_possible>max_possible):
#         min_y = (data_idxs[1]-128)+(512-(data_idxs[1]+128))
#         max_y = 512
#     ########################################################################
#     return min_x,max_x,min_y,max_y

class MapperWrapper():
    '''
    A mapper_wrapper per env
    '''
    def __init__(self, config=None, device=torch.device("cuda")):
        self.config=config
        self.device=device

        self.obs_threshold=config.BEYOND.GLOBAL_POLICY.CELL_MIN_POINT_COUNT

        self.hfov = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV

        self.map_size_meters=config.BEYOND.GLOBAL_POLICY.MAP_SIZE_METERS
        self.map_cell_size=config.BEYOND.GLOBAL_POLICY.MAP_CELL_SIZE
        self.n_classes=config.BEYOND.GLOBAL_POLICY.N_CLASSES

        self.current_pose = torch.eye(4).float()

        self.shift_origin_gps = np.zeros(2)

        self.long_term_map = self.init_map2d_with_classes()

        # self.map_aux = torch.zeros(4,int(self.map_size_in_cells()),int(self.map_size_in_cells()), device=self.device)
        self.map_aux = torch.zeros(3,int(self.map_size_in_cells()),int(self.map_size_in_cells()), device=self.device)

        self.camera_height = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.POSITION[1]
        self.near_clip = config.BEYOND.GLOBAL_POLICY.NEAR_CLIP

        self.current_step = 0
        self.current_agent_cell = None

        self.pool = ChannelPool(1)
        self.shift = None

        # self.explored = 0
        # self.prev_explored = 0
        # self.explored_diff = 0
        # self.prev_explored_diff = 0
        # self.target_astar = None

        # self.target_explored = 0
        # self.prev_target_explored = 0
        # self.target_explored_diff = 0

        #parametrize this
        # map_total_cells = 256*256 #if the entire map could be seen in a single step
        # map_total_cells = 128*128 #considering 1/4 of the map can be seen in a single step
        # self.map_total_diff_cells = 1/(64*64) #considering 1/4 of the map can be seen in a single step

        '''
        # area of a trapezium isosceles of 11px to 66px
        due is because of hfov of 79 res of 640x480 w,h near clip of 0.6 and max_depth of 5meters
        for cell res of 0.1 is 1330px
        so for cell res of 0.05 is the double
        '''
        self.max_possible_explored_per_step = 2660

        self.mapper = DirectDepthMapper(
            camera_height=self.camera_height,
            near_th=config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH+self.near_clip,
            far_th=config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH,
            h_min=0.1,
            h_max=self.camera_height,
            map_size=self.map_size_meters,
            map_cell_size=self.map_cell_size,
            device=self.device,
            hfov=self.hfov,
            obs_threshold=self.obs_threshold,
        )
    ############################################################################
    def map_size_in_cells(self):
        return int(self.map_size_meters / self.map_cell_size)
    ############################################################################
    def init_map2d_with_classes(self):
        return (
            torch.zeros(self.n_classes, self.map_size_in_cells(), self.map_size_in_cells()
            )
            .float()
            .to(self.device)
        )
    def update_current_step(self,step):
        self.current_step = step
    def update_pose(self, angle, position):
        ########################################################################
        '''
        Introduce concept of map shift and forget old boundaries

        GPS is in meters we will work on map coordinates
        that map resolution dependant
        '''
        #meters to cell
        # print("gps before 0",position)
        # print("self.shift_origin_gps 0",self.shift_origin_gps)
        position=position+self.shift_origin_gps
        # print("gps after 0",position)

        position_in_cell = position/self.map_cell_size
        position_in_cell = position_in_cell.long()

        translation_vec=np.zeros(3)
        translation_vec[0] = -position_in_cell[1] #x
        #+Y height     [1] = 0            #y
        translation_vec[2] = position_in_cell[0]  #z
        # print("translation_vec",translation_vec)

        '''
        map is in ZX coordinates

        origin is at the center so 256,256
        '''
        shift_min_o=np.zeros(2,dtype=np.int32)
        shift_max_o=np.ones(2,dtype=np.int32)*512

        shift_min_new=np.zeros(2,dtype=np.int32)
        shift_max_new=np.ones(2,dtype=np.int32)*512
        should_shift=False

        if( translation_vec[2]>= 128 ):
            shift_min_o[0]=translation_vec[2]
            shift_max_o[0]=512

            shift_min_new[0]=0
            shift_max_new[0]=512-translation_vec[2]
            should_shift=True
            self.shift_origin_gps[0]-= position[0]

        if( translation_vec[2]<= -128 ):
            shift_min_o[0]=0
            shift_max_o[0]=512+translation_vec[2]#negative

            shift_min_new[0]=-1*translation_vec[2]#negative to positive
            shift_max_new[0]=512
            should_shift=True
            self.shift_origin_gps[0]+= -1*position[0]
        ########################################################################
        if( translation_vec[0]>= 128 ):
            shift_min_o[1]=translation_vec[0]
            shift_max_o[1]=512

            shift_min_new[1]=0
            shift_max_new[1]=512-translation_vec[0]
            should_shift=True
            self.shift_origin_gps[1]+= -1*position[1]


        if( translation_vec[0]<= -128 ):
            shift_min_o[1]=0
            shift_max_o[1]=512+translation_vec[0]#negative

            shift_min_new[1]=-1*translation_vec[0]#negative to positive
            shift_max_new[1]=512
            should_shift=True
            self.shift_origin_gps[1]-=  position[1]

        if(should_shift):
            # print("should_shift",should_shift)
            # print("shift_min_o",shift_min_o)
            # print("shift_max_o",shift_max_o)
            # print("shift_min_new",shift_min_new)
            # print("shift_max_new",shift_max_new)

            zeros = torch.zeros(self.n_classes+3, self.map_size_in_cells(), self.map_size_in_cells()).float().to(self.device)
            # zeros = torch.zeros(self.n_classes+4, self.map_size_in_cells(), self.map_size_in_cells()).float().to(self.device)

            zeros[:self.n_classes,shift_min_new[0]:shift_max_new[0],shift_min_new[1]:shift_max_new[1]]=self.long_term_map[:,shift_min_o[0]:shift_max_o[0],shift_min_o[1]:shift_max_o[1]]
            zeros[self.n_classes:,shift_min_new[0]:shift_max_new[0],shift_min_new[1]:shift_max_new[1]]=self.map_aux[:,shift_min_o[0]:shift_max_o[0],shift_min_o[1]:shift_max_o[1]]

            self.long_term_map=zeros[:self.n_classes,:,:]
            self.map_aux=zeros[self.n_classes:,:,:]

            # print("gps before 1",position)
            # print("self.shift_origin_gps 1",self.shift_origin_gps)
            position=position+self.shift_origin_gps
            # print("gps after 1",position)
        ########################################################################


        self.current_pose = pose_from_angle_and_position(angle, position)
    ############################################################################
    def update_map(self, depth, sseg):
        result = self.mapper.semantic_map(depth, sseg, self.current_pose)

        self.long_term_map = torch.max(result,self.long_term_map)

        '''
        aux 0 , is the occupied cells for the a* planner
        '''
        self.map_aux[0] = self.pool(self.long_term_map.unsqueeze(0)).squeeze(0)

        del result

        return self.post_proccess()
    def post_proccess(self):
        '''
        crop map around agent and compute shift, concat with aux
        '''

        global_input = self.long_term_map

        pts3d = torch.zeros(1,3)
        pts3d[0][0] = self.current_pose[0][3]
        pts3d[0][1] = self.current_pose[1][3]
        pts3d[0][2] = self.current_pose[2][3]

        pts2d = torch.cat([pts3d[:, 2:3], pts3d[:, 0:1]], dim=1)
        del pts3d

        data_idxs = torch.round(
            project2d_pcl_into_worldmap(pts2d, self.mapper.map_size_meters, self.mapper.map_cell_size)
        )

        data_idxs = data_idxs.squeeze(0).long()
        self.current_agent_cell = data_idxs

        '''
        create 3x3 centered on the agent
        local_map[e, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.

        1,2
        self.map_aux[1:3, data_idxs[0]-1:data_idxs[0]+2 , data_idxs[1]-1:data_idxs[1]+2  ] = 1.0
        '''

        '''
        aux 1 , is past locations
        aux 2 , is the current location
        '''
        # self.map_aux[2].fill_(0)
        self.map_aux[2,:,:]=0
        self.map_aux[1:3, data_idxs[0]-1:data_idxs[0]+2 , data_idxs[1]-1:data_idxs[1]+2  ] = 1.0

        '''
        aux 3 , is the new prediction
        '''

        global_input = torch.cat([global_input,self.map_aux],dim=0)

        # centered_cropped = torch.zeros((26,256,256),device=self.device)
        '''
        Problem with the boundaries
        '''
        min_x=data_idxs[0]-128
        max_x=data_idxs[0]+128
        min_y=data_idxs[1]-128
        max_y=data_idxs[1]+128


        centered_cropped = global_input[:,min_x:max_x,min_y:max_y]
        #bad but to avoid breaking
        if(centered_cropped.shape[1] != 256 or centered_cropped.shape[2] != 256):
            centered_cropped = global_input[:,128:128+256,128:128+256]

        self.shift = [min_x,max_x,min_y,max_y]


        # self.explored = torch.sum(centered_cropped[-4]).item()
        # self.explored_diff = (self.explored - self.prev_explored)
        #
        # if(self.explored_diff > 0):
        #     self.prev_explored_diff = self.explored_diff
        #
        # self.prev_explored = self.explored

        return centered_cropped

    def compute_reward(self):
        computed_reward = 0
        if(self.explored_diff==0):
            computed_reward = -self.prev_explored_diff
        else:
            computed_reward = self.explored_diff
        #normalize to [-1,1]
        computed_reward = computed_reward/self.max_possible_explored_per_step
        #scale down to [-0.1,0.1]
        computed_reward = computed_reward*0.1

        return computed_reward


    ############################################################################
    def reset_map(self):
        self.long_term_map = (
            torch.zeros(self.n_classes, self.map_size_in_cells(), self.map_size_in_cells()
            )
            .float()
            .to(self.device)
        )
        # self.map_aux = torch.zeros(4,int(self.map_size_in_cells()),int(self.map_size_in_cells()), device=self.device)
        self.map_aux = torch.zeros(3,int(self.map_size_in_cells()),int(self.map_size_in_cells()), device=self.device)
        self.shift_origin_gps = np.zeros(2)
        # self.explored = 0
        # self.prev_explored = 0
        # self.explored_diff = 0
        # self.prev_explored_diff = 0
        #
        # self.target_explored = 0
        # self.prev_target_explored = 0
        # self.target_explored_diff = 0

    ############################################################################

def project2d_pcl_into_worldmap(zx, map_size, cell_size):
    # print("zx.shape",zx.shape)

    device = zx.device
    shift = int(np.floor(get_map_size_in_cells(map_size, cell_size) / 2.0))
    topdown2index = torch.tensor(
        [[1.0 / cell_size, 0, shift], [0, 1.0 / cell_size, shift], [0, 0, 1]],
        device=device,
    )
    world_coords_h = torch.cat(
        [zx.view(-1, 2), torch.ones((len(zx), 1), device=device)], dim=1
    )

    # print("topdown2index.shape",topdown2index.shape,"world_coords_h.t().shape", world_coords_h.t().shape)

    world_coords = torch.mm(topdown2index, world_coords_h.t())
    return world_coords.t()[:, :2]

def pcl_to_obstacles(pts3d, map_size=24, cell_size=0.05, min_pts=10, sseg=None, obs_threshold=1.0):
    r"""Counts number of 3d points in 2d map cell.
    Height is sum-pooled.
    """
    # print("sseg.shape[-1] 1",sseg.shape[-1])
    # print("EXITING inside pcl_to_obstacles")

    device = pts3d.device
    map_size_in_cells = get_map_size_in_cells(map_size, cell_size)

    if len(pts3d) <= 1:
        init_map_final = torch.zeros(
            (sseg.shape[-1], map_size_in_cells, map_size_in_cells, ), device=device
        )
        # print("init_map_final",init_map_final.shape)
        return init_map_final

    init_map = torch.zeros(
        (map_size_in_cells, map_size_in_cells, sseg.shape[-1]), device=device
    )

    num_pts, dim = pts3d.size()

    pts2d = torch.cat([pts3d[:, 2:3], pts3d[:, 0:1]], dim=1)

    data_idxs = torch.round(
        project2d_pcl_into_worldmap(pts2d, map_size, cell_size)
    )

    if len(data_idxs) > min_pts:
        ones = torch.ones(
            (map_size_in_cells, map_size_in_cells, sseg.shape[-1]), device=device
        )

        zeros = torch.zeros(
            (map_size_in_cells, map_size_in_cells, sseg.shape[-1]), device=device
        )
        ########################################################################
        '''
        This block is super slow we need to OPTIMIZE it
        '''
        u, map_index, counts = torch.unique(data_idxs, sorted=True, return_inverse=True, return_counts=True, dim=0)

        u=u.long()
        counts=counts.float()

        ########################################################################
        '''
        equivalent to with a small rsme of 10-7 and
        t o 3.3116657733917236
        t new 0.0040149688720703125 in seconds

        size = map_index.shape[0]
        for i in range(size):
            init_map[ u[map_index[i]][0] ][ u[map_index[i]][1] ] += sseg[i]
        '''

        counts_sseg = torch.zeros(counts.shape[0],sseg.shape[-1],device=device)
        dim = 0
        counts_sseg.index_add_(dim, map_index, sseg)
        init_map[u[:, 0], u[:, 1]] = counts_sseg
        ########################################################################
        # print("init_map",torch.unique(init_map))
        ########################################################################

        #if obs_threshold==1 any cell != 0 is 1
        '''
        commented to save unecessary computation since for now obs_threshold is 1.0
        init_map = init_map / obs_threshold
        '''
        # init_map = init_map / obs_threshold
        init_map = torch.where( init_map >= 0.5, ones, zeros )

        # init_map[init_map >= 0.5] = 1.0
        # init_map[init_map < 0.5] = 0.0
        #normalize
        # init_map = init_map/max_possible_value
        #put in channel x h x w
        # init_map = init_map.permute(2,0,1)

        # print()
        # print("init_map",init_map.shape)
        # print("init_map",torch.unique(init_map))
        ########################################################################
        del counts_sseg

    init_map = init_map.permute(2,0,1)

    return init_map

class DirectDepthMapper():
    r"""Estimates obstacle map given the depth image
    ToDo: replace numpy histogram counting with differentiable
    pytorch soft count like in
    https://papers.nips.cc/paper/7545-unsupervised-learning-of-shape-and-pose-with-differentiable-point-clouds.pdf
    """

    def __init__(
        self,
        camera_height=0.88,
        near_th=0.1,
        far_th=4.0,
        h_min=0.264,
        h_max=0.88,
        map_size=40,
        map_cell_size=0.1,
        device=torch.device("cuda"),
        hfov=58,
        obs_threshold=1.0,
        **kwargs
    ):
        self.device = device
        self.near_th = near_th
        self.far_th = far_th
        self.h_min_th = h_min
        self.h_max_th = h_max
        self.camera_height = camera_height
        self.map_size_meters = map_size
        self.map_cell_size = map_cell_size
        self.hfov = hfov* np.pi / 180.
        self.obs_threshold = obs_threshold

        # print("self.map_size_meters",self.map_size_meters)
        # print("self.map_cell_size",self.map_cell_size)
        # print("get_map_size_in_cells(self.map_size_meters, self.map_cell_size)",get_map_size_in_cells(self.map_size_meters, self.map_cell_size))
    ############################################################################
    def semantic_map(self, depth, sseg, pose=torch.eye(4).float()):
        self.device = depth.device

        #logic for generic h,w, hfov
        hfov=self.hfov
        vfov= hfov*depth.size(0)/depth.size(1)

        self.fx = float((depth.size(1) / 2.)* np.cos(hfov / 2.)/np.sin(hfov / 2.))
        self.fy = float((depth.size(0) / 2.)* np.cos(vfov / 2.)/np.sin(vfov / 2.))

        self.cx = int(depth.size(1)/2) - 1
        self.cy = int(depth.size(0)/2) - 1
        pose = pose.to(self.device)
        ########################################################################
        sseg_pcl = sseg.permute(1,0,2)#transpose but keep channels necessary because of depth2local3d

        local_3d_pcl = depth2local3d(depth, self.fx, self.fy, self.cx, self.cy)

        survived_points = local_3d_pcl
        sseg_pcl_survived = sseg
        # del sseg_pcl
        # print("local_3d_pcl=",local_3d_pcl.shape)

        ########################################################################
        #depth clipping
        idxs = (torch.abs(local_3d_pcl[:, 2]) < self.far_th) * (
            torch.abs(local_3d_pcl[:, 2]) >= self.near_th
        )
        #depth
        survived_points = local_3d_pcl[idxs]
        #sseg
        # print("idxs.shape",idxs.shape)
        # print("survived_points=",survived_points.shape)

        # print("sseg_pcl_survived.shape",sseg_pcl_survived.shape)
        sseg_pcl_survived = sseg_pcl_survived[ idxs.view( sseg_pcl_survived.shape[0],sseg_pcl_survived.shape[1],1 ).expand( sseg_pcl_survived.size() ) ]
        sseg_pcl_survived = sseg_pcl_survived.view(survived_points.shape[0],sseg.shape[2])
        # print("sseg_pcl_survived.shape",sseg_pcl_survived.shape)
        #
        ########################################################################
        #handle min points
        # print("sseg.shape[-1] 0",sseg.shape[-1])

        if len(survived_points) < 20:
            map_size_in_cells = (
                get_map_size_in_cells(self.map_size_meters, self.map_cell_size)
            )
            init_map = torch.zeros(
                (sseg.shape[-1], map_size_in_cells, map_size_in_cells), device=self.device
            )
            # print("init_map semantic_map",init_map.shape)
            return init_map
        ########################################################################
        #egocentric map to geocentric
        global_3d_pcl = reproject_local_to_global(survived_points, pose)[:, :3]
        # print("global_3d_pcl",global_3d_pcl.shape)

        # Because originally y looks down and from agent camera height
        global_3d_pcl[:, 1] = -global_3d_pcl[:, 1] + self.camera_height
        ########################################################################
        ##height clipping
        idxs = (global_3d_pcl[:, 1] > self.h_min_th) * (
            global_3d_pcl[:, 1] < self.h_max_th
        )
        # print("idxs",idxs.shape)
        #depth
        global_3d_pcl = global_3d_pcl[idxs]
        #sseg
        sseg_pcl_survived = sseg_pcl_survived[ idxs.view(idxs.shape[0],1).expand(sseg_pcl_survived.size()) ]
        sseg_pcl_survived = sseg_pcl_survived.view(global_3d_pcl.shape[0],sseg.shape[2])
        # #######################################################################
        # print("EXITING before pcl_to_obstacles")
        # exit()
        # print("global_3d_pcl",global_3d_pcl.shape)
        # print("sseg_pcl_survived",sseg_pcl_survived.shape)

        obstacle_map = pcl_to_obstacles(
            global_3d_pcl, self.map_size_meters, self.map_cell_size, sseg=sseg_pcl_survived, obs_threshold=self.obs_threshold
        )
        # print("obstacle_map",obstacle_map.shape)
        ########################################################################
        del local_3d_pcl, sseg_pcl_survived, global_3d_pcl, idxs
        # del local_3d_pcl, global_3d_pcl, idxs
        ########################################################################
        return obstacle_map
    ############################################################################
