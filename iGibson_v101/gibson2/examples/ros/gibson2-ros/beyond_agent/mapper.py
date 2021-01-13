import numpy as np
import quaternion
import torch
import torch.nn as nn
import time

EPSILON = 1e-8

def _compute_pointgoal(source_position, source_rotation, goal_position):
        # print("goal_position",goal_position,"source_rotation",source_rotation)
        direction_vector = goal_position - source_position
        # print("direction_vector",direction_vector, "source_rotation.inverse()",source_rotation.inverse())
        direction_vector_agent = quaternion_rotate_vector(
            source_rotation.inverse(), direction_vector
        )
        #POLAR 2D
        rho, phi = cartesian_to_polar(
            -direction_vector_agent[2], direction_vector_agent[0]
        )
        return np.array([rho, -phi], dtype=np.float32)

        # if self._goal_format == "POLAR":
        #     if self._dimensionality == 2:
        #         rho, phi = cartesian_to_polar(
        #             -direction_vector_agent[2], direction_vector_agent[0]
        #         )
        #         return np.array([rho, -phi], dtype=np.float32)
        #     else:
        #         _, phi = cartesian_to_polar(
        #             -direction_vector_agent[2], direction_vector_agent[0]
        #         )
        #         theta = np.arccos(
        #             direction_vector_agent[1]
        #             / np.linalg.norm(direction_vector_agent)
        #         )
        #         rho = np.linalg.norm(direction_vector_agent)
        #
        #         return np.array([rho, -phi, theta], dtype=np.float32)
        # else:
        #     if self._dimensionality == 2:
        #         return np.array(
        #             [-direction_vector_agent[2], direction_vector_agent[0]],
        #             dtype=np.float32,
        #         )
        #     else:
        #         return direction_vector_agent

# def _compute_pointgoal(source_position, source_rotation, goal_position):
#     direction_vector = goal_position - source_position
#     direction_vector_agent = quaternion_rotate_vector(
#         source_rotation.inverse(), direction_vector
#     )
#
#     return np.array(
#         [-direction_vector_agent[2], direction_vector_agent[0]],
#         dtype=np.float32,
#     )

def batch_original_3d_to_episodic_3d(states, goals):

    result = []
    for i in range(len(goals)):
        result.append(original_3d_to_episodic_3d(states[i].position, states[i].rotation, goals[i]))

    return torch.from_numpy(np.array(result))


def original_3d_to_episodic_3d(source_position, source_rotation, goal_position):
    # source_position = np.array(episode.start_position, dtype=np.float32)
    # rotation_world_start = quaternion_from_coeff(episode.start_rotation)
    # goal_position = np.array(episode.goals[0].position, dtype=np.float32)

    # return self._compute_pointgoal(
    #     source_position, rotation_world_start, goal_position
    # )

    direction_vector = goal_position - source_position
    direction_vector_agent = quaternion_rotate_vector(
        source_rotation.inverse(), direction_vector
    )

    return np.array(
        [-direction_vector_agent[2], direction_vector_agent[0]],
        dtype=np.float32,
    )




def cartesian_to_polar(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi

def _quat_to_xy_heading(quat):
    direction_vector = np.array([0, 0, -1])

    heading_vector = quaternion_rotate_vector(quat, direction_vector)

    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    return np.array([phi], dtype=np.float32)

def pose_from_angle_and_position(angle, position):
    # print("angle",angle)

    phi = 2*np.pi - angle

    q_r,q_i,q_j,q_k = angle_to_quaternion_coeff(phi, [0.,1.,0.])

    matrix_rot = quaternion_to_rotation(q_r,q_i,q_j,q_k)
    # print("matrix_rot",matrix_rot.shape)

    gps = position

    translation_vec = torch.zeros(3)
    translation_vec[0] = gps[1]
    translation_vec[2] = gps[0]

    # pose = np.eye(4)
    pose = torch.eye(4)
    pose[0:3,3] = translation_vec

    # pose2 = np.eye(4)
    pose2 = torch.eye(4)
    pose2[0:3,0:3] = torch.from_numpy(matrix_rot)

    #l = T*R*S
    # pose = np.matmul(pose,pose2)
    pose = torch.matmul(pose,pose2)

    return pose,phi

def quaternion_from_coeff(coeffs: np.ndarray) -> np.quaternion:
    r"""Creates a quaternions from coeffs in [x, y, z, w] format"""
    quat = np.quaternion(0, 0, 0, 0)
    quat.real = coeffs[3]
    quat.imag = coeffs[0:3]
    return quat

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

def quaternion_from_two_vectors(v0: np.array, v1: np.array) -> np.quaternion:
    r"""Computes the quaternion representation of v1 using v0 as the origin."""
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    c = v0.dot(v1)
    # Epsilon prevents issues at poles.
    if c < (-1 + EPSILON):
        c = max(c, -1)
        m = np.stack([v0, v1], 0)
        _, _, vh = np.linalg.svd(m, full_matrices=True)
        axis = vh.T[:, 2]
        w2 = (1 + c) * 0.5
        w = np.sqrt(w2)
        axis = axis * np.sqrt(1 - w2)
        return np.quaternion(w, *axis)

    axis = np.cross(v0, v1)
    s = np.sqrt((1 + c) * 2)
    return np.quaternion(s * 0.5, *(axis / s))


def map_to_2dworld(coo, map_size_meters, map_cell_size):
    shift = int(np.floor(get_map_size_in_cells(map_size_meters, map_cell_size) / 2.0))
    return (coo - shift)*map_cell_size

def coo2dworld_to_3d(coo,origin):
    #
    agent_batch = []
    for i in range(coo.shape[0]):

        agent_position = np.array([coo[i][1],0,-coo[i][0]])

        agent_position = quaternion_rotate_vector(
                origin[i][1], agent_position
            )

        agent_position = agent_position + origin[i][0]
        '''
        reverse this op
        agent_position = quaternion_rotate_vector(
                rotation_world_start.inverse(), agent_position - origin
            )
        '''
        agent_batch.append(agent_position)

    return np.array(agent_batch)

def worldpointgoal_to_map(pts2d_o, map_size_meters, map_cell_size):

    data_idxs = torch.round(
        project2d_pcl_into_worldmap(pts2d_o, map_size_meters, map_cell_size)
    )

    data_idxs = data_idxs.long()

    return data_idxs

def angle_to_quaternion_coeff(angle, axis):
    qw = np.cos(angle/2)
    qx = axis[0] * np.sin(angle/2)
    qy = axis[1] * np.sin(angle/2)
    qz = axis[2] * np.sin(angle/2)

    # return np.quaternion(qw,qx,qy,qz)
    return qw,qx,qy,qz

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


def get_map_size_in_cells(map_size_in_meters, cell_size_in_meters):
    return int(np.ceil(map_size_in_meters / cell_size_in_meters)) + 1


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

# from numba import jit
# from numba import njit
def pcl_to_obstacles(pts3d, map_size=24, cell_size=0.05, min_pts=10, sseg=None, max_possible_value=40960):
    r"""Counts number of 3d points in 2d map cell.
    Height is sum-pooled.
    """
    device = pts3d.device
    map_size_in_cells = get_map_size_in_cells(map_size, cell_size) - 1

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

        #normalize
        init_map = init_map/max_possible_value
        #put in channel x h x w
        init_map = init_map.permute(2,0,1)
        # print()
        # print("init_map",init_map.shape)
        ########################################################################
        del counts_sseg
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
        max_possible_value=40960,
        batch_size=1,
        **kwargs
    ):
        super(DirectDepthMapper, self).__init__()
        self.device = device
        self.near_th = near_th
        self.far_th = far_th
        self.h_min_th = h_min
        self.h_max_th = h_max
        self.camera_height = camera_height
        self.map_size_meters = map_size
        self.map_cell_size = map_cell_size
        self.hfov = hfov* np.pi / 180.
        self.max_possible_value = max_possible_value
        self.batch_size = batch_size

        return

    def semantic_map(self, depth, sseg, pose=torch.eye(4).float()):

        # print("depth",depth.shape,"sseg", sseg.shape,"pose", pose.shape)

        self.device = depth.device

        #logic for generic h,w, hfov
        hfov=self.hfov
        vfov= hfov*depth.size(0)/depth.size(1)

        self.fx = float((depth.size(1) / 2.)* np.cos(hfov / 2.)/np.sin(hfov / 2.))
        self.fy = float((depth.size(0) / 2.)* np.cos(vfov / 2.)/np.sin(vfov / 2.))

        self.cx = int(depth.size(1)/2) - 1
        self.cy = int(depth.size(0)/2) - 1
        pose = pose.to(self.device)


        #sseg
        '''
            First problem depth and sseg don't have the same size anymore
        '''
        # sseg_pcl = sseg.t().flatten()
        # h w c
        sseg_pcl = sseg.permute(1,0,2)#transpose but keep channels
        # batch h w c
        # sseg_pcl = sseg.permute(0,2,1,3)#transpose but keep channels
        #depth
        # print("depth2local3d")

        local_3d_pcl = depth2local3d(depth, self.fx, self.fy, self.cx, self.cy)

        survived_points = local_3d_pcl
        sseg_pcl_survived = sseg_pcl
        del sseg_pcl


        #depth clipping
        idxs = (torch.abs(local_3d_pcl[:, 2]) < self.far_th) * (
            torch.abs(local_3d_pcl[:, 2]) >= self.near_th
        )
        #depth
        survived_points = local_3d_pcl[idxs]
        #sseg
        # print("idxs.shape",idxs.shape)
        sseg_pcl_survived = sseg_pcl_survived[ idxs.view(sseg_pcl_survived.shape[0],sseg_pcl_survived.shape[1],1).expand(sseg_pcl_survived.size()) ]
        sseg_pcl_survived = sseg_pcl_survived.view(survived_points.shape[0],sseg.shape[2])
        # print("sseg_pcl_survived.shape",sseg_pcl_survived.shape)
        #
        #
        #handle min points
        if len(survived_points) < 20:
            map_size_in_cells = (
                get_map_size_in_cells(self.map_size_meters, self.map_cell_size)
                - 1
            )
            init_map = torch.zeros(
                (sseg.shape[-1], map_size_in_cells, map_size_in_cells), device=self.device
            )
            # print("init_map semantic_map",init_map.shape)
            return init_map

        # print("reproject_local_to_global")
        #egocentric map to geocentric
        global_3d_pcl = reproject_local_to_global(survived_points, pose)[:, :3]


        # Because originally y looks down and from agent camera height
        global_3d_pcl[:, 1] = -global_3d_pcl[:, 1] + self.camera_height

        ##height clipping
        idxs = (global_3d_pcl[:, 1] > self.h_min_th) * (
            global_3d_pcl[:, 1] < self.h_max_th
        )
        # idxs = (global_3d_pcl[:, 1] > self.h_min_th)

        #depth
        global_3d_pcl = global_3d_pcl[idxs]
        #sseg
        sseg_pcl_survived = sseg_pcl_survived[ idxs.view(idxs.shape[0],1).expand(sseg_pcl_survived.size()) ]
        sseg_pcl_survived = sseg_pcl_survived.view(global_3d_pcl.shape[0],sseg.shape[2])

        # print("pcl_to_obstacles")
        # time_cost = time.time()
        obstacle_map = pcl_to_obstacles(
            global_3d_pcl, self.map_size_meters, self.map_cell_size, sseg=sseg_pcl_survived, max_possible_value=self.max_possible_value
        )
        # time_cost = time.time() - time_cost
        # print("time_cost ONLY pcl_to_obstacles",time_cost)

        del local_3d_pcl, sseg_pcl_survived, global_3d_pcl, idxs

        return obstacle_map

class MapperWrapper():
    # def __init__(self, hfov, map_size_meters=12., map_cell_size=0.05):
    def __init__(self, map_size_meters=25.6, map_cell_size=0.05, config=None, n_classes=81, batch_size=1):
        self.device = "cuda"
        self.batch_size = batch_size
        self.hfov = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV
        # self.map_size_meters = 40
        # self.map_size_meters = 100
        # self.map_size_meters = 20
        # self.map_size_meters = 8
        # self.map_size_meters = 16
        # self.map_size_meters = 25.6
        self.map_size_meters = map_size_meters
        self.map_cell_size = map_cell_size
        # self.map_cell_size = 0.2
        # self.map_cell_size = 0.25
        self.n_classes = n_classes
        self.current_obstacles = self.init_map2d_with_classes()

        self.map2DObstacles = self.init_map2d_with_classes()

        # self.mapper = DirectDepthMapper(
        #     camera_height=config.CAMERA_HEIGHT,
        #     near_th=config.D_OBSTACLE_MIN,
        #     far_th=config.D_OBSTACLE_MAX,
        #     h_min=config.H_OBSTACLE_MIN,
        #     h_max=config.H_OBSTACLE_MAX,
        #     map_size=config.MAP_SIZE,
        #     map_cell_size=config.MAP_CELL_SIZE,
        #     device=device,
        # )
        self.camera_height = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.POSITION[1]

        self.mapper = DirectDepthMapper(
            camera_height=self.camera_height,
            near_th=config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH+0.1,
            far_th=config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH,
            h_min=self.camera_height-0.25,
            h_max=self.camera_height+0.25,
            map_size=self.map_size_meters,
            map_cell_size=self.map_cell_size,
            hfov=self.hfov,
            max_possible_value=config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH*((config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH-config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH)/self.map_cell_size),
            batch_size=self.batch_size,
        )
        # self.pose6D = self.init_pose6d()
        self.pose6D = self.init_pose6d_batch(batch_size)

        # self.map_aux = torch.zeros(batch_size,2,int(self.map_size_in_cells()/2),int(self.map_size_in_cells()/2), device=self.device)
        self.map_aux = torch.zeros(batch_size,2,int(self.map_size_in_cells()),int(self.map_size_in_cells()), device=self.device)

    # def init_pose6d(self):
    #     return torch.eye(4).float().to(self.device)

    def init_pose6d_batch (self, batch_size):
        return torch.zeros(batch_size,4,4).float().to(self.device)

    def map_size_in_cells(self):
        return int(self.map_size_meters / self.map_cell_size)

    # def init_map2d(self):
    #     return (
    #         torch.zeros(
    #             1, 1, self.map_size_in_cells(), self.map_size_in_cells()
    #         )
    #         .float()
    #         .to(self.device)
    #     )

    def init_map2d_with_classes(self):
        return (
            torch.zeros(self.batch_size, self.n_classes, self.map_size_in_cells(), self.map_size_in_cells()
            )
            .float()
            .to(self.device)
        )

    def reset_map_batch_idx(self,i):
        self.map2DObstacles[i] = 0.
        self.map_aux[i] = 0.

    def update_map(self, depth, sseg, phi):
        # print("self.batch_size",self.batch_size)
        # print("depth",depth.shape)
        # print("sseg",sseg.shape)


        # current_obstacles = self.init_map2d_with_classes()
        current_obstacles = self.current_obstacles

        for i in range(self.batch_size):
            tmp = self.mapper.semantic_map(depth[i], sseg[i], self.pose6D[i])
            if(tmp.shape[0]!=sseg.shape[-1]):
                tmp = tmp.permute(2,0,1)
            current_obstacles[i] = tmp

        # print("self.current_obstacles",self.current_obstacles.shape)

        self.current_obstacles = current_obstacles
        self.map2DObstacles = torch.max(current_obstacles,self.map2DObstacles)

        del current_obstacles
        # print("self.map2DObstacles",self.map2DObstacles.shape)

        return self.post_proccess(phi)

    def post_proccess(self, phi):
        # global_input = nn.MaxPool2d(2)(self.map2DObstacles)
        global_input = self.map2DObstacles
        # global_input = self.init_map2d_with_classes()
        for i in range(self.batch_size):

            pts3d = torch.zeros(1,3)
            pts3d[0][0] = self.pose6D[i][0][3]
            pts3d[0][1] = self.pose6D[i][1][3]
            pts3d[0][2] = self.pose6D[i][2][3]

            pts2d = torch.cat([pts3d[:, 2:3], pts3d[:, 0:1]], dim=1)
            del pts3d

            # data_idxs = torch.round(
            #     project2d_pcl_into_worldmap(pts2d, self.mapper.map_size_meters/2, self.mapper.map_cell_size)
            # )

            data_idxs = torch.round(
                project2d_pcl_into_worldmap(pts2d, self.mapper.map_size_meters, self.mapper.map_cell_size)
            )

            data_idxs = data_idxs.squeeze(0).long()

            '''
            update agent cell pose here
            '''
            # self.map_aux[i][0][data_idxs[0]][data_idxs[1]] = 1.
            # self.map_aux[i][0][data_idxs[0]][data_idxs[1]] = 2.
            self.map_aux[i][0][data_idxs[0]][data_idxs[1]] = phi[i]

            # value, x, rnn_hxs = self.g_policy(global_input, rnn_hxs, masks, extras)
            # x = self.g_policy(global_input[:,:32,:,:], extras)

            '''
            update trajectory here
            '''
            #only previous poses
            self.map_aux[i][1][data_idxs[0]][data_idxs[1]] = 1.

        #end for
        # print("global_input",global_input.shape)
        # print("self.map_aux",self.map_aux.shape)
        global_input = torch.cat([global_input,self.map_aux],dim=1)
        # global_input = nn.MaxPool2d(2)(global_input)


        return global_input
        #######################################################################

    # def seeker(self, object_class):
    #     # self.map2DObstacles
    #
    #     for i in range(self.batch_size):
    #
    #         pts3d = torch.zeros(1,3)
    #         pts3d[0][0] = self.pose6D[i][0][3]
    #         pts3d[0][1] = self.pose6D[i][1][3]
    #         pts3d[0][2] = self.pose6D[i][2][3]
    #
    #         pts2d = torch.cat([pts3d[:, 2:3], pts3d[:, 0:1]], dim=1)
    #
    #         data_idxs = torch.round(
    #             project2d_pcl_into_worldmap(pts2d, self.mapper.map_size_meters, self.mapper.map_cell_size)
    #         )
    #
    #         data_idxs = data_idxs.squeeze(0).long()
    #
    #         # self.map_aux[i][1][data_idxs[0]][data_idxs[1]] = 1.
    #         # self.map2DObstacles
    #         '''
    #             would be the agent on the target layer self.map2DObstacles[i][object_class][data_idxs[0]][data_idxs[1]]
    #             so we want the closest non-zero cell x,y to data_idxs
    #
    #
    #         '''
    #         # self.map2DObstacles[i][object_class][data_idxs[0]][data_idxs[1]]
    #
    #     return
