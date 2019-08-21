import numpy as np
import tensorflow as tf
from pdb import set_trace as st

def transform_op(meshes, move):
    """
    Transform meshes along the vector (to_point - from_point)
    Args:
        meshes (ndarray): List of meshes. Each mesh is an array of shape (n_triangles, 3, 3).
        from_point (list or ndarray): [x, y, z]
        to_point (list or ndarray): [x, y, z]
    Return:
        new_meshes (ndarray)
    """
    # from_point = np.array(from_point)
    # to_point = np.array(to_point)
    # move = to_point - from_point
    new_meshes = meshes + move
    return new_meshes
def scale_op(meshes, factor):
    """
    Scale meshes with factor
    Args:
        meshes (ndarray): List of meshes. Each mesh is an array of shape (n_triangles, 3, 3).
        center_point (list or ndarray): [x, y, z]
    Return:
        new_meshes (ndarray)
    """
    # bottom_center_point = np.array(bottom_center_point)
    new_meshes = meshes * factor
    # new_bottom_center_point = bottom_center_point * factor
    return new_meshes
def keep_top_n(meshes, top_n):
    """
    Filter a list of meshes keeping only the top `n` wider triangles
    """
    def triangle_area(edge_1, edge_2):
        return np.linalg.norm(np.cross(edge_1, edge_2)) / 2.0
    meshes_filtered = []
    for mesh in meshes:
        areas = [triangle_area(edge_1=t[1] - t[0], edge_2=t[1] - t[2]) for t in mesh]
        widest_triangles = mesh[np.argsort(areas)[-top_n:]]
        meshes_filtered.append(widest_triangles)
    return np.asarray(meshes_filtered)
def local_rotate(vertices, angle_radians):
    '''Rotate the vertices in the object's reference frame,
        along the z axis.
    :angle in degree
    :angle_radians in radians
    :vertices N x 3 matrix
    '''
    # angle_radians = np.radians(angle)
    rotate_mat_t = tf.stack([
        [tf.cos(angle_radians), tf.sin(angle_radians), 0],
        [-tf.sin(angle_radians), tf.cos(angle_radians), 0],
        [0, 0, 1]
    ])
    center = tf.reduce_mean(vertices, axis=0, keepdims=True)
    centered_vertices = vertices - center
    rotated_vertices = tf.matmul(centered_vertices, rotate_mat_t)
    moved_back_vertices = rotated_vertices + center
    return moved_back_vertices
def local_translate(vertices, vector):
    return vertices + tf.reshape(vector, (1, 3))
class Renderer():
    def __init__(self, vertices, faces, all_ray_direction, all_ray_distance, face_labels=None, target_label=1, n_detections=2083, n_channels=64, max_angle=112.0009, min_angle=86.71596, 
                max_triangles=None, max_dist=60, has_ground=False, mesh_trainable=True, batch_size=64):
        """
        Create a Renderer object. 
        The lidar_origin should be provided in runtime, and he rendering output can be accessed through `point_cloud` attribute.
         
        Args:
            meshes (ndarray): List of meshes. Each mesh is an array of shape (n_triangles, 3, 3).
            n_detections (int): Number of the dections for one round of scanning.
            n_channels (int): Number of the rays in LiDAR.
            max_angle (float): Maximum angle of elevation in degree.
            min_angle (float): Minimum angle of depression in degree.
            max_triangles (int): Maximum number of meshes to process.
            max_dist (float): Maximum detection distence of LiDAR.
            mesh_trainable (bool): True if meshes are trainable.
        """
        self.epsilon = 1e-9
        # self.batch_size = batch_size
        self.n_detections = n_detections
        self.n_channels = n_channels
        self.n_rays = n_detections * n_channels
        self.max_angle = max_angle/180*np.pi
        self.min_angle = min_angle/180*np.pi
        self.max_dist = max_dist
        with tf.name_scope("renderer"):
            self.pl_lidar_origin = tf.placeholder(tf.float32, shape=(3,), name="origin")
            self.lidar_origin = tf.Variable(np.zeros(3), dtype=tf.float32, trainable=False)
            self.setup = []
            self.setup.append(self.lidar_origin.assign(self.pl_lidar_origin))
            # if max_triangles:
            #     meshes = keep_top_n(list(meshes), top_n=max_triangles)
            self.vertices =  tf.Variable(vertices, name="vertices", dtype=tf.float32)
            self.eps_vertices = tf.placeholder(tf.float32, shape=vertices.shape, name="eps_vertices")
            self.vertices_const =  tf.constant(vertices, name="vertices", dtype=tf.float32)
            self.reset_vertices = self.vertices.assign(self.vertices_const)
            self.angle_rad_noise = tf.placeholder(tf.float32, name="angle_rad_noise")
            self.dist_noise = tf.placeholder(tf.float32, shape=(3,), name="dist_noise")
            self.tf_vertices = local_rotate(local_translate(self.vertices + self.eps_vertices,self.dist_noise),self.angle_rad_noise)
            self.meshes =  tf.nn.embedding_lookup(self.tf_vertices, faces)
            # self.face_labels = face_labels
            # self.target_label = target_label
            self.n_meshes = self.meshes.shape[0]
            # print("Render: ", self.n_detections, self.n_channels, self.n_meshes)
            # build partial ray
                        
            mask = np.zeros([64,2083])
            mask[:,:100] = 1
            mask[:,900:1400] = 1
            # mask[:,800:1200] = 1
            mask_mesh = (mask == 1).flatten()
            mask_bg = (mask == 0).flatten()
            self.ray_direction = tf.constant(all_ray_direction[mask_mesh],dtype=tf.float32)
            self.bg_dist = all_ray_distance[mask_mesh]
            self.n_rays = self.ray_direction.shape[0]
            # self.ray_direction = self.get_ray_direction()
            
            mesh_ray_distance, mesh_ray_intersection = self.render()
            bg_ray_distance = tf.constant(all_ray_distance[mask_bg],dtype=tf.float32)
            bg_ray_intersection = tf.constant((all_ray_direction[mask_bg].T * all_ray_distance[mask_bg]).T, dtype=tf.float32)
            # st()
            ray_distance = tf.concat([mesh_ray_distance,bg_ray_distance], axis = 0)
            ray_intersection = tf.concat([mesh_ray_intersection,bg_ray_intersection], axis = 0)
            # ray_distance = bg_ray_distance
            # ray_intersection = bg_ray_intersection
            dist_mask = tf.less(ray_distance, self.max_dist, name="dist_mask")
            # self.point_dist = tf.boolean_mask(ray_distance, dist_mask, name="point_dist")
            self.point_cloud = tf.boolean_mask(ray_intersection, dist_mask, name="point_cloud")
    def get_ray_direction(self):
        with tf.name_scope("get_ray_direction"):
            # theta is the angle with z-axis
            # phi is the angle with x-axis
            # theta_ls = tf.lin_space(self.min_angle, self.max_angle, num=self.n_channels, name='phi_sep')
            # phi_ls = tf.lin_space(0.0, 2*np.pi, num=self.n_detections, name="theta_sep")
            theta_ls = tf.constant(np.load('data/theta.npy'),dtype=tf.float32)
            phi_ls = tf.constant(np.load('data/phi.npy'),dtype=tf.float32)
            self.n_rays = theta_ls.shape[-1]
            # st()
            # theta_sep = tf.tile(tf.expand_dims(theta_ls, 0), [phi_ls.shape[0], 1])  
            # theta_sep = tf.expand_dims(theta_sep, 2) 
            # phi_sep = tf.tile(tf.expand_dims(phi_ls, 1), [1, theta_ls.shape[0]]) 
            # phi_sep = tf.expand_dims(phi_sep, 2) 
            
            # param_grid = tf.concat([theta_sep, phi_sep], axis=-1, name="param_grid") 
            # param_grid_flat = tf.reshape(param_grid, shape=(-1, 2), name="param_grid_flat")
            # sin_param = tf.sin(param_grid_flat)
            # cos_param = tf.cos(param_grid_flat)
            
            # ray_x = sin_param[:,0] * cos_param[:,1] # x = sin(theta) * cos(phi)
            # ray_y = sin_param[:,0] * sin_param[:,1] # y = sin(theta) * sin(phi)
            # ray_z = cos_param[:,0]                  # z = cos(theta)
            ray_x = tf.sin(theta_ls) * tf.cos(phi_ls)
            ray_y = tf.sin(theta_ls) * tf.sin(phi_ls)
            ray_z = tf.cos(theta_ls)
            
            # self.n_detections, self.n_channels, 3
            ray_direction = tf.stack([ray_x, ray_y, ray_z], axis=1, name="ray_direction")
            # ray_direction_list = tf.split(ray_direction, self.n_rays//2)
            return ray_direction
    def render(self):
        with tf.name_scope("render"):
            edge_1 = self.meshes[:,1] - self.meshes[:,0]
            edge_2 = self.meshes[:,2] - self.meshes[:,0]
            origin_tile = tf.tile(tf.expand_dims(self.lidar_origin, 0), [self.n_meshes, 1])
            T = origin_tile - self.meshes[:,0]
            # [#rays, #meshes, 3]
            rays = tf.tile(tf.reshape(self.ray_direction, [-1, 1, 3]), [1, self.n_meshes, 1])
            edge_1 = tf.tile(tf.reshape(edge_1, [1, -1, 3]), [self.n_rays, 1, 1])
            edge_2 = tf.tile(tf.reshape(edge_2, [1, -1, 3]), [self.n_rays, 1, 1])
            T = tf.tile(tf.reshape(T, [1, -1, 3]), [self.n_rays, 1, 1])
            P = tf.cross(rays, edge_2)
            det = tf.reduce_sum(tf.multiply(edge_1, P), 2)
            det_sign = tf.sign(det)
            det_sign_tile = tf.tile(tf.reshape(det_sign, [self.n_rays, self.n_meshes, 1]), [1, 1, 3])
            T = tf.multiply(T, det_sign_tile)
            det = tf.abs(det)
            u = tf.reduce_sum(tf.multiply(T, P), 2)
            Q = tf.cross(T, edge_1)
            v = tf.reduce_sum(tf.multiply(rays, Q), 2)
            t = tf.reduce_sum(tf.multiply(edge_2, Q), 2)
            t /= (det+self.epsilon)
            t = tf.where(det < self.epsilon, tf.ones_like(t)*self.max_dist, t)
            t = tf.where(u < 0, tf.ones_like(t)*self.max_dist, t)
            t = tf.where(u > det, tf.ones_like(t)*self.max_dist, t)
            t = tf.where(v < 0, tf.ones_like(t)*self.max_dist, t)
            t = tf.where(u + v > det, tf.ones_like(t)*self.max_dist, t)
            t = tf.where(t < 0, tf.ones_like(t)*self.max_dist, t)
            min_t = tf.reduce_min(t, axis=1)
            # bg_dist = tf.constant(np.load('data/dist.npy'),dtype=tf.float32)
            bg_dist = tf.constant(self.bg_dist,dtype=tf.float32)
            min_t = tf.reduce_min(tf.stack([min_t,bg_dist]),0)
            ray_distance = min_t
            min_t_tile = tf.tile(tf.expand_dims(min_t, axis=1), [1, 3])
            # st()
            origin_tile = tf.tile(tf.expand_dims(self.lidar_origin, 0), [self.n_rays, 1])
            ray_intesction = origin_tile + min_t_tile * self.ray_direction
            return ray_distance, ray_intesction