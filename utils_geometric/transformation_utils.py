import numpy as np
import open3d as o3d
def degrees_to_radians(degrees):
    return np.radians(degrees)

def create_transformation_matrix(rotation, translation, inverse=False):
    # Create a rotation matrix from Euler angles (degrees) [roll, pitch, yaw]
    rotation_radians = degrees_to_radians(np.array(rotation))
    R = o3d.geometry.get_rotation_matrix_from_xyz(rotation_radians)

    # Create a transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation
    if inverse:
        T = np.linalg.inv(T)

    return T