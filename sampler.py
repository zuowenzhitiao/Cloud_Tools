import os
from random import sample

import point_cloud_utils as pcu
import numpy as np
from tqdm import tqdm

def sample_meshes_in_directory(mesh_dir, num_samples=100000):
    files = [os.path.join(root, file) for root, _, files in os.walk(mesh_dir) for file in files if file.endswith('.off')]

    for mesh_path in tqdm(files, desc='Sampling meshes'):
        v, f, n = pcu.load_mesh_vfn(mesh_path)

        # Generate random samples on the mesh
        f_i, bc = pcu.sample_mesh_random(v, f, num_samples=num_samples)
        v_sampled = pcu.interpolate_barycentric_coords(f, f_i, bc, v)

        # Save barycentric coordinates to .xyz file
        sample_path = os.path.splitext(mesh_path)[0] + '.xyz'
        np.savetxt(sample_path, v_sampled, fmt='%.12f')

mesh_dir = '/home/ubuntu/usrs/JK/Pointfilter-master/Pointfilter-master/Dataset/GT/PUNet/meshes/test1'
sample_meshes_in_directory(mesh_dir)
# n_sampled = pcu.interpolate_barycentric_coords(f, f_i, bc, n)