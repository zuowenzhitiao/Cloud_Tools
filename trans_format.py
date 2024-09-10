import os
import shutil
import numpy as np
from plyfile import PlyData, PlyElement
from datetime import datetime

def npy2ply(directory):
    timestamp = datetime.now().strftime('%m%d_%H%M')
    npy_directory = os.path.join(directory, f'npy_{timestamp}')
    os.makedirs(npy_directory)
    ply_directory = os.path.join(directory, f'ply_{timestamp}')
    os.makedirs(ply_directory)

    for file in os.listdir(directory):
        if file.endswith('.npy'):
            shutil.move(os.path.join(directory, file), os.path.join(npy_directory, file))

    npy_files = [f for f in os.listdir(npy_directory) if f.endswith('.npy')]

    for npy_file in npy_files:
        pts = np.load(os.path.join(npy_directory, npy_file))
        vertex = [tuple(item) for item in pts]
        vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        ply_file = os.path.splitext(npy_file)[0] + '.ply'
        PlyData([PlyElement.describe(vertex, 'vertex')], text=True).write(os.path.join(ply_directory, ply_file))

def xyz2ply(directory):
    timestamp = datetime.now().strftime('%m%d_%H%M')
    xyz_directory = os.path.join(directory, f'xyz_{timestamp}')
    os.makedirs(xyz_directory)
    ply_directory = os.path.join(directory, f'ply_{timestamp}')
    os.makedirs(ply_directory)

    for file in os.listdir(directory):
        if file.endswith('.xyz'):
            shutil.move(os.path.join(directory, file), os.path.join(xyz_directory, file))

    xyz_files = [f for f in os.listdir(xyz_directory) if f.endswith('.xyz')]

    for xyz_file in xyz_files:
        pts = np.loadtxt(os.path.join(xyz_directory, xyz_file))
        vertex = [tuple(item) for item in pts]
        vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        ply_file = os.path.splitext(xyz_file)[0] + '.ply'
        PlyData([PlyElement.describe(vertex, 'vertex')], text=True).write(os.path.join(ply_directory, ply_file))

def npy2xyz(directory):
    timestamp = datetime.now().strftime('%m%d_%H%M')
    npy_directory = os.path.join(directory, f'npy_{timestamp}')
    os.makedirs(npy_directory)
    xyz_directory = os.path.join(directory, f'xyz_{timestamp}')
    os.makedirs(xyz_directory)

    for file in os.listdir(directory):
        if file.endswith('.npy'):
            shutil.move(os.path.join(directory, file), os.path.join(npy_directory, file))

    npy_files = [f for f in os.listdir(npy_directory) if f.endswith('.npy')]

    for npy_file in npy_files:
        pts = np.load(os.path.join(npy_directory, npy_file))
        xyz_file = os.path.splitext(npy_file)[0] + '.xyz'
        np.savetxt(os.path.join(xyz_directory, xyz_file), pts, fmt='%f')

def xyznpy(directory):
    timestamp = datetime.now().strftime('%m%d_%H%M')
    xyz_directory = os.path.join(directory, f'xyz_{timestamp}')
    os.makedirs(xyz_directory)
    npy_directory = os.path.join(directory, f'npy_{timestamp}')
    os.makedirs(npy_directory)

    for file in os.listdir(directory):
        if file.endswith('.xyz'):
            shutil.move(os.path.join(directory, file), os.path.join(xyz_directory, file))

    xyz_files = [f for f in os.listdir(xyz_directory) if f.endswith('.xyz')]

    for xyz_file in xyz_files:
        pts = np.loadtxt(os.path.join(xyz_directory, xyz_file))
        npy_file = os.path.splitext(xyz_file)[0] + '.npy'
        np.save(os.path.join(npy_directory, npy_file), pts)

def ply2npy(directory):
    timestamp = datetime.now().strftime('%m%d_%H%M')
    ply_directory = os.path.join(directory, f'ply_{timestamp}')
    os.makedirs(ply_directory)
    npy_directory = os.path.join(directory, f'npy_{timestamp}')
    os.makedirs(npy_directory)

    for file in os.listdir(directory):
        if file.endswith('.ply'):
            shutil.move(os.path.join(directory, file), os.path.join(ply_directory, file))

    ply_files = [f for f in os.listdir(ply_directory) if f.endswith('.ply')]

    for ply_file in ply_files:
        plydata = PlyData.read(os.path.join(ply_directory, ply_file))
        pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        npy_file = os.path.splitext(ply_file)[0] + '.npy'
        np.save(os.path.join(npy_directory, npy_file), pts)

def ply2xyz(directory):
    timestamp = datetime.now().strftime('%m%d_%H%M')
    ply_directory = os.path.join(directory, f'ply_{timestamp}')
    os.makedirs(ply_directory)
    xyz_directory = os.path.join(directory, f'xyz_{timestamp}')
    os.makedirs(xyz_directory)

    for file in os.listdir(directory):
        if file.endswith('.ply'):
            shutil.move(os.path.join(directory, file), os.path.join(ply_directory, file))

    ply_files = [f for f in os.listdir(ply_directory) if f.endswith('.ply')]

    for ply_file in ply_files:
        plydata = PlyData.read(os.path.join(ply_directory, ply_file))
        pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        xyz_file = os.path.splitext(ply_file)[0] + '.xyz'
        np.savetxt(os.path.join(xyz_directory, xyz_file), pts, fmt='%f')