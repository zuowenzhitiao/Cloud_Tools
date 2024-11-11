# -*- coding: utf-8 -*-
"""
@author lizheng
@date 2022年10月31日 00:01:10
@packageName 
@className Geometry_fr_PCQA
@version 1.0.0
@describe
"""

import open3d as o3d
import os
import numpy as np
from open3d.cpu.pybind.geometry import PointCloud
from sklearn.neighbors import KDTree
import time

from torch.utils.data import DataLoader

from ft.coefficient_calu import corr_value


# from ft.ft_code import MiniDataset


def cal_po2po_drms(v_deg, v_or):
    pcd_v_deg = o3d.geometry.PointCloud()
    pcd_v_or = o3d.geometry.PointCloud()
    pcd_v_deg.points = o3d.utility.Vector3dVector(v_deg)
    pcd_v_or.points = o3d.utility.Vector3dVector(v_or)
    knn_tree = o3d.geometry.KDTreeFlann(pcd_v_or)
    dist_sum = 0
    for i in range(v_deg.shape[0]):
        [_, idx_k, _] = knn_tree.search_knn_vector_3d(pcd_v_deg.points[i], 1)
        dist_sum += np.sum((pcd_v_deg.points[i] - pcd_v_or.points[idx_k[0]]) ** 2)
    d_rms = np.sqrt(dist_sum / v_deg.shape[0])
    return d_rms


def cal_po2po_haussdorf(v_deg, v_or):
    pcd_v_deg = o3d.geometry.PointCloud()
    pcd_v_or = o3d.geometry.PointCloud()
    pcd_v_deg.points = o3d.utility.Vector3dVector(v_deg)
    pcd_v_or.points = o3d.utility.Vector3dVector(v_or)
    knn_tree = o3d.geometry.KDTreeFlann(pcd_v_or)
    l2_list = []
    for i in range(v_deg.shape[0]):
        [_, idx_k, _] = knn_tree.search_knn_vector_3d(pcd_v_deg.points[i], 1)
        l2_list.append(np.linalg.norm(pcd_v_deg.points[i] - pcd_v_or.points[idx_k[0]]))
    d_haussdorf = np.max(l2_list)
    return d_haussdorf


def cal_po2pl_drms(v_deg, v_or):
    pcd_v_deg = o3d.geometry.PointCloud()
    pcd_v_or = o3d.geometry.PointCloud()
    pcd_v_deg.points = o3d.utility.Vector3dVector(v_deg)
    pcd_v_or.points = o3d.utility.Vector3dVector(v_or)
    pcd_v_or.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.01, 30))
    knn_tree = o3d.geometry.KDTreeFlann(pcd_v_or)
    dist_sum = 0
    for i in range(v_deg.shape[0]):
        [_, idx_k, _] = knn_tree.search_knn_vector_3d(pcd_v_deg.points[i], 1)
        err_vector = pcd_v_deg.points[i] - pcd_v_or.points[idx_k[0]]
        dist_sum += np.dot(err_vector, pcd_v_or.normals[idx_k[0]]) ** 2
    d_rms = np.sqrt(dist_sum / v_deg.shape[0])
    return d_rms


def cal_po2pl_haussdorf(v_deg, v_or):
    pcd_v_deg = o3d.geometry.PointCloud()
    pcd_v_or = o3d.geometry.PointCloud()
    pcd_v_deg.points = o3d.utility.Vector3dVector(v_deg)
    pcd_v_or.points = o3d.utility.Vector3dVector(v_or)
    pcd_v_or.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.01, 30))
    knn_tree = o3d.geometry.KDTreeFlann(pcd_v_or)
    l2_list = []
    for i in range(v_deg.shape[0]):
        [_, idx_k, _] = knn_tree.search_knn_vector_3d(pcd_v_deg.points[i], 1)
        err_vector = pcd_v_deg.points[i] - pcd_v_or.points[idx_k[0]]
        l2_list.append(np.dot(err_vector, pcd_v_or.normals[idx_k[0]]))
    d_haussdorf = np.max(l2_list)
    return d_haussdorf

def cd_distance(source_cloud, target_cloud):
    # Initialize Chamfer distance module
    source_cloud.requires_grad = True
    # Initialize Chamfer distance module
    chamferDist = ChamferDistance()
    # Compute Chamfer distance
    dist_forward = chamferDist(source_cloud, target_cloud, point_reduction='mean')
    # print("Forward Chamfer distance:", dist_forward.detach().cpu().item())

    # Chamfer distance depends on the direction in which it is computed (as the
    # nearest neighbour varies, in each direction). One can either flip the order
    # of the arguments, or simply use the "reverse" flag.
    dist_backward = chamferDist(source_cloud, target_cloud, point_reduction='mean', reverse=True)
    # print("Backward Chamfer distance:", dist_backward.detach().cpu().item())
    # Or, if you rather prefer, flip the order of the arguments.
    # dist_backward = chamferDist(target_cloud, source_cloud, point_reduction='mean')
    # print("Backward Chamfer distance:", dist_backward.detach().cpu().item())

    # As a sanity check, chamfer distance between a pointcloud and itself must be
    # zero.
    # dist_self = chamferDist(source_cloud, source_cloud)
    # print("Chamfer distance (self):", dist_self.detach().cpu().item())
    # dist_self = chamferDist(target_cloud, target_cloud)
    # print("Chamfer distance (self):", dist_self.detach().cpu().item())

    return dist_backward.detach().cpu().item() + dist_forward.detach().cpu().item()

class AngularSimilarity:
    def __init__(self, v_deg, v_or):
        """
        return angular similarity between point clouds
        :param v_deg: point cloud A
        :param v_or: point cloud B
        """
        self.a = v_deg
        self.b = v_or
        self.radius = 0.1
        self.max_nn = 50

        pcd_or = o3d.geometry.PointCloud()
        pcd_or.points = o3d.utility.Vector3dVector(self.a)
        pcd_deg = o3d.geometry.PointCloud()
        pcd_deg.points = o3d.utility.Vector3dVector(self.b)

        pcd_or.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(self.radius, self.max_nn))
        pcd_deg.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(self.radius, self.max_nn))

        self.a_ = np.array(pcd_or.normals)
        self.b_ = np.array(pcd_deg.normals)

    @staticmethod
    def compute_average(a, b, a_, b_):
        kdt = KDTree(b, leaf_size=30, metric="euclidean")
        indices = kdt.query(a, k=1, return_distance=False)

        s = 1 - (2 * np.arccos(abs(np.sum((a_ * b_[indices].reshape(-1, 3)), axis=1) / (
                np.linalg.norm(a_, axis=1) * np.linalg.norm(b_[indices].reshape(-1, 3), axis=1))))) / np.pi

        rmse = np.sqrt(np.mean(s ** 2))

        mean = np.mean(s ** 2)

        haussdorf = np.max(s ** 2)

        return mean, rmse, haussdorf

    def compute_distance(self):
        _, rmse1, hf1 = self.compute_average(self.a, self.b, self.a_, self.b_)
        _, rmse2, hf2 = self.compute_average(self.b, self.a, self.b_, self.a_)
        rmse = np.max((rmse1, rmse2))
        haussdorf = np.max((hf1, hf2))

        x = np.concatenate((self.a[:, 0], self.b[:, 0]), axis=0)
        y = np.concatenate((self.a[:, 1], self.b[:, 1]), axis=0)
        z = np.concatenate((self.a[:, 2], self.b[:, 2]), axis=0)
        x_max = np.array([np.max(x), np.max(y), np.max(z)])
        x_min = np.array([np.min(x), np.min(y), np.min(z)])
        p = np.linalg.norm(x_max - x_min)
        pnsr_geom = 10. * np.log10(p ** 2 / rmse ** 2)

        return rmse, haussdorf, pnsr_geom


class Metric_Po2Po:
    def __init__(self, v_deg, v_or):
        self.v_deg = v_deg
        self.v_or = v_or

    @staticmethod
    def compute(v_deg, v_or):
        kdt = KDTree(v_or, leaf_size=30, metric="euclidean")
        indices = kdt.query(v_deg, k=1, return_distance=False)
        err_vector = v_deg - v_or[indices].reshape(-1, 3)

        s = np.sum(err_vector ** 2, axis=1)
        mean = np.mean(s)

        s = np.sum(err_vector ** 2, axis=1)
        rmse = np.sqrt(np.mean(s))

        s = np.linalg.norm(err_vector ** 2, axis=1)
        haussdorf = np.max(s)
        return mean, rmse, haussdorf

    def compute_distance(self):
        _, rmse1, hf1 = self.compute(self.v_deg, self.v_or)
        _, rmse2, hf2 = self.compute(self.v_or, self.v_deg)
        rmse = np.max((rmse1, rmse2))
        haussdorf = np.max((hf1, hf2))

        x = np.concatenate((self.v_deg[:, 0], self.v_or[:, 0]), axis=0)
        y = np.concatenate((self.v_deg[:, 1], self.v_or[:, 1]), axis=0)
        z = np.concatenate((self.v_deg[:, 2], self.v_or[:, 2]), axis=0)
        x_max = np.array([np.max(x), np.max(y), np.max(z)])
        x_min = np.array([np.min(x), np.min(y), np.min(z)])
        p = np.linalg.norm(x_max - x_min)
        pnsr_geom = 10. * np.log10(p ** 2 / rmse ** 2)
        return rmse, haussdorf, pnsr_geom


class Metric_Po2Pl:
    def __init__(self, v_deg, v_or):
        self.v_deg = v_deg
        self.v_or = v_or

    @staticmethod
    def compute(v_deg, v_or):
        kdt = KDTree(v_or, leaf_size=30, metric="euclidean")
        indices = kdt.query(v_deg, k=1, return_distance=False)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(v_or)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.01, 30))
        normals = np.asarray(pcd.normals)
        err_vector = v_deg - v_or[indices].reshape(-1, 3)
        s = np.sum(err_vector * normals[indices].reshape(-1, 3), axis=1)

        mean = np.mean(s ** 2)

        rmse = np.sqrt(np.mean(s ** 2))

        haussdorf = np.max(s ** 2)
        return mean, rmse, haussdorf

    def compute_distance(self):
        _, rmse1, hf1 = self.compute(self.v_deg, self.v_or)
        _, rmse2, hf2 = self.compute(self.v_or, self.v_deg)

        rmse = np.max((rmse1, rmse2))
        haussdorf = np.max((hf1, hf2))

        x = np.concatenate((self.v_deg[:, 0], self.v_or[:, 0]), axis=0)
        y = np.concatenate((self.v_deg[:, 1], self.v_or[:, 1]), axis=0)
        z = np.concatenate((self.v_deg[:, 2], self.v_or[:, 2]), axis=0)
        x_max = np.array([np.max(x), np.max(y), np.max(z)])
        x_min = np.array([np.min(x), np.min(y), np.min(z)])
        p = np.linalg.norm(x_max - x_min)
        pnsr_geom = 10. * np.log10(p ** 2 / rmse ** 2)
        return rmse, haussdorf, pnsr_geom


def metric_po2po(v_deg, v_or):
    d_symmetric_rms = np.max(
        [cal_po2po_drms(v_deg, v_or), cal_po2po_drms(v_or, v_deg)]
    )
    d_symmetric_haussdorf = np.max(
        [cal_po2po_haussdorf(v_deg, v_or), cal_po2po_haussdorf(v_or, v_deg)]
    )

    # metric psnr_geom
    x = np.concatenate((v_deg[:, 0], v_or[:, 0]), axis=0)
    y = np.concatenate((v_deg[:, 1], v_or[:, 1]), axis=0)
    z = np.concatenate((v_deg[:, 2], v_or[:, 2]), axis=0)
    x_max = np.array([np.max(x), np.max(y), np.max(z)])
    x_min = np.array([np.min(x), np.min(y), np.min(z)])
    p = np.linalg.norm(x_max - x_min)
    pnsr_geom = 10. * np.log10(p ** 2 / d_symmetric_rms ** 2)

    return d_symmetric_rms, d_symmetric_haussdorf, pnsr_geom


def metric_po2pl(v_deg, v_or):
    d_symmetric_rms = np.max(
        [cal_po2pl_drms(v_deg, v_or), cal_po2pl_drms(v_or, v_deg)]
    )
    d_symmetric_haussdorf = np.max(
        [cal_po2pl_haussdorf(v_deg, v_or), cal_po2pl_haussdorf(v_or, v_deg)]
    )
    # metric psnr_geom
    x = np.concatenate((v_deg[:, 0], v_or[:, 0]), axis=0)
    y = np.concatenate((v_deg[:, 1], v_or[:, 1]), axis=0)
    z = np.concatenate((v_deg[:, 2], v_or[:, 2]), axis=0)
    x_max = np.array([np.max(x), np.max(y), np.max(z)])
    x_min = np.array([np.min(x), np.min(y), np.min(z)])
    p = np.linalg.norm(x_max - x_min)
    pnsr_geom = 10. * np.log10(p ** 2 / d_symmetric_rms ** 2)

    return d_symmetric_rms, d_symmetric_haussdorf, pnsr_geom


if __name__ == '__main__':
    '''
    data_root = r'E:\homegate\R_Quality_Assessment\Dataset\G-PCD\stimuli\D01'
    bunny_or = o3d.io.read_point_cloud(
        os.path.join(data_root, 'bunny.ply'))  # type:o3d.cpu.pybind.geometry.PointCloud
    bunny_l1 = o3d.io.read_point_cloud(
        os.path.join(data_root, 'bunny_D01_L01.ply'))  # type:o3d.cpu.pybind.geometry.PointCloud
    bunny_l2 = o3d.io.read_point_cloud(
        os.path.join(data_root, 'bunny_D01_L02.ply'))  # type:o3d.cpu.pybind.geometry.PointCloud
    bunny_l3 = o3d.io.read_point_cloud(
        os.path.join(data_root, 'bunny_D01_L03.ply'))  # type:o3d.cpu.pybind.geometry.PointCloud
    bunny_l4 = o3d.io.read_point_cloud(
        os.path.join(data_root, 'bunny_D01_L04.ply'))  # type:o3d.cpu.pybind.geometry.PointCloud

    # print("\td_symmetric_rms, d_symmetric_haussdorf, pnsr_geom: Po2Po")
    t1 = time.time()
    # breakpoint()
    print("L1:", metric_po2po(np.asarray(bunny_l1.points), np.asarray(bunny_or.points)))
    print("L2:", metric_po2po(np.asarray(bunny_l2.points), np.asarray(bunny_or.points)))
    print("L3:", metric_po2po(np.asarray(bunny_l3.points), np.asarray(bunny_or.points)))
    t2 = time.time()

    print("L1:", Metric_Po2Po(np.asarray(bunny_l1.points), np.asarray(bunny_or.points)).compute_distance())
    print("L2:", Metric_Po2Po(np.asarray(bunny_l2.points), np.asarray(bunny_or.points)).compute_distance())
    print("L3:", Metric_Po2Po(np.asarray(bunny_l3.points), np.asarray(bunny_or.points)).compute_distance())
    t3 = time.time()
    print(t2 - t1, t3 - t2)

    # print("L4:", metric_po2po(np.asarray(bunny_l4.points), np.asarray(bunny_or.points)))
    #
    # print("\td_symmetric_rms, d_symmetric_haussdorf, pnsr_geom: Po2Pl")
    t1 = time.time()
    print("L1:", metric_po2pl(np.asarray(bunny_l1.points), np.asarray(bunny_or.points)))
    print("L2:", metric_po2pl(np.asarray(bunny_l2.points), np.asarray(bunny_or.points)))
    print("L3:", metric_po2pl(np.asarray(bunny_l3.points), np.asarray(bunny_or.points)))
    print("L4:", metric_po2pl(np.asarray(bunny_l4.points), np.asarray(bunny_or.points)))
    t2 = time.time()
    print("L1:", Metric_Po2Pl(np.asarray(bunny_l1.points), np.asarray(bunny_or.points)).compute_distance())
    print("L2:", Metric_Po2Pl(np.asarray(bunny_l2.points), np.asarray(bunny_or.points)).compute_distance())
    print("L3:", Metric_Po2Pl(np.asarray(bunny_l3.points), np.asarray(bunny_or.points)).compute_distance())
    print("L4:", Metric_Po2Pl(np.asarray(bunny_l4.points), np.asarray(bunny_or.points)).compute_distance())
    t3 = time.time()
    '''


    # 江坤
    data_path_root = r'D:\wcc\PCQA-NEW\ft\dataset\test1.txt'
    groundtruth_root = r'D:\wcc\PCQA-NEW\groundtruth'
    # 覆盖PCQA.csv文件
    with open(r'D:\wcc\PCQA-NEW\PCQA.csv', 'w') as f:
        f.write('filename' + ',' + 'p2p_rmse'+',' + 'p2p_haussdorf'+ ','+ 'p2p_pnsr_geom' + ','\
                + 'p2f_rmse'+',' + 'p2f_haussdorf'+ ','+ 'p2f_pnsr_geom' + '\n')
    # 计算directory中的所有文件的指标
    p2p_rmse = []
    p2p_haussdorf = []
    p2p_pnsr_geom = []
    p2f_rmse = []
    p2f_haussdorf = []
    p2f_pnsr_geom = []
    opinion_score = []
    # 读取test1.txt文件中文件名
    with open(data_path_root, 'r') as f:
        lines = f.readlines()
        for line in lines:
            stimuli_file_name = line.split(' ')[0].split('\\')[-1]
            label = line.split(' ')[1]
            opinion_score.append(label)
            # 读取文件
            pcd_stimuli = o3d.io.read_point_cloud(line.split(' ')[0])
            # 读取ground_truth文件夹下文件
            with open(groundtruth_root + '\\ground_truth.txt', 'r') as f:
                gt_lines = f.readlines()
                for gt_line in gt_lines:
                    gt_name = gt_line.split('.')[0]
                    # 匹配文件名字，如果stimuli_file_name名字包含gt_name，则计算指标
                    if stimuli_file_name.startswith(gt_name):
                        pcd_gt = o3d.io.read_point_cloud(os.path.join(groundtruth_root, gt_line.split('\n')[0]))
                        # 计算指标
                        metric_po2po = Metric_Po2Po(np.asarray(pcd_stimuli.points), np.asarray(pcd_gt.points)).compute_distance()
                        p2p_rmse.append(metric_po2po[0])
                        p2p_haussdorf.append(metric_po2po[1])
                        p2p_pnsr_geom.append(metric_po2po[2])
                        metric_po2pl = Metric_Po2Pl(np.asarray(pcd_stimuli.points), np.asarray(pcd_gt.points)).compute_distance()
                        p2f_rmse.append(metric_po2pl[0])
                        p2f_haussdorf.append(metric_po2pl[1])
                        p2f_pnsr_geom.append(metric_po2pl[2])
                        print(line.split(' ')[0], gt_name, metric_po2po, metric_po2pl)
                        # 把Metric_po2po,Metric_po2pl参数保存到PCQA.csv文件中
                        with open(r'D:\wcc\PCQA-NEW\PCQA.csv', 'a') as f:
                            f.write(stimuli_file_name + ',' + str(metric_po2po[0]) + ',' + str(metric_po2po[1])+',' + str(metric_po2po[2])\
                                    +',' + str(metric_po2pl[0]) + ',' + str(metric_po2pl[1]) +',' + str(metric_po2pl[2])+ '\n')
    # 对每个指标计算PLCC,SRCC,KRCC,RMSE，保存到PREDICTION_PERFORMANCE.csv文件中
    # 覆盖PREDICTION_PERFORMANCE.csv文件
    with open(r'D:\wcc\PCQA-NEW\PREDICTION_PERFORMANCE.csv', 'w') as f:
        f.write('method' + ',' + 'PLCC' + ',' + 'SRCC' + ',' + 'KRCC' + ',' + 'RMSE' + '\n')
    opinion_score = np.array(opinion_score, dtype=float)
    p2p_rmse = np.array(p2p_rmse, dtype=float)
    p2p_haussdorf = np.array(p2p_haussdorf, dtype=float)
    p2p_pnsr_geom = np.array(p2p_pnsr_geom, dtype=float)
    p2f_rmse = np.array(p2f_rmse, dtype=float)
    p2f_haussdorf = np.array(p2f_haussdorf, dtype=float)
    p2f_pnsr_geom = np.array(p2f_pnsr_geom, dtype=float)
    print(len(opinion_score), len(p2p_rmse), len(p2p_haussdorf), len(p2p_pnsr_geom), len(p2f_rmse), len(p2f_haussdorf), len(p2f_pnsr_geom))
    methods_scores = {'p2p_rmse': p2p_rmse, 'p2p_haussdorf': p2p_haussdorf, 'p2p_pnsr_geom': p2p_pnsr_geom, \
                      'p2f_rmse': p2f_rmse, 'p2f_haussdorf': p2f_haussdorf, 'p2f_pnsr_geom': p2f_pnsr_geom}
    for method, score in methods_scores.items():
        print(len(opinion_score), len(score))
        PLCC, SRCC, KRCC, RMSE = corr_value(opinion_score, score, fit_flag=False)
        with open(r'D:\wcc\PCQA-NEW\PREDICTION_PERFORMANCE.csv', 'a') as f:
            f.write(method + ',' + str(PLCC) + ',' + str(SRCC) + ',' + str(KRCC) + ',' + str(RMSE) + '\n')



    # print(t2 - t1, t3 - t2)

    # ang = AngularSimilarity(np.asarray(bunny_or.points), np.asarray(bunny_l1.points))
    # print("L1:", ang.compute_distance())
    #
    # ang = AngularSimilarity(np.asarray(bunny_or.points), np.asarray(bunny_l2.points))
    # print("L2:", ang.compute_distance())
    #
    # ang = AngularSimilarity(np.asarray(bunny_or.points), np.asarray(bunny_l3.points))
    # print("L3:", ang.compute_distance())

    # # visualization
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name='Bunny')
    # vis.get_render_option().point_size = 1
    # opt = vis.get_render_option()
    # opt.background_color = np.asarray([1, 1, 1])
    # vis.add_geometry(bunny_or)
    #
    # vis.run()
    # vis.destroy_window()