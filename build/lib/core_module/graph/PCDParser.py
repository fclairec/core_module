import importlib
import pandas as pd
import numpy as np
import os
from core_module.utils_general.general_functions import to_open3d, from_open3d
from core_module.pem.PcPEM import PcPEM
from plyfile import PlyData, PlyElement
from pathlib import Path
import laspy
import copy



"""DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(DIR_PATH, ''))
sys.path.insert(0, DIR_PATH)
sys.path.append(os.path.join(DIR_PATH, "partition/cut-pursuit/build/src"))
sys.path.append(os.path.join(DIR_PATH, "partition/ply_c"))
sys.path.append(os.path.join(DIR_PATH, "partition"))"""

# import libcp
"""from ply_c import libply_c

from plyfile import PlyData, PlyElement
from pathlib import Path"""


class PCD:
    def __init__(self, pc_type=None):
        self.points = None
        self.instance_labels = None
        self.type_int = None # class label
        self.discipline_int = None
        self.colors = None
        self.pcd_files = None
        self.point_indices_per_instance = {}
        self.preprocessing_steps_performed = []
        self.points_grouped_by_room = None
        self.pc_type = pc_type

    def parse_pcd(self, pcd_paths, pcd_format):
        # gets parsing information from the config file
        # search for the module of name pcd_formats
        self.pcd_files = pcd_paths

        if pcd_format != "helios":
            module = importlib.import_module("pcd_formats.generic")
        else:
            module = importlib.import_module("pcd_formats.helios")

        for pcd_file in self.pcd_files:
            points, instance_labels = module.load(pcd_file)

            self.incrememtal_pc_addition(points, instance_labels)

    def incrememtal_pc_addition(self, points, instance_labels):
        if self.points is None:
            self.points = points
            self.instance_labels = instance_labels
            self.colors = np.zeros((points.shape[0], 3), dtype='uint8')
            self._update_point_indices_per_instance(np.unique(instance_labels))
        else:
            self.points = np.vstack((self.points, points))
            self.instance_labels = np.vstack((self.instance_labels, instance_labels))
            self.colors = np.vstack((self.colors, np.zeros((points.shape[0], 3), dtype='uint8')))
            self._update_point_indices_per_instance(np.unique(instance_labels))

    def _update_point_indices_per_instance(self, unique_labels, drop_old=False):
        if drop_old:
            self.point_indices_per_instance = {}
        for label in unique_labels:
            self.point_indices_per_instance[label] = np.where(self.instance_labels == label)[0]


    def reindex_labels(self, pem_file=None):
        """
        Reindex the labels of the point cloud to the internal label format that goes from 0 to n_classes.
        This is needed for the SPG method to work.
        :param save_to_file: if true, the project element map is updated with the corresponding index.
        :return:
        """
        # we reindex spg labels multiple times. once when we lead from ifc labels (they might not be continuous),
        # once when we downsample. by downsampling we might comletely remove some labels

        labels_initial, indices = np.unique(self.instance_labels, return_inverse=True)
        # if labels_initial are continuous and start at 0, we do not need to reindex
        if labels_initial[0] == 0 and labels_initial[-1] == len(labels_initial) - 1:
            print(f"labels are already continuous and start at 0. No reindexing needed.")
            return

        old_new_dict = {key: value for key, value in zip(labels_initial, range(len(labels_initial)))}

        pem = PcPEM(self.pc_type)
        pem.load_pem(pem_file)
        pem.reindex_spg_label(old_new_dict, drop=True)
        pem.save_pem(pem_file)

        self.instance_labels = indices.reshape((-1, 1))
        self._update_point_indices_per_instance(np.unique(self.instance_labels))
        self.preprocessing_steps_performed.append('reindex_labels')

    def clean(self):
        print("cleaning point cloud with SOR ")
        o3_pcd = to_open3d(self.points, self.instance_labels)
        clean_o3_pcd, _ = o3_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2)
        self.points, self.instance_labels, self.colors = from_open3d(clean_o3_pcd)
        self._update_point_indices_per_instance(np.unique(self.instance_labels), drop_old=True)
        self.preprocessing_steps_performed.append('cleaned')

    def transform(self, transformation_matrix):
        print("transforming point cloud ... ") # use numpy
        self.points = np.dot(np.hstack((self.points, np.ones((self.points.shape[0], 1)))), transformation_matrix.T)[:, :3]
        self.preprocessing_steps_performed.append('transformed')


    def down_sample(self, voxel_size, save_result=False):
        """as alternative to prune using open3d"""
        print("downsampling point cloud ... ")
        o3_pcd = to_open3d(self.points, self.instance_labels)
        # down_o3_pcd = o3_pcd.voxel_down_sample(voxel_size=voxel_size)
        min_bound = o3_pcd.get_min_bound() - voxel_size
        max_bound = o3_pcd.get_max_bound() + voxel_size
        down_o3_pcd, _, _ = o3_pcd.voxel_down_sample_and_trace(voxel_size, min_bound, max_bound, approximate_class=True)
        self.points, self.instance_labels, self.colors = from_open3d(down_o3_pcd)
        self.preprocessing_steps_performed.append(f'downsampled_{voxel_size}')
        self._update_point_indices_per_instance(np.unique(self.instance_labels), drop_old=True)
        a=0

    def devide_instances_by_clustering(self, pem_file):
        pem = PcPEM(self.pc_type)
        pem.load_pem(pem_file)
        point_indices_per_instance_new = {}
        point_idx_to_remove = []
        points = []
        instances = []
        for id, idxs in self.point_indices_per_instance.items():
            pcd = self.points[idxs]
            o3d_pcd = to_open3d(pcd, np.ones((len(pcd), 1))*id)
            clusters = np.array(o3d_pcd.cluster_dbscan(eps=0.2, min_points=50, print_progress=True))
            clusters_u = np.unique(clusters)
            # all except cluster -1
            clusters_u = clusters_u[clusters_u != -1]
            for i, cluster_id in enumerate(clusters_u):
                pts_in_cluster = np.where(clusters == cluster_id)[0]
                # instanciate new instance.
                new_id = pem.assign_new_guid()

                new_instance = copy.deepcopy(pem.get_instance_entry(id))
                new_instance["guid_int"] = new_id
                pem.add_instance_entry(new_instance)

                # update self.point_indices_per_instance
                point_indices_per_instance_new[new_id] = pts_in_cluster + len(points)
                # update instance_labels
                #self.instance_labels[idxs[pts_in_cluster]] = np.reshape(np.ones(len(pts_in_cluster))*int(new_id), (-1, 1))
                # update points
                points.append(pcd[pts_in_cluster])
                instances.append(np.reshape(np.ones(len(pts_in_cluster))*int(new_id), (-1, 1)))
            # we have to remove the points that do not fit into a cluster
            #clust_minus_1 = np.where(clusters == -1)[0]
            #point_idx_to_remove.extend(idxs[clust_minus_1])


            # delete old instance
            #self.point_indices_per_instance.pop(id)
            pem.remove_instance_entry(id)
        self.point_indices_per_instance = point_indices_per_instance_new
        self.points = np.vstack(points)
        self.instance_labels = np.vstack(instances)
        #self.points = np.delete(self.points, point_idx_to_remove, axis=0)
        #self.instance_labels = np.delete(self.instance_labels, point_idx_to_remove, axis=0)
        pem.save_pem(pem_file)


    def add_scalars_from_pem(self, pem_file, values):
        pem = PcPEM(self.pc_type)
        pem.load_pem(pem_file)

        for value in values:
            scalar_point_template = np.ones(len(self.points))*999
            inst_attr = pem.get_physical_instances(value)
            for inst_id, scalar in inst_attr.items():
                if inst_id not in self.point_indices_per_instance:
                    print(f"instance {inst_id} not covered by point cloud. Replace waypoints if needed")
                    continue
                indices = self.point_indices_per_instance[inst_id]
                scalar_point_template[indices] = scalar
            setattr(self, value, scalar_point_template)
            if any(getattr(self, value)) == 999:
                print(f"scalar {value} not assigned to all points. Review processing steps")



    def output_point_cloud_las(self, downsampled_file_name, output_level="model", instance_dir=None):
        if output_level == "model":
            self._las_assembly(self.points, self.instance_labels, self.type_int, self.discipline_int, downsampled_file_name)
        elif output_level == "instance":
            for id, idxs in self.point_indices_per_instance.items():
                file_name = Path(instance_dir) / f"{id}.las"
                if self.discipline_int is not None:
                    self._las_assembly(self.points[idxs], self.instance_labels[idxs], self.type_int[idxs], self.discipline_int[idxs], file_name)
                else:
                    self._las_assembly(self.points[idxs], self.instance_labels[idxs], self.type_int[idxs], None, file_name)

    def _las_assembly(self, points, instance_labels, type_int, discipline_int, file_name):

        # Create a new LAS file
        header = laspy.LasHeader(version="1.4", point_format=6)
        header.scales = [0.00001, 0.00001, 0.00001]  # X, Y, Z scales

        las = laspy.LasData(header)

        # Assign point coordinates
        las.x = points[:, 0]
        las.y = points[:, 1]
        las.z = points[:, 2]

        # Assign instance labels (s) and type_int (s1) and discipline (s2)
        las.add_extra_dim(laspy.ExtraBytesParams(name="instance", type=np.int32))
        las.add_extra_dim(laspy.ExtraBytesParams(name="type", type=np.int32))

        las.instance = instance_labels[:, 0]
        las.type = type_int

        if discipline_int is not None:
            las.add_extra_dim(laspy.ExtraBytesParams(name="discipline", type=np.int32))
            las.discipline = discipline_int


        # Write the LAS file
        las.write(file_name)


    def to_spg_format(self, computed_clusters=None):
        """ outputs the point cloud in a format needed for the SPG computation.
        :param computed_clusters: if None the point cloud is assumed to be clustered by label. If not None, the point cloud is assumed to be clustered by the computed clusters.
        """
        if computed_clusters is None:
            print("assuming point cloud is clustered by label")
            _in_components = self.instance_labels.flatten()
            cluster = _in_components

            unique_clusters = np.unique(cluster)

            # Get indices for each unique cluster without using a DataFrame
            _components = [np.where(cluster == component_nb)[0].tolist() for component_nb in unique_clusters]
            _components = np.array(_components, dtype='object')

        else:
            print("assuming point cloud is clustered by computed clusters")
            raise NotImplementedError

        labels = self.instance_labels

        # needs to be given: _components have size of the max number in _in_components

        return self.points, labels, _components, _in_components
















if __name__ == '__main__':
    file_path = Path("/home/appuser/input_data/base/built_test/full_points_2023-10-03_11-09-35_downsampled.xyz")
    project_element_map = Path("/home/appuser/input_data/base/built_test/project_element_map.csv")
    PointCloudObject = PCD()
    PointCloudObject.parse_pcd(file_path, "helios")
    PointCloudObject.reindex_labels(pem_file=project_element_map)
    PointCloudObject.output_point_cloud(typed=False)

    format_pcd_4_spg = PointCloudObject.to_spg_format()

    a = 0
