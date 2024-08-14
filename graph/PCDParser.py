import importlib
import pandas as pd
import numpy as np
import os
from core_module.utils_general.general_functions import to_open3d, from_open3d
from core_module.pem.PcPEM import PcPEM
from plyfile import PlyData, PlyElement
from pathlib import Path
import laspy


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

        module_name = "pcd_formats." + pcd_format
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            module = 0
            print(f"No module named '{module_name}' available.")


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

    def _update_point_indices_per_instance(self, unique_labels):
        for label in unique_labels:
            self.point_indices_per_instance[label] = np.where(self.instance_labels == label)[0]




    def reindex_labels(self, pem_file=None, step=1):
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

        pem.reindex_spg_label(old_new_dict)

        pem.save_pem(pem_file)

        self.instance_labels = indices.reshape((-1, 1))
        self._update_point_indices_per_instance(np.unique(self.instance_labels))
        self.preprocessing_steps_performed.append('reindex_labels')

    def clean(self):
        print("cleaning point cloud with SOR ")
        o3_pcd = to_open3d(self.points, self.instance_labels)
        clean_o3_pcd, _ = o3_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2)
        self.points, self.instance_labels, self.colors = from_open3d(clean_o3_pcd)
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
        self._update_point_indices_per_instance(np.unique(self.instance_labels))

    def output_point_cloud(self, project_element_map_file, downsampled_file_name, col_id="spg_label"):
        # prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('s', 'i4'), ('s1', 'i4'), ('s2', 'i4')]
        prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('s', 'i4'), ('s1', 'i4')]
        vertex_all = np.empty(len(self.points), dtype=prop)
        for i_prop in range(0, 3):
            vertex_all[prop[i_prop][0]] = self.points[:, i_prop]
        """for i_prop in range(0, 3):
            vertex_all[prop[i_prop + 3][0]] = self.colors[:, i_prop]"""

        # spg labels
        vertex_all[prop[3][0]] = self.instance_labels[:, 0]

        # add semantic type as additional channel
        scalars = self.add_scalars_from_map(project_element_map_file=project_element_map_file, value="type_int", id_column=col_id)
        vertex_all[prop[4][0]] = scalars

        """# add room id as additional channel
        scalars = self.add_scalars_from_map(project_element_map_file=project_element_map_file, value="space_id")
        vertex_all[prop[8][0]] = scalars
        """
        # add

        ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
        # assemble file name with all preprocessing steps in self.preprocessing_steps_performed concatenated to string
        ply.write(downsampled_file_name)

    def add_scalars_from_pem(self, pem_file, values):
        pem = PcPEM(self.pc_type)
        pem.load_pem(pem_file)

        for value in values:
            scalar_point_template = np.ones(len(self.points))*999
            inst_attr = pem.get_physical_instances(value)
            for inst_id, scalar in inst_attr.items():
                indices = self.point_indices_per_instance[inst_id]
                scalar_point_template[indices] = scalar
            setattr(self, value, scalar_point_template)
            if any(getattr(self, value)) == 999:
                print(f"scalar {value} not assigned to all points. Review processing steps")



    def output_point_cloud_las(self, downsampled_file_name):

        # Create a new LAS file
        header = laspy.LasHeader(version="1.4", point_format=6)
        header.scales = [0.00001, 0.00001, 0.00001]  # X, Y, Z scales

        las = laspy.LasData(header)

        # Assign point coordinates
        las.x = self.points[:, 0]
        las.y = self.points[:, 1]
        las.z = self.points[:, 2]

        # Assign instance labels (s) and type_int (s1) and discipline (s2)
        las.add_extra_dim(laspy.ExtraBytesParams(name="instance", type=np.int32))
        las.add_extra_dim(laspy.ExtraBytesParams(name="type", type=np.int32))
        las.add_extra_dim(laspy.ExtraBytesParams(name="discipline", type=np.int32))
        las.instance = self.instance_labels[:, 0]
        las.type = self.type_int
        las.discipline = self.discipline_int


        # Write the LAS file
        las.write(downsampled_file_name)


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




    def split_pcd_by_room(self, built_pem_file, b_features_file, subset_pcd_template, subset_pem_template, subset_feat_template, super_points_by_room):
        """ function outputs a point cloud object by room into a seperate folder"""
        sub_clouds = []
        room_ids = []
        geom_features = pd.read_csv(b_features_file, sep=',', header=0, index_col=0)
        for room_id, spg_labels in super_points_by_room.items():
            point_selection = np.isin(self.instance_labels.flatten(), spg_labels)
            room_points = self.points[point_selection]
            room_instance_labels = self.instance_labels[point_selection]

            room_pcd = PCD()
            room_pcd.points = room_points
            room_pcd.instance_labels = room_instance_labels

            subset_pcd_filename = str(subset_pcd_template).format(room_id, room_id)
            subset_pem_filename = str(subset_pem_template).format(room_id, room_id)
            if not os.path.exists(subset_pcd_filename):
                os.makedirs(Path(subset_pcd_filename).parent)
            else:
                return ValueError(f"subset path {subset_pcd_filename} already exists. Review processing steps")

            pem = pd.read_csv(built_pem_file, sep=',', header=0, index_col=0)
            prem_room = pem[pem['spg_label'].isin(spg_labels)]
            #prem_room.to_csv(subset_pem_filename, index=True, header=True, sep=',')
            save_pem(subset_pem_filename, prem_room, mode="b")

            room_pcd.reindex_labels(subset_pem_filename, 2)
            room_pcd.output_point_cloud(subset_pem_filename, subset_pcd_filename, col_id="spg_label")

            # save subset features to subset_feat
            subset_feat_filename = str(subset_feat_template).format(room_id, room_id)
            subset_feats = geom_features.loc[spg_labels]
            # write to csv with new index
            subset_feats.reset_index(drop=True, inplace=True)
            subset_feats.to_csv(subset_feat_filename, index=True, header=True, sep=',')


            sub_clouds.append(room_pcd)
            room_ids.append(room_id)

        return sub_clouds, room_ids













if __name__ == '__main__':
    file_path = Path("/home/appuser/input_data/base/built_test/full_points_2023-10-03_11-09-35_downsampled.xyz")
    project_element_map = Path("/home/appuser/input_data/base/built_test/project_element_map.csv")
    PointCloudObject = PCD()
    PointCloudObject.parse_pcd(file_path, "helios")
    PointCloudObject.reindex_labels(pem_file=project_element_map)
    PointCloudObject.output_point_cloud(typed=False)

    format_pcd_4_spg = PointCloudObject.to_spg_format()

    a = 0
