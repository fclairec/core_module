import importlib
import pandas as pd
import numpy as np
import os
from core_module.utils.general_functions import to_open3d, from_open3d

from plyfile import PlyData, PlyElement
from pathlib import Path

from core_module.pem.io import load_pem

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
        self.colors = None
        self.pcd_files = None
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

        self.pcd_files = module.init_pcd_format(pcd_paths)
        points = []
        instance_labels = []

        for pcd_file in self.pcd_files:
            generic_load = pd.read_csv(pcd_file, sep=module.seperator, header=0)
            #set header as 'X Y Z intensity echoWidth returnNumber numberOfReturns fullwaveIndex hitObjectId class gpsTime\n')
            generic_load.columns = ['X', 'Y', 'Z', 'intensity', 'echoWidth', 'returnNumber', 'numberOfReturns', 'fullwaveIndex', 'hitObjectId', 'class', 'gpsTime']
            points_i = np.ascontiguousarray(generic_load[module.point_column_names], dtype='float32')
            instance_labels_i = np.ascontiguousarray(generic_load[module.instance_label_column_name], dtype='uint32')
            points.append(points_i)
            instance_labels.append(instance_labels_i)
        self.points = np.vstack(points)
        self.instance_labels = np.vstack(instance_labels)
        if module.colors_column_names is not None:
            self.colors = np.ascontiguousarray(generic_load[module.colors_column_names], dtype='uint8')
        else:
            self.colors = np.zeros((self.points.shape[0], 3), dtype='uint8')
        del generic_load

    def incrememtal_pc_addition(self, points, instance_labels):
        if self.points is None:
            self.points = points
            self.instance_labels = instance_labels
            self.colors = np.zeros((points.shape[0], 3), dtype='uint8')
        else:
            self.points = np.vstack((self.points, points))
            self.instance_labels = np.vstack((self.instance_labels, instance_labels))
            self.colors = np.vstack((self.colors, np.zeros((points.shape[0], 3), dtype='uint8')))

    def reindex_labels(self, save_id_to_file=True, project_element_map_file=None, step=1):
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
        new_instance_labels = indices.reshape((-1, 1))

        if save_id_to_file:
            # check if project element file exists
            new_vs_old = pd.DataFrame(
                {'old': self.instance_labels.flatten(), 'new': new_instance_labels.flatten()})

            if os.path.exists(project_element_map_file):
                # load project element map
                project_element_map = pd.read_csv(project_element_map_file, sep=',', header=0, index_col=0)

                # if project_element_map already has column "spg_label", we merge on that. this happens when we are in
                # the second reindexing phase
                if step == 2:
                    left_key = "spg_label"
                else:
                    left_key = "guid_int"

                result = project_element_map.merge(new_vs_old, left_on=left_key, right_on='old', how='left')
                result = result.drop_duplicates(subset=['guid_int', 'new'], keep='first').reset_index(drop=True)

                assert result.shape[0] == project_element_map.shape[0]


                #result['new'] = result['new'].fillna(-1).astype(int)
                if step==2:
                    # incase left from before
                    result = result.drop("spg_label", axis=1)

                result = result.rename(columns={'new': 'spg_label'}).drop('old', axis=1)

                # remove rows with spg_label = nan
                result = result[result['spg_label'].notna()]
                # reset index
                result = result.reset_index(drop=True)
                result['spg_label'] = result['spg_label'].astype(int)

                # delete the old project_element_map_file
                os.remove(project_element_map_file)

                result.to_csv(project_element_map_file, index=True, header=True, sep=',')

                a = 0
            else:
                # rename column instance_label_int_new to spg_label in new_vs_old
                new_vs_old = new_vs_old.rename(columns={'new': 'spg_label', 'old': 'guid_int'})
                # save to file
                new_vs_old.to_csv(project_element_map_file, index=True, header=True, sep=',')

        self.instance_labels = new_instance_labels
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

    def add_scalars_from_map(self, project_element_map_file, value, id_column="spg_label"):
        """ find the class type point cloud according to the instance label"""
        if self.pc_type == "bim_sampled":
            project_element_map = load_pem(project_element_map_file, mode="d")
        else:
            project_element_map = pd.read_csv(project_element_map_file, sep=',', header=0, index_col=0)
            # delete all sp_label that are -1. and make spg_label the index
            project_element_map = project_element_map[project_element_map[id_column] != -1].set_index(id_column)
        # find the class type point cloud according to the instance label
        scalars = []
        if value == "space_id":
            print(
                "careful when visualizing the space_id. Doors are assigned to muliples spaces and will receive a numpy based value here.")
        for i in self.instance_labels:
            try:
                s = project_element_map.loc[i[0], value]
            except:
                print(f"scalar value {value}:{i} not found in project element map. Review processing steps")
            scalars.append(s)
        return scalars

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
            prem_room.to_csv(subset_pem_filename, index=True, header=True, sep=',')

            room_pcd.reindex_labels(True, subset_pem_filename, 2)
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
    PointCloudObject.reindex_labels(save_id_to_file=True, project_element_map_file=project_element_map)
    PointCloudObject.output_point_cloud(typed=False)

    format_pcd_4_spg = PointCloudObject.to_spg_format()

    a = 0
