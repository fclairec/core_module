import numpy as np
import matplotlib.pyplot as plt
from core_module.default_config.config import internali2internalt
from core_module.default_config.config import current_models
from core_module.default_config.default_cfg import update_config_value
import os
import shutil


def invert_dict(dict_in: dict) -> dict:
    """ the class map inverse is needed to look up effectively"""
    dict_inverse = {}
    for key, value in dict_in.items():
        for entry in value:
            dict_inverse[entry] = key
    return dict_inverse


def invert_dict_simple(dict_in: dict) -> dict:
    """ the class map inverse is needed to look up effectively"""
    dict_inverse = {}
    for key, value in dict_in.items():
        dict_inverse[value] = key
    return dict_inverse


def map_dict_keys(dict_in: dict, dict_map: dict) -> dict:
    """ map the keys of dict one by another dict"""
    dict_out = {}
    for key, value in dict_in.items():
        try:
            dict_out[dict_map[key]] = value
        except:
            raise KeyError(f"Key -{key}- int2txt, please revise config_global file.")
    return dict_out


def select_ifc_classes_per_discipline(disciplines: list, parsed_ifc_classes_dict: dict) -> list:
    """ select the ifc classes that are relevant for the given disciplines"""
    ifc_classes = []
    for discipline in disciplines:
        ifc_classes = ifc_classes + parsed_ifc_classes_dict[discipline]
    return ifc_classes

def get_int_classes_per_discipline(ifc_parsing_dict: dict) -> dict:
    """ select the ifc classes that are relevant for the given disciplines"""
    internalt2internali = invert_dict_simple(internali2internalt)
    int_classes = {}
    for discipline, class_list in ifc_parsing_dict.items():
        class_list = [internalt2internali[class_i_t] for class_i_t in class_list.keys()]
        int_classes[discipline] = class_list
    return int_classes


def labels_to_colors_255(labels, max_label):
    " returns a dictionary with the label as key and the color as value"
    unique_labels = np.unique(labels)
    normalized_values = unique_labels / (max_label if max_label > 0 else 1)
    colors = plt.get_cmap("tab20")(normalized_values)
    rgb_255_per_label = (colors[:, :3] * 255).astype(np.uint8)  # Convert to 0-255 range and ensure it's integer type

    # Create a dictionary mapping
    label_to_color_dict = {label: color for label, color in zip(unique_labels, rgb_255_per_label)}

    # Inflate the colors to the number of points per label
    rgb_255 = np.array([rgb_255_per_label[label] for label in labels]).reshape(-1, 3)
    return rgb_255, label_to_color_dict


def color_to_label(color, lookup_table):
    # Round the color values to handle precision issues
    rounded_color = tuple(round(val, 4) for val in color)
    return lookup_table.get(rounded_color, -1)


def to_open3d(points, labels):
    """ make open3d point cloud from pandas dataframe
    :param pcd: pandas dataframe with columns X,Y,Z,class"""
    import open3d as o3d
    import numpy as np

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    # hack the pcd['class'] label into red channel of colors
    hack_colors = np.zeros((points.shape[0], 3), dtype='int')
    hack_colors[:, 0] = labels.flatten()
    point_cloud.colors = o3d.utility.Vector3dVector(hack_colors)
    return point_cloud


def from_open3d(point_cloud):
    """ make pandas dataframe from open3d point cloud
    :param point_cloud: open3d point cloud"""
    import numpy as np

    points = np.asarray(point_cloud.points)
    # hack the pcd['class'] label into red channel of colors
    hack_colors = np.asarray(point_cloud.colors)
    labels = hack_colors[:, 0]
    # set type to int
    # delete all points that have a class of type float
    float_locs = labels % 1 == 0
    points = points[float_locs]
    labels = labels[float_locs]
    colors = np.zeros((points.shape[0], 3), dtype='uint8')
    labels = labels.astype(int).reshape((-1, 1))

    return points, labels, colors


def prepare_project_dirs(cfg):
    if not os.path.exists(cfg.experiment_dir):
        os.makedirs(cfg.experiment_dir)

    project_configs = []
    for project in cfg.building_projects:
        if project in current_models.keys() and cfg.design.ifc_file is None:
            ifc_file_name = current_models[project]
        elif cfg.design.ifc_file is not None:
            print("overwriting project name with ifc file name")
            ifc_file_name_d = cfg.design.ifc_file
            ifc_file_name_b = cfg.built.ifc_file
        else:
            print(f"Project {project} not found in current_models. Please provide the ifc file name.")
            raise SystemExit
        d_ifc = os.path.join(cfg.root_root_dir, "ifc_models", ifc_file_name_d)
        b_ifc = os.path.join(cfg.root_root_dir, "ifc_models", ifc_file_name_b)
        waypoint_file = os.path.join(cfg.root_root_dir, "waypoint_files", project+f"_{cfg.built.waypoints}")

        cfg = update_config_value(cfg, ["project_name", project,
                                  "root_dir", os.path.join(cfg.experiment_dir, project),
                                  "design.ifc_file", ifc_file_name_d,
                                  "built.ifc_file", ifc_file_name_b])




        d_dir = os.path.join(cfg.root_dir, "d")
        b_dir = os.path.join(cfg.root_dir, "b")
        if not os.path.exists(d_dir):
            os.makedirs(d_dir)
            os.makedirs(b_dir)

        # copy ifc into to d and b folders
        shutil.copy(d_ifc, os.path.join(d_dir, ifc_file_name_d))
        shutil.copy(b_ifc, os.path.join(b_dir, ifc_file_name_b))
        shutil.copy(waypoint_file, cfg.built.manual_waypoints_selection)


        # make d and b folders and

        yield cfg



    """else:
        print(f"Directory {cfg.experiment_dir} already exists.")
        # end the program
        raise SystemExit"""