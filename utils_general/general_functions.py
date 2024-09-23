import numpy as np
import matplotlib.pyplot as plt
from core_module.default.match_config import internali2internalt


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


def invert_dict_list(dict_in: dict) -> dict:
    """ the class map inverse is needed to look up effectively"""
    dict_out = {}
    for key, value in dict_in.items():
        if value not in dict_out:
            dict_out[value] = [key]
        else:
            dict_out[value].append(key)

    return dict_out

def merge_dicts(dict1, dict2):
    # Create a new dictionary to hold the merged results
    merged_dict = {}

    # Iterate over the keys in both dictionaries
    for key in set(dict1) | set(dict2):
        # If the key is in both dictionaries, merge the lists
        if key in dict1 and key in dict2:
            merged_dict[key] = dict1[key] + dict2[key]
        # If the key is only in dict1, use the value from dict1
        elif key in dict1:
            merged_dict[key] = dict1[key]
        # If the key is only in dict2, use the value from dict2
        else:
            merged_dict[key] = dict2[key]

    return merged_dict


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
