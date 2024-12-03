import numpy as np


def get_geometric_features(cfg, pcd, bbox=None):
    feature_dict = {}

    pcd += np.random.normal(0, 0.01, pcd.shape)

    for task in cfg:
        method = globals().get(task)
        if method is not None and callable(method):
            features = method(pcd = pcd, bbox=bbox)
            if task == 'pca_and_extents':
                # the last three values are the extents
                feature_dict["extents"] = features[-3:]
                feature_dict["pca"] = features[:-3]
                feature_dict["normal"] = np.cross(feature_dict["pca"][:3], feature_dict["pca"][3:6])
            else:
                feature_dict[task] = features
        else:
            print(f"Method does not exist - {task} skipped")

    return feature_dict


def centroid(pcd, bbox):
    """ if bbox is supplies we prefer to use this"""
    if bbox is not None:
        center_point = np.array((bbox.Center().X(), bbox.Center().Y(), bbox.Center().Z()))
    else:
        center_point = pcd.mean(axis=0)
    return center_point


def pca_and_extents(pcd, bbox):

    if bbox is not None:
        dir_xyz = [[bbox.XDirection().X(), bbox.XDirection().Y(), bbox.XDirection().Z()]
                     , [bbox.YDirection().X(), bbox.YDirection().Y(), bbox.YDirection().Z()]
                     ,[bbox.ZDirection().X(), bbox.ZDirection().Y(), bbox.ZDirection().Z()]]

        extent_xyz_n = [2*bbox.XHSize(), 2*bbox.YHSize(), 2*bbox.ZHSize()]
        extent_xyz = [round(extent, 3) for extent in extent_xyz_n]

        # from 1, 2,3 . descending order
        pca_order = np.argsort(extent_xyz)[::-1]

        components = np.array([dir_xyz[i] for i in pca_order])
        lengths = np.array([extent_xyz[i] for i in pca_order])

    else:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=3)
        pca.fit(pcd)

        components = pca.components_

        centered_data = pcd - np.mean(pcd, axis=0)
        max_values = np.max(centered_data @ components.T, axis=0)
        min_values = np.min(centered_data @ components.T, axis=0)
        lengths = np.abs(max_values - min_values)

    return np.concatenate([components.flatten(), lengths])

def transform_features(features, apply_transform):
    features[["cp_x", "cp_y", "cp_z"]] = np.dot(features[["cp_x", "cp_y", "cp_z"]],
                                                apply_transform[:3, :3].T) + apply_transform[:3, 3]
    # change normals and pca
    features[["normal_x", "normal_y", "normal_z"]] = np.dot(features[["normal_x", "normal_y", "normal_z"]],
                                                            apply_transform[:3, :3].T)
    features[["pca1x", "pca1y", "pca1z"]] = np.dot(features[["pca1x", "pca1y", "pca1z"]], apply_transform[:3, :3].T)
    features[["pca2x", "pca2y", "pca2z"]] = np.dot(features[["pca2x", "pca2y", "pca2z"]], apply_transform[:3, :3].T)
    features[["pca3x", "pca3y", "pca3z"]] = np.dot(features[["pca3x", "pca3y", "pca3z"]], apply_transform[:3, :3].T)
    return features


def transform_geometric_graph(graph, apply_transform):
    # graph (networkx contains nodes attributes cp_x, cp_y, cp_z
    graph_t = graph.copy()
    for node in graph.nodes:
        node_data = graph.nodes[node]
        node_data["cp_x"], node_data["cp_y"], node_data["cp_z"] = np.dot([node_data["cp_x"], node_data["cp_y"], node_data["cp_z"]],
                                                                         apply_transform[:3, :3].T) + apply_transform[:3, 3]
        """# change normals and pca
        node_data["normal_x"], node_data["normal_y"], node_data["normal_z"] = np.dot([node_data["normal_x"], node_data["normal_y"], node_data["normal_z"]],
                                                                                     apply_transform[:3, :3].T)
        node_data["pca1x"], node_data["pca1y"], node_data["pca1z"] = np.dot([node_data["pca1x"], node_data["pca1y"], node_data["pca1z"]], apply_transform[:3, :3].T)
        node_data["pca2x"], node_data["pca2y"], node_data["pca2z"] = np.dot([node_data["pca2x"], node_data["pca2y"], node_data["pca2z"]], apply_transform[:3, :3].T)
        node_data["pca3x"], node_data["pca3y"], node_data["pca3z"] = np.dot([node_data["pca3x"], node_data["pca3y"], node_data["pca3z"]], apply_transform[:3, :3].T)"""


    return graph_t