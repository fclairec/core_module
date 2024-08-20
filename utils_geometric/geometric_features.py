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

        extent_xyz = [2*bbox.XHSize(), 2*bbox.YHSize(), 2*bbox.ZHSize()]

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

