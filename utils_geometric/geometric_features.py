import numpy as np


def get_geometric_features(cfg, pcd):
    feature_dict = {}

    pcd += np.random.normal(0, 0.01, pcd.shape)

    for task in cfg:
        method = globals().get(task)
        if method is not None and callable(method):
            features = method(pcd)
            if task == 'pca_and_extents':
                # the last three values are the extents
                feature_dict["extents"] = features[-3:]
                feature_dict["pca"] = features[:-3]
            else:
                feature_dict[task] = features
        else:
            print(f"Method does not exist - {task} skipped")

    return feature_dict


def centroid(pcd=None):
    center_point = pcd.mean(axis=0)
    return center_point


def pca_and_extents(pcd=None):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=3)
    pca.fit(pcd)

    components = pca.components_

    centered_data = pcd - np.mean(pcd, axis=0)
    max_values = np.max(centered_data @ components.T, axis=0)
    min_values = np.min(centered_data @ components.T, axis=0)
    lengths = np.abs(max_values - min_values)
    return np.concatenate([components.flatten(), lengths])


def calculate_extent_along_axis(self, axis):
    self.extent_min = float(np.array(self.shape.geometry.verts).reshape(-1, 3)[:, axis].min())
    self.extent_max = float(np.array(self.shape.geometry.verts).reshape(-1, 3)[:, axis].max())
    return self.extent_min, self.extent_max


"""
    def compute_sdf(self):
        a = 0
        igl.shape_diameter_function(np.array(self.shape.geometry.verts).reshape((-1, 3)),
                                    np.array(self.shape.geometry.faces).reshape((-1, 3)),
                                    np.array(self.shape.geometry.verts).reshape((-1, 3)),
                                    np.array(self.shape.geometry.normals).reshape((-1, 3)), 1)"""
