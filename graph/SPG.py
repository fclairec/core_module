import numpy as np
from scipy.spatial import Delaunay
from numpy import linalg as LA
import os
import h5py


class SPG:
    def __init__(self):
        self.label_hist = None
        self.has_labels = None
        self.n_edg = None
        self.n_labels = None
        self.n_com = None
        self.n_sedg = None
        self.a = 0
        self.graph_sp = dict()
        self._components = None
        self._in_component = None

    def init_spg(self):
        self.graph_sp["sp_centroids"] = np.zeros((self.n_com, 3), dtype='float32')
        self.graph_sp["sp_length"] = np.zeros((self.n_com, 1), dtype='float32')
        self.graph_sp["sp_surface"] = np.zeros((self.n_com, 1), dtype='float32')
        self.graph_sp["sp_volume"] = np.zeros((self.n_com, 1), dtype='float32')
        self.graph_sp["sp_point_count"] = np.zeros((self.n_com, 1), dtype='uint64')
        self.graph_sp["pca1"] = np.zeros((self.n_com, 3), dtype='float32')
        self.graph_sp["pca2"] = np.zeros((self.n_com, 3), dtype='float32')
        self.graph_sp["pca3"] = np.zeros((self.n_com, 3), dtype='float32')
        self.graph_sp["sp_extents"] = np.zeros((self.n_com, 3), dtype='float32')
        self.graph_sp["sp_labels"] = np.zeros((self.n_com, self.n_labels + 1), dtype='uint32')

        self.graph_sp["source"] = np.zeros((self.n_sedg, 1), dtype='uint32')
        self.graph_sp["target"] = np.zeros((self.n_sedg, 1), dtype='uint32')
        self.graph_sp["se_delta_mean"] = np.zeros((self.n_sedg, 3), dtype='float32')
        self.graph_sp["se_delta_std"] = np.zeros((self.n_sedg, 3), dtype='float32')
        self.graph_sp["se_delta_norm"] = np.zeros((self.n_sedg, 1), dtype='float32')
        self.graph_sp["se_delta_centroid"] = np.zeros((self.n_sedg, 3), dtype='float32')
        self.graph_sp["se_length_ratio"] = np.zeros((self.n_sedg, 1), dtype='float32')
        self.graph_sp["se_surface_ratio"] = np.zeros((self.n_sedg, 1), dtype='float32')
        self.graph_sp["se_volume_ratio"] = np.zeros((self.n_sedg, 1), dtype='float32')
        self.graph_sp["se_point_count_ratio"] = np.zeros((self.n_sedg, 1), dtype='float32')


    def assemble_spg(self, formatted_pcd, d_max):
        """ method assembles super point graph, the main data structure for the SPG method (Landrieu et al 2017)
        :param formatted_pcd: [xyz, labels, components, in_component] (mind spg method specific format)
        :param d_max: maximum distance between points to be considered as edges"""
        xyz, labels, self.components, self.in_component = formatted_pcd[0], formatted_pcd[1], formatted_pcd[2], formatted_pcd[3]
        self.n_com = max(self.in_component) + 1
        self.has_labels = len(labels) > 1
        self.label_hist = self.has_labels and len(labels.shape) > 1 and labels.shape[1] > 1
        self.n_labels = len(np.unique(labels))

        tri = Delaunay(xyz)

        interface = self.in_component[tri.simplices[:, 0]] != self.in_component[tri.simplices[:, 1]]

        edg1 = np.vstack((tri.simplices[interface, 0], tri.simplices[interface, 1]))
        edg1r = np.vstack((tri.simplices[interface, 1], tri.simplices[interface, 0]))
        interface = self.in_component[tri.simplices[:, 0]] != self.in_component[tri.simplices[:, 2]]
        edg2 = np.vstack((tri.simplices[interface, 0], tri.simplices[interface, 2]))
        edg2r = np.vstack((tri.simplices[interface, 2], tri.simplices[interface, 0]))
        interface = self.in_component[tri.simplices[:, 0]] != self.in_component[tri.simplices[:, 3]]
        edg3 = np.vstack((tri.simplices[interface, 0], tri.simplices[interface, 3]))
        edg3r = np.vstack((tri.simplices[interface, 3], tri.simplices[interface, 0]))
        interface = self.in_component[tri.simplices[:, 1]] != self.in_component[tri.simplices[:, 2]]
        edg4 = np.vstack((tri.simplices[interface, 1], tri.simplices[interface, 2]))
        edg4r = np.vstack((tri.simplices[interface, 2], tri.simplices[interface, 1]))
        interface = self.in_component[tri.simplices[:, 1]] != self.in_component[tri.simplices[:, 3]]
        edg5 = np.vstack((tri.simplices[interface, 1], tri.simplices[interface, 3]))
        edg5r = np.vstack((tri.simplices[interface, 3], tri.simplices[interface, 1]))
        interface = self.in_component[tri.simplices[:, 2]] != self.in_component[tri.simplices[:, 3]]
        edg6 = np.vstack((tri.simplices[interface, 2], tri.simplices[interface, 3]))
        edg6r = np.vstack((tri.simplices[interface, 3], tri.simplices[interface, 2]))
        del tri, interface
        edges = np.hstack((edg1, edg2, edg3, edg4, edg5, edg6, edg1r, edg2r,
                           edg3r, edg4r, edg5r, edg6r))
        del edg1, edg2, edg3, edg4, edg5, edg6, edg1r, edg2r, edg3r, edg4r, edg5r, edg6r
        edges = np.unique(edges, axis=1)

        if d_max > 0:
            dist = np.sqrt(((xyz[edges[0, :]] - xyz[edges[1, :]]) ** 2).sum(1))
            edges = edges[:, dist < d_max]

        # ---sort edges by alphanumeric order wrt to the components of their source/target---
        self.n_edg = len(edges[0])
        edge_comp = self.in_component[edges]  # what happens here?
        edge_comp_index = self.n_com * edge_comp[0, :] + edge_comp[1, :]
        order = np.argsort(edge_comp_index)
        edges = edges[:, order]
        edge_comp = edge_comp[:, order]
        edge_comp_index = edge_comp_index[order]
        # marks where the edges change components iot compting them by blocks
        jump_edg = np.vstack((0, np.argwhere(np.diff(edge_comp_index)) + 1, self.n_edg)).flatten()
        self.n_sedg = len(jump_edg) - 1

        self.init_spg()
        self.compute_sp_features(xyz, labels)
        self.compute_se_features(xyz, edges, jump_edg, edge_comp)


    def compute_sp_features(self, xyz, labels):
        """ method computes super point features and adds to graph
        :param xyz: point cloud
        :param components: components of the point cloud (mind method specific format)
        :param labels: labels of the point cloud
        """
        for i_com in range(0, self.n_com):
            comp = self.components[i_com]

            if self.has_labels and not self.label_hist:
                self.graph_sp["sp_labels"][i_com, :] = \
                np.histogram(labels[comp], bins=[float(i) - 0.5 for i in range(0, self.n_labels + 2)])[0]
            if self.has_labels and self.label_hist:
                self.graph_sp["sp_labels"][i_com, :] = sum(labels[comp, :])

            self.graph_sp["sp_point_count"][i_com] = len(comp)
            xyz_sp = np.unique(xyz[comp, :], axis=0)

            if len(xyz_sp) == 1:
                self.graph_sp["sp_centroids"][i_com] = xyz_sp
                self.graph_sp["sp_length"][i_com] = 0
                self.graph_sp["sp_surface"][i_com] = 0
                self.graph_sp["sp_volume"][i_com] = 0

            elif len(xyz_sp) == 2:
                self.graph_sp["sp_centroids"][i_com] = np.mean(xyz_sp, axis=0)
                self.graph_sp["sp_length"][i_com] = np.sqrt(np.sum(np.var(xyz_sp, axis=0)))
                self.graph_sp["sp_surface"][i_com] = 0
                self.graph_sp["sp_volume"][i_com] = 0
            else:
                ev = LA.eig(np.cov(np.transpose(xyz_sp), rowvar=True))
                ev = -np.sort(-ev[0])  # descending order
                self.graph_sp["sp_centroids"][i_com] = np.mean(xyz_sp, axis=0)
                try:
                    self.graph_sp["sp_length"][i_com] = ev[0]
                except TypeError:
                    self.graph_sp["sp_length"][i_com] = 0
                try:
                    self.graph_sp["sp_surface"][i_com] = np.sqrt(ev[0] * ev[1] + 1e-10)
                except TypeError:
                    self.graph_sp["sp_surface"][i_com] = 0
                try:
                    self.graph_sp["sp_volume"][i_com] = np.sqrt(ev[0] * ev[1] * ev[2] + 1e-10)
                except TypeError:
                    self.graph_sp["sp_volume"][i_com] = 0

                from sklearn.decomposition import PCA

                pca = PCA(n_components=3)
                pca.fit(xyz_sp)
                pca = pca.components_
                self.graph_sp["pca1"][i_com] = pca[0]
                self.graph_sp["pca2"][i_com] = pca[1]
                self.graph_sp["pca3"][i_com] = pca[2]

                centered_data = xyz_sp - np.mean(xyz_sp, axis=0)
                max_values = np.max(centered_data @ pca.T, axis=0)
                min_values = np.min(centered_data @ pca.T, axis=0)
                lengths = np.abs(max_values - min_values)
                self.graph_sp["sp_extents"][i_com] = lengths

    def compute_se_features(self, xyz, edges, jump_edg, edge_comp):
        """ method comuptes super edge features and adds to graph
        :param xyz: point cloud
        :param edges: edges of the spg
        :param jump_edg: (method specific)
        :param edge_comp: (method specific)
        """
        for i_sedg in range(0, self.n_sedg):
            i_edg_begin = jump_edg[i_sedg]
            i_edg_end = jump_edg[i_sedg + 1]
            ver_source = edges[0, range(i_edg_begin, i_edg_end)]
            ver_target = edges[1, range(i_edg_begin, i_edg_end)]
            com_source = edge_comp[0, i_edg_begin]
            com_target = edge_comp[1, i_edg_begin]
            xyz_source = xyz[ver_source, :]
            xyz_target = xyz[ver_target, :]
            self.graph_sp["source"][i_sedg] = com_source
            self.graph_sp["target"][i_sedg] = com_target
            # ---compute the ratio features---
            self.graph_sp["se_delta_centroid"][i_sedg, :] = self.graph_sp["sp_centroids"][com_source, :] - self.graph_sp["sp_centroids"][
                                                                                           com_target, :]

            self.graph_sp["se_length_ratio"][i_sedg] = self.graph_sp["sp_length"][com_source] / (
                    self.graph_sp["sp_length"][com_target] + 1e-6)
            self.graph_sp["se_surface_ratio"][i_sedg] = self.graph_sp["sp_surface"][com_source] / (
                    self.graph_sp["sp_surface"][com_target] + 1e-6)
            self.graph_sp["se_volume_ratio"][i_sedg] = self.graph_sp["sp_volume"][com_source] / (
                    self.graph_sp["sp_volume"][com_target] + 1e-6)
            self.graph_sp["se_point_count_ratio"][i_sedg] = self.graph_sp["sp_point_count"][com_source] / (
                    self.graph_sp["sp_point_count"][com_target] + 1e-6)
            # ---compute the offset set---
            delta = xyz_source - xyz_target
            if len(delta) > 1:
                self.graph_sp["se_delta_mean"][i_sedg] = np.mean(delta, axis=0)
                self.graph_sp["se_delta_std"][i_sedg] = np.std(delta, axis=0)
                self.graph_sp["se_delta_norm"][i_sedg] = np.mean(np.sqrt(np.sum(delta ** 2, axis=1)))
            else:
                self.graph_sp["se_delta_mean"][i_sedg, :] = delta
                self.graph_sp["se_delta_std"][i_sedg, :] = [0, 0, 0]
                self.graph_sp["se_delta_norm"][i_sedg] = np.sqrt(np.sum(delta ** 2))

    def write_spg(self, file_name):
        if os.path.isfile(file_name):
            os.remove(file_name)
        data_file = h5py.File(file_name, 'w')
        grp = data_file.create_group('components')
        for i_com in range(0, self.n_com):
            grp.create_dataset(str(i_com), data=self.components[i_com], dtype='uint32')
        data_file.create_dataset('in_component'
                                 , data=self.in_component, dtype='uint32')
        data_file.create_dataset('sp_labels'
                                 , data=self.graph_sp["sp_labels"], dtype='uint32')
        data_file.create_dataset('sp_centroids'
                                 , data=self.graph_sp["sp_centroids"], dtype='float32')
        data_file.create_dataset('sp_length'
                                 , data=self.graph_sp["sp_length"], dtype='float32')
        data_file.create_dataset('sp_surface'
                                 , data=self.graph_sp["sp_surface"], dtype='float32')
        data_file.create_dataset('sp_volume'
                                 , data=self.graph_sp["sp_volume"], dtype='float32')
        data_file.create_dataset('sp_point_count'
                                 , data=self.graph_sp["sp_point_count"], dtype='uint64')
        data_file.create_dataset('source'
                                 , data=self.graph_sp["source"], dtype='uint32')
        data_file.create_dataset('target'
                                 , data=self.graph_sp["target"], dtype='uint32')
        data_file.create_dataset('se_delta_mean'
                                 , data=self.graph_sp["se_delta_mean"], dtype='float32')
        data_file.create_dataset('se_delta_std'
                                 , data=self.graph_sp["se_delta_std"], dtype='float32')
        data_file.create_dataset('se_delta_norm'
                                 , data=self.graph_sp["se_delta_norm"], dtype='float32')
        data_file.create_dataset('se_delta_centroid'
                                 , data=self.graph_sp["se_delta_centroid"], dtype='float32')
        data_file.create_dataset('se_length_ratio'
                                 , data=self.graph_sp["se_length_ratio"], dtype='float32')
        data_file.create_dataset('se_surface_ratio'
                                 , data=self.graph_sp["se_surface_ratio"], dtype='float32')
        data_file.create_dataset('se_volume_ratio'
                                 , data=self.graph_sp["se_volume_ratio"], dtype='float32')
        data_file.create_dataset('se_point_count_ratio'
                                 , data=self.graph_sp["se_point_count_ratio"], dtype='float32')
        data_file.create_dataset('pca1'
                                 , data=self.graph_sp["pca1"], dtype='float32')
        data_file.create_dataset('pca2'
                                 , data=self.graph_sp["pca2"], dtype='float32')
        data_file.create_dataset('pca3'
                                 , data=self.graph_sp["pca3"], dtype='float32')
        data_file.create_dataset('sp_extents'
                                 , data=self.graph_sp["sp_extents"], dtype='float32')

        print("SPG file written...")

    def load_spg(self, file_name):
        if not os.path.isfile(file_name):
            print("SPG file not found")
            return
        else:
            print("loading SPG file...")
            data_file = h5py.File(file_name, 'r')
            self.graph_sp = dict([("is_nn", False)])
            self.graph_sp["source"] = np.array(data_file["source"], dtype='uint32')
            self.graph_sp["target"] = np.array(data_file["target"], dtype='uint32')
            self.graph_sp["sp_centroids"] = np.array(data_file["sp_centroids"], dtype='float32')
            self.graph_sp["sp_length"] = np.array(data_file["sp_length"], dtype='float32')
            self.graph_sp["sp_surface"] = np.array(data_file["sp_surface"], dtype='float32')
            self.graph_sp["sp_volume"] = np.array(data_file["sp_volume"], dtype='float32')
            self.graph_sp["sp_point_count"] = np.array(data_file["sp_point_count"], dtype='uint64')
            self.graph_sp["se_delta_mean"] = np.array(data_file["se_delta_mean"], dtype='float32')
            self.graph_sp["se_delta_std"] = np.array(data_file["se_delta_std"], dtype='float32')
            self.graph_sp["se_delta_norm"] = np.array(data_file["se_delta_norm"], dtype='float32')
            self.graph_sp["se_delta_centroid"] = np.array(data_file["se_delta_centroid"], dtype='float32')
            self.graph_sp["se_length_ratio"] = np.array(data_file["se_length_ratio"], dtype='float32')
            self.graph_sp["se_surface_ratio"] = np.array(data_file["se_surface_ratio"], dtype='float32')
            self.graph_sp["se_volume_ratio"] = np.array(data_file["se_volume_ratio"], dtype='float32')
            self.graph_sp["se_point_count_ratio"] = np.array(data_file["se_point_count_ratio"], dtype='float32')
            self.graph_sp["sp_extents"] = np.array(data_file["sp_extents"], dtype='float32')
            self.graph_sp["pca1"] = np.array(data_file["pca1"], dtype='float32')
            self.graph_sp["pca2"] = np.array(data_file["pca2"], dtype='float32')
            self.graph_sp["pca3"] = np.array(data_file["pca3"], dtype='float32')
            self._in_component = np.array(data_file["in_component"], dtype='uint32')
            self.n_com = len(self.graph_sp["sp_length"])
            self.graph_sp["sp_labels"] = np.array(data_file["sp_labels"], dtype='uint32')
            grp = data_file['components']
            self._components = np.empty((self.n_com,), dtype=object)
            for i_com in range(0, self.n_com):
                self._components[i_com] = np.array(grp[str(i_com)], dtype='uint32').tolist()

