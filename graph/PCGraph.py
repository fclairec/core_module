import networkx as nx
import numpy as np
import pandas as pd
from core_module.graph.myGraph import MyGraph
from core_module.default_config.config import sp_feature_translation_dict


def retrieve_initial_label(spg_label, map_filename):
    """ the sp labels were indexed new to guarantee a continuous index. This function retrieves the original
    label by loading it from the project element map."""
    # load project element map
    project_element_map = pd.read_csv(map_filename, sep=',', header=0, index_col=0)
    instance_guid_ints = []
    for i in spg_label:
        instance_guid_int = project_element_map.loc[project_element_map['spg_label'] == i, 'instance_guid_int'].values[0]
        instance_guid_ints.append(instance_guid_int)

    return instance_guid_ints


class PCGraph(MyGraph):
    """ Point cloud graph. """
    def __init__(self):
        super().__init__("b")
        self.type_g = "b"

    def assemble_from_spg(self, spg, split_features_file):
        """ assemble graph from spg. geometric fetures calculated from the points are inserted into the node attributes.
         :param spg: SPG object
         :param project_element_map_filename: path to project element map for looking up initial instance labels"""

        # node ids
        guid_int = np.argmax(spg.graph_sp['sp_labels'], axis=1)
        #initial_labels = retrieve_initial_label(spg_ids, project_element_map_filename)
        node_ids = guid_int

        # node features (as a possible subset of all features from SPG, can be extended if needed)
        # TODO extend this with whole from config
        node_features= pd.DataFrame()
        for task, feature_translation in sp_feature_translation_dict.items():
            spg_feat_names = feature_translation["SPG"]
            my_names = feature_translation["myGraph"]
            node_features_task = pd.DataFrame()

            for i, spg_feat_name in enumerate(spg_feat_names):
                factor = len(my_names) // len(spg_feat_names)
                if factor == 3:
                    for j in range(3):
                        node_features_task[my_names[i * factor + j]] = [point[j] for point in spg.graph_sp[spg_feat_name]]
                elif factor == 1:
                    node_features_task[my_names[i]] = spg.graph_sp[spg_feat_name]

            node_features = pd.concat([node_features, node_features_task], axis=1)

        node_features.reset_index(inplace=True)
        node_features.rename(columns={'index': 'guid_int'}, inplace=True)
        node_features.to_csv(split_features_file, index=True, header=True)

        node_attributes = {}
        for spg_id in guid_int:
            node_attributes[spg_id] = node_features.loc[node_features['guid_int'] == spg_id].to_dict('records')[0]

        # edges
        spid_2_nodeid = dict(zip(guid_int, node_ids))
        _edge_pairs_sp_ids = list(zip(spg.graph_sp['source'].flatten(), spg.graph_sp['target'].flatten()))
        edge_pairs = [(spid_2_nodeid[i], spid_2_nodeid[j]) for i, j in _edge_pairs_sp_ids]

        # edge features (as a possible subset of all features from SPG, can be extended if needed)
        edge_features = np.ones(len(node_ids))

        self.graph.add_nodes_from(node_ids)
        self.graph.add_edges_from(edge_pairs)
        nx.set_node_attributes(self.graph, node_attributes)




