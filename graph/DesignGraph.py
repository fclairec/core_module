from core_module.graph.myGraph import MyGraph
import pandas as pd
import numpy as np
import networkx as nx
from core_module.pem.IfcPEM import IfcPEM


class DesignGraph(MyGraph):
    """ Point cloud graph. """
    def __init__(self):
        super().__init__("d")
        self.type_g = "d"

    def assemble_graph_files(self, cfg, adjacency_type, selected_guids=[], feats=True):
        # check if all files are there. Adjacency, features, element_map
        features_file = cfg.features_file
        # TODO change for built
        if adjacency_type == "final":
            adjacency_file = cfg.final_adjacency_file
        elif adjacency_type == "element":
            adjacency_file = cfg.adjacency_file
        else:
            raise ValueError("invalid adjacency type - graph can not be assembled")


        if feats:
            features = pd.read_csv(features_file, sep=',', index_col="guid_int")
        else:
            features = None
        col_names = ["Start_node", "End_node", "distance", "Edge_type"]
        adjacency = pd.read_csv(adjacency_file, sep=',', header=None, names=col_names)

        if selected_guids != []:
            if features is not None:
                features = features[features.index.isin(selected_guids)]
            adjacency = adjacency[adjacency["Start_node"].isin(selected_guids) & adjacency["End_node"].isin(selected_guids)]

        self.graph.add_nodes_from(selected_guids)

        pem = IfcPEM()
        pem.load_pem(cfg.pem_file)
        if features is not None:
            node_attributes = self.get_features_from_files(features, selected_guids, pem)
            nx.set_node_attributes(self.graph, node_attributes)

        # add edges from adjacency file
        _left_nodes = adjacency["Start_node"].to_numpy().flatten()
        _right_node = adjacency["End_node"].to_numpy().flatten()
        edge_pairs = list(map(tuple, np.stack((_left_nodes, _right_node), axis=1)))
        self.graph.add_edges_from(edge_pairs)



    def get_features_from_files(self, features, guid_ints, pem: IfcPEM):
        node_attribute_dict = {}

        for guid_int in guid_ints:
            if guid_int in features.index:
                node_attribute_dict[guid_int] = features.loc[guid_int].to_dict()
            else:
                print(f"guid_int {guid_int} element without geometric features")
                # take template from features
                template = features.iloc[0].to_dict()
                # set all values to None
                for key in template.keys():
                    template[key] = None
                node_attribute_dict[guid_int] = template


            # add ifc guid to node attributes
            pem_entry = pem.get_instance_entry(guid_int)
            node_attribute_dict[guid_int]["ifc_guid"] = pem_entry["ifc_guid"]
            # add the node type
            node_attribute_dict[guid_int]["node_type"] = pem_entry["instance_type"]
            # add property "has_face"
            node_attribute_dict[guid_int]["has_face"] = 0 if np.isnan(pem_entry["associated_face"]) else 1


        return node_attribute_dict



    def delete_mre_nodes(self, cfg):
        # from pem get all guit_ints where type_txt is in ["Wall", "Ceiling", "Floor"] and instance_type is "element"
        pem = IfcPEM()
        pem.load_pem(cfg.pem_file)
        mre_guid_ints = pem.get_instance_guids_by_attribute_condition("spanning_element", 1)
        self.graph.remove_nodes_from(mre_guid_ints)

    def delete_space_nodes(self, cfg):
        # from pem get all guit_ints where type_txt is in ["Wall", "Ceiling", "Floor"] and instance_type is "element"
        pem = IfcPEM()
        pem.load_pem(cfg.pem_file)
        space_guid_ints = pem.get_instance_guids_by_type("space")
        self.graph.remove_nodes_from(space_guid_ints)


