from core_module.graph.myGraph import MyGraph
import pandas as pd
import numpy as np
import networkx as nx
from core_module.pem.io import load_pem


class DesignGraph(MyGraph):
    """ Point cloud graph. """
    def __init__(self):
        super().__init__("d")
        self.type_g = "d"

    def assemble_graph_files(self, cfg, adjacency_type, faces=False, by_guid_int=[], feats=True):
        # check if all files are there. Adjacency, features, element_map
        pem_file = cfg.pem_file
        features_file = cfg.features_file
        # TODO change for built
        if adjacency_type == "final":
            adjacency_file = cfg.final_adjacency_file
        elif adjacency_type == "element":
            adjacency_file = cfg.adjacency_file
        else:
            raise ValueError("invalid adjacency type - graph can not be assembled")

        pem = load_pem(pem_file, mode="design")
        if feats:
            features = pd.read_csv(features_file, sep=',', index_col="guid_int")
        else:
            features = None
        col_names = ["Start_node", "End_node", "distance", "Edge_type"]
        adjacency = pd.read_csv(adjacency_file, sep=',', header=None, names=col_names)

        if by_guid_int != []:
            pem = pem[pem.index.isin(by_guid_int)]
            if features is not None:
                features = features[features.index.isin(by_guid_int)]
            adjacency = adjacency[adjacency["Start_node"].isin(by_guid_int) & adjacency["End_node"].isin(by_guid_int)]

        guid_ints = pem.index.to_numpy()
        self.graph.add_nodes_from(guid_ints)

        if features is not None:
            node_attributes = self.get_features_from_files(faces, features, guid_ints, pem)
            nx.set_node_attributes(self.graph, node_attributes)

        # add edges from adjacency file
        _left_nodes = adjacency["Start_node"].to_numpy().flatten()
        _right_node = adjacency["End_node"].to_numpy().flatten()
        edge_pairs = list(map(tuple, np.stack((_left_nodes, _right_node), axis=1)))
        self.graph.add_edges_from(edge_pairs)



    def get_features_from_files(self, fetch_faces, features, guid_ints, project_element_map):
        node_attribute_dict = {}
        element_vs_face = project_element_map.copy().groupby("instance_type")
        try:
            # sometimes there are no faces
            if fetch_faces:
                faces = element_vs_face.get_group("face")
        except:
            fetch_faces = False

        for guid_int in guid_ints:
            if guid_int not in features.index:
                print(f"guid_int {guid_int} element without geometric features")
                # set attributes to None
                # take template from features
                template = features.iloc[0].to_dict()
                # set all values to None
                for key in template.keys():
                    template[key] = None
                node_attribute_dict[guid_int] = template
            else:
                # add all the geometric features from the file
                node_attribute_dict[guid_int] = features.loc[guid_int].to_dict()
            # add ifc guid to node attributes
            node_attribute_dict[guid_int]["ifc_guid"] = project_element_map.loc[guid_int]["ifc_guid"]
            # add the node type
            node_attribute_dict[guid_int]["node_type"] = project_element_map.loc[guid_int]["instance_type"]
            # add property "has_face"
            if project_element_map.loc[guid_int]["instance_type"] == "element":
                if fetch_faces:
                    if project_element_map.loc[guid_int]["ifc_guid"] in faces["ifc_guid"].to_numpy():
                        node_attribute_dict[guid_int]["has_face"] = 1
                else:
                    node_attribute_dict[guid_int]["has_face"] = 0
            else:
                node_attribute_dict[guid_int]["has_face"] = 0


        return node_attribute_dict



    def delete_mre_nodes(self, cfg):
        # from pem get all guit_ints where type_txt is in ["Wall", "Ceiling", "Floor"] and instance_type is "element"
        pem = load_pem(cfg.pem_file, mode="design")

        mre_guid_ints = pem[pem["spanning_element"] == 1].index.to_numpy()

        self.graph.remove_nodes_from(mre_guid_ints)

    def delete_space_nodes(self, cfg):
        # from pem get all guit_ints where type_txt is in ["Wall", "Ceiling", "Floor"] and instance_type is "element"
        pem = load_pem(cfg.pem_file, mode="design")

        space_guid_ints = pem.loc[(pem["type_txt"].isin(["Space"]))].index.to_numpy()

        self.graph.remove_nodes_from(space_guid_ints)


