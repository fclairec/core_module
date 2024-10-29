from core_module.graph.myGraph import MyGraph
import pandas as pd
import numpy as np
import networkx as nx
import ast
from core_module.pem.IfcPEM import IfcPEM
import warnings


class DesignGraph(MyGraph):
    """ Point cloud graph. """
    def __init__(self):
        super().__init__("d")
        self.type_g = "d"

    def assemble_graph_files(self, cfg, adjacency_type, selected_guids=[], feats=True, rooms=False):
        # check if all files are there. Adjacency, features, element_map
        features_file = cfg.features_file
        # TODO change for built
        if adjacency_type == "final":
            adjacency_file = cfg.final_adjacency_file
            col_names = ["Start_node", "End_node", "distance", "Edge_type"]
            adjacency = pd.read_csv(adjacency_file, sep=',', header=None, names=col_names)
        elif adjacency_type == "element":
            # TODO unify the adjacency files to same format....
            adjacency_file = cfg.containment_file
            adjacency = pd.read_csv(adjacency_file, sep=',')
            # stlit column "pair" into two columns
            pairs = adjacency["pair"].str.replace("(", "").str.replace(")", "").str.split(", ", expand=True)
            # make ints
            pairs = pairs.astype(int)
            pairs.columns = ["Start_node", "End_node"]
            adjacency = pd.concat([adjacency, pairs], axis=1)

        else:
            raise ValueError("invalid adjacency type - graph can not be assembled")


        if feats:
            features = pd.read_csv(features_file, sep=',', index_col="guid_int")
        else:
            features = None

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

        # add element to room edges if spaces are included
        if rooms:
            room_edges = []
            for guid_int in selected_guids:
                pem_entry = pem.get_instance_entry(guid_int)
                if pem_entry["type_txt"] != "Space":
                    room_id = pem_entry["room_id"]
                    try:
                        room_id_list = ast.literal_eval(room_id)
                    except:
                        warnings.warn(f"room_id {room_id} could not be converted to list. Skipping.")
                        room_id_list = []

                    for id in room_id_list:
                        room_edges.append((guid_int, int(id)))
                else:
                    continue

            self.graph.add_edges_from(edge_pairs + room_edges)
        else:
            self.graph.add_edges_from(edge_pairs)



    def get_features_from_files(self, features, guid_ints, pem: IfcPEM):
        node_attribute_dict = {}

        for guid_int in guid_ints:

            # add ifc guid to node attributes
            pem_entry = pem.get_instance_entry(guid_int)


            if guid_int in features.index:

                if pem_entry["type_txt"] == "Space":
                    feats = features.loc[guid_int].to_dict()
                    feats["cp_z"] += 6 # room node shift to be above the element
                    node_attribute_dict[guid_int] = feats
                else:
                    node_attribute_dict[guid_int] = features.loc[guid_int].to_dict()

                a=0
            else:
                print(f"guid_int {guid_int} element without geometric features")
                # take template from features
                template = features.iloc[0].to_dict()
                # set all values to None
                for key in template.keys():
                    template[key] = None
                node_attribute_dict[guid_int] = template

            node_attribute_dict[guid_int]["ifc_guid"] = pem_entry["ifc_guid"]
            # add the node type
            node_attribute_dict[guid_int]["node_type"] = pem_entry["instance_type"]

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


