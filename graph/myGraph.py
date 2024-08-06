import networkx as nx
import numpy as np
from collections import namedtuple
from plyfile import PlyData, PlyElement
from pathlib import Path
from core_module.default_config.config import sp_feature_translation_dict, int2color, enrichment_feature_dict, internali2internalt, internali2internalt_discipline
import copy
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from core_module.default_config.config import discipline_wise_classes
from core_module.utils_general.general_functions import invert_dict, invert_dict_simple
from core_module.pem.IfcPEM import IfcPEM
from core_module.pem.PcPEM import PcPEM


class MyGraph():
    """ Base class for graphs extending the networkx graph class. The BIM graph and point cloud graph are
    specialisations of this class"""

    def __init__(self, status=None):
        self.graph = nx.Graph()
        self.status = status

    def enrich_graph(self, pem_file, enrichment_feature_dict, node_color):
        """ enriches the graph with additional features"""
        if self.status == "d":
            pem = IfcPEM()
            pem.load_pem(pem_file)
        else:
            pem = PcPEM().load_pem(pem_file)
        graph_node_ids = list(self.graph.nodes)

        for enrichment_task, feature_name in enrichment_feature_dict.items():
            if enrichment_task in ['label', 'discipline']:
                features = pem.get_feature_vector(graph_node_ids, feature_name)
                nx.set_node_attributes(self.graph, dict(zip(graph_node_ids, features)), enrichment_task)
            elif enrichment_task == 'room':
                features = pem.get_feature_vector(graph_node_ids, feature_name)
                nx.set_node_attributes(self.graph, dict(zip(graph_node_ids, features)), enrichment_task)
            elif enrichment_task == 'color':
                features = pem.get_feature_vector(graph_node_ids, "type_int")
                colors = [int2color[feature] for feature in features]
                nx.set_node_attributes(self.graph, dict(zip(graph_node_ids, colors)), enrichment_task)
                # output a legend for the colors in a png file

                legend_patches = [mpatches.Patch(color=np.array(color)[:3] / 255, label=internali2internalt[label] + f"({label})") for label, color in int2color.items()]
                plt.figure(figsize=(5, 3))
                plt.legend(handles=legend_patches, loc='center')
                plt.axis('off')  # Turn off axes

                # Save the legend as a PNG file
                plt.savefig(node_color, bbox_inches='tight')
                plt.close()



        #self.verify_graph(node_attrbs)

    def add_custom_node(self, attributes):
        """function should add a node to the graph with the given adjacencies and features. It also adds the node to
        the project element map"""
        # get a new node id that is not already in the graph
        new_node_id = max(self.graph.nodes()) + 1
        self.graph.add_node(new_node_id, **attributes)
        # self.project_element_map = self.project_element_map.append(attributes, ignore_index=True)
        print(f"adding node")
        return new_node_id

    def add_custom_nodes(self, node_ids, attributes):
        """function should add a node to the graph with the given adjacencies and features. It also adds the node to
        the project element map"""
        # get a new node id that is not already in the graph
        graph_ids = []
        for node_id, attrs in zip(node_ids, attributes):
            if len(self.graph.nodes()) != 0:
                new_node_id = max(self.graph.nodes()) + 1
            else:
                new_node_id = 0
            graph_ids.append(new_node_id)
            self.graph.add_node(new_node_id, **attrs)
        # self.project_element_map = self.project_element_map.append(attributes, ignore_index=True)
        print(f"adding nodes")
        return graph_ids

    def add_custom_edges(self, edges):
        for edge in edges:
            self.graph.add_edge(*edge)






    def find_shared_adjacencies(self, node_id, node_id2):
        """ function should find the node by the id given, then check their adjacencies and return the shared ones"""
        adj_target = list(dict(self.graph.adj[node_id]).keys())
        adj_space = list(dict(self.graph.adj[node_id2]).keys())
        shared_adj = copy.deepcopy(list(set(adj_target).intersection(adj_space)))
        return shared_adj

    def verify_graph(self, node_attrbs):
        # check if all nodes have the same attributes
        # node_attrs = self.graph.nodes(data=True)[0].keys()
        ii = 0
        for i, (_, feat_dict) in enumerate(self.graph.nodes(data=True)):
            if set(feat_dict.keys()) != set(node_attrbs):
                # print node id and attributes
                print(f"node {i} has attributes {feat_dict.keys()}, but {ii} before did")
                raise ValueError('Not all nodes contain the same attributes')
            ii += 1

        print(f"Graph verified with features {node_attrbs}")

    def graph2viz(self, node_attrs2scalar: list, filename: Path) -> None:
        self.graph2ply(node_attrs2scalar, filename)
        filename_new = filename.with_suffix('.obj')
        self.graph2cc_format(filename_new)

    def graph2ply(self, node_attrs2scalar: list, filename: Path) -> None:
        """ Converts a graph to a ply file.
            :param graph: graph to convert
            :param node_attrs2scalar: list of node attributes to convert to scalar
            :param filename: output filename in Path object
            :return: None
            """
        Col = namedtuple('column', 'name type')
        vertex_prop = [Col('x', 'f4'), Col('y', 'f4'), Col('z', 'f4'), Col('red', 'u1'), Col('green', 'u1'), Col('blue', 'u1'),
                       Col('alpha', 'u1')]
        vertex_prop_scalar_extension = [Col(f's{i}', 'int32') for i in range(len(node_attrs2scalar))]
        vertex_prop = vertex_prop + vertex_prop_scalar_extension

        # fill centroids values
        vertex_val = np.empty(self.graph.number_of_nodes(), dtype=vertex_prop)
        for i in range(3):
            # loops over x_pos,y_pos,z_pos values and inserts them into vertex_val
            try:
                vertex_val[vertex_prop[i].name] = np.array(list(
                    nx.get_node_attributes(self.graph, sp_feature_translation_dict["centroid"]["myGraph"][i]).values()))
            except:
                raise ValueError(f"centroid {i} not available in networkx graph")
        # fill color values
        try:
            color = np.array(list(nx.get_node_attributes(self.graph, enrichment_feature_dict["color"]).values()))
            for i in range(0, 4):
                # loops over red, blue, green, alpha values and inserts them into vertex_val
                vertex_val[vertex_prop[i + 3].name] = color[:, i]
        except:
            # print warning if color is not available. fill color values with grey color and alpha 0
            print("color not available - filling with grey color")
            if i != 3:
                vertex_val[vertex_prop[i + 3].name] = np.array([128] * self.graph.number_of_nodes())
            else:
                vertex_val[vertex_prop[i + 3].name] = np.array([0] * self.graph.number_of_nodes())

        # fill scalar values
        for i in range(0, len(vertex_prop_scalar_extension)):
            # loops over scalar values and inserts them into vertex_val
            try:
                vertex_val[vertex_prop[i + 7].name] = np.array(
                    list(nx.get_node_attributes(self.graph, node_attrs2scalar[i]).values()))
            except:
                print(f"scalar {node_attrs2scalar[i]} not available in networkx graph")
        # for now this is limited to the label only. can be extended
        # vertex_val[vertex_prop[i].name] = np.array(list(nx.get_node_attributes(self.G, 'instance_id').values()))

        edges_prop = [('vertex1', 'int32'), ('vertex2', 'int32')]
        edges_val = np.empty(self.graph.number_of_edges(), dtype=edges_prop)
        node_positions = np.array(list(self.graph.nodes))
        source, target = [], []
        for edge in self.graph.edges():
            source.append(np.where(node_positions == edge[0]))
            target.append(np.where(node_positions == edge[1]))
        edges_val[edges_prop[0][0]] = np.array(source).flatten()
        edges_val[edges_prop[1][0]] = np.array(target).flatten()

        ply = PlyData([PlyElement.describe(vertex_val, 'vertex'), PlyElement.describe(edges_val, 'edge')], text=True)

        ply.write(filename)

    def graph2cc_format(self, filename) -> None:
        """ Converts a graph to a cc format. Needed for cloud compare visualisation.
        :param graph: graph to convert
        :param filename: output filename
        :return: None
        """

        Col = namedtuple('column', 'name type')
        vertex_prop_centroid = [Col('x', 'f4'), Col('y', 'f4'), Col('z', 'f4'), Col('node_index_cc', 'int32'),
                                Col('node_id', 'int32')]
        vertex_val_centroid = np.empty(self.graph.number_of_nodes(), dtype=vertex_prop_centroid)

        for i, attr_name in enumerate(sp_feature_translation_dict["centroid"]["myGraph"]):
            attr_dict = nx.get_node_attributes(self.graph, attr_name)
            node_val = list(attr_dict.values())
            vertex_val_centroid[vertex_prop_centroid[i].name] = np.array(node_val)

        node_id = list(attr_dict.keys())

        # add a column for the node index. The node index is just a number from 1 to n
        vertex_val_centroid[vertex_prop_centroid[3].name] = np.array(list(range(1, self.graph.number_of_nodes() + 1)))

        vertex_val_centroid[vertex_prop_centroid[4].name] = np.array(node_id)

        vertex_lines = []
        for i, (x, y, z, i_idx, _) in enumerate(vertex_val_centroid):
            vertex_lines.append("v {} {} {} {}".format(x, y, z, i_idx))

        # transform vertex_val_centroid into a dict where key is the node_id and value is the rest of the row
        combined = {}
        for _, (_, _, _, i_idx, node_id) in enumerate(vertex_val_centroid):
            combined[node_id] = i_idx

        line_lines = []
        for edge in self.graph.edges():
            line_lines.append("l {} {}".format(combined[edge[0]], combined[edge[1]]))

        # delete file if it exists
        if filename.exists():
            os.remove(filename)

        with open(filename, 'a') as the_file:
            for v in vertex_lines:
                the_file.write(v + '\n')
            for l in line_lines:
                the_file.write(l + '\n')

    def plot_graph(self, title=None, save_file=False):
        """
        plots graph in matplotlib for debugging (no real coordinates)
        :return:
        """
        import matplotlib.pyplot as plt

        nx.draw(self.graph, pos=nx.spring_layout(self.graph), with_labels=True)
        plt.draw()
        plt.title(title)

        if save_file:
            plt.savefig(save_file)


        """node_and_x = nx.get_node_attributes(self.graph, "cp_x")
        node_and_y = nx.get_node_attributes(self.graph, "cp_y")
        pos = dict()
        for node in node_and_x:
            x = node_and_x[node]
            y = node_and_y[node]
            pos[node] = (x, y)

        # nx.draw(self.G, pos=pos, node_color=node_color)
        nx.draw(self.graph, pos=pos)

        plt.draw()
        plt.title(title)
        plt.show()"""

    def graph_to_pkl(self, path=None):
        # TODO colors are a list value in attributes. graphml does not support this
        # for now just delete the color attributes
        # TODO change try to check if color is inside
        output_graph = copy.deepcopy(self.graph)
        for node in output_graph.nodes:
            # convert the color to a string with the format "r,g,b,a"
            color = ",".join([str(i) for i in output_graph.nodes[node]["color"]])
            output_graph.nodes[node]["color"] = color

        nx.write_graphml_lxml(output_graph, path)

    def load_graph_from_pickle(self, filename):
        """ loads graph from graph ML file"""
        self.graph = nx.read_graphml(f"{filename}")

    def delete_nodes_of_type(self, deletion_modes: list):
        nodes_to_delete = []
        for deletion_mode in deletion_modes:
            if deletion_mode == "element_with_child":
                for node in self.graph.nodes:
                    if self.graph.nodes[node]["node_type"] == "element":
                        if self.graph.nodes[node]["has_face"] is not None:
                            nodes_to_delete.append(node)
            elif deletion_mode == "delete_furniture":
                for node in self.graph.nodes:
                    if self.graph.nodes[node]["node_type"] == "furniture":
                        nodes_to_delete.append(node)
        self.graph.remove_nodes_from(nodes_to_delete)
        return len(nodes_to_delete)
