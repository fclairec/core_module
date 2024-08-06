

from core_module.graph.myGraph import MyGraph


class LayeredGraph(MyGraph):
    """ Point cloud graph.
    returns subgraph of elements by room """
    def __init__(self, graph_dict, bipartite_edges: list=[]):
        super().__init__()
        """graph_dict: {"room": graph, "ARC": graph, "PLB": graph, "VTL": graph, "EL": graph, "FUR":graph}"""

        for name, graph in graph_dict.items():
            if graph is None:
                print("LayeredGraph: {} is None".format(name))
                continue
            # if the graph aleady has the "layer" attribute, do not add it again
            if all('layer' in data for _, data in graph.nodes(data=True)):
                layer = graph
            else:
                layer = self._add_layer_attribute(graph, name)
            self.graph.add_nodes_from(layer.nodes(data=True))
            self.graph.add_edges_from(layer.edges(data=True))

        # Add bipartite edges (edges between room and elements)
        if len(bipartite_edges) > 0:
            self.graph.add_edges_from(bipartite_edges)

    def _add_layer_attribute(self, graph, layer_name):
        """ Add layer attribute to all nodes in a graph. """
        for node in graph.nodes:
            graph.nodes[node]['layer'] = layer_name

        # check if all nodes in the graph have the "layer" attribute key
        assert all([graph.nodes[node].get('layer') == layer_name for node in graph.nodes])
        return graph


    def getroomwise_subgraph(self, room_id):
        """ Return a subgraph of elements associated with a specific room. """
        neighbors = list(self.graph.neighbors(room_id))
        element_nodes = [n for n in neighbors if self.graph.nodes[n]['layer'] != 'room']
        return self.graph.subgraph(element_nodes + [room_id])

    def get_discipline_wise_subgraph(self, discipline, room_id=None):
        """ Return a subgraph of elements associated with a specific room. """
        if room_id is None:
            element_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n].get('layer') == discipline]
            return self.graph.subgraph(element_nodes)
        else:
            room_sub_graph = self.getroomwise_subgraph(room_id)
            element_nodes = [n for n in room_sub_graph.nodes() if room_sub_graph.nodes[n].get('layer') == discipline]
            return room_sub_graph.subgraph(element_nodes)


    def get_nodes_with_attributes(self, attrs):
        nodes = []
        for n, attr in self.graph.nodes(data=True):
            if all(attr.get(k) == v for k, v in attrs.items()):
                nodes.append(n)
        return nodes

    def what_room_am_i_in(self, node):
        """ Return the room node that a given element node belongs to. """
        neighbors = list(self.graph.neighbors(node))
        room_nodes = [n for n in neighbors if self.graph.nodes[n]['layer'] == 'room']
        return room_nodes
