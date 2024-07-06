

from core_module.graph.myGraph import MyGraph


class LayeredGraph(MyGraph):
    """ Point cloud graph.
    returns subgraph of elements by room """
    def __init__(self, room_graph, element_graph, bipartite_edges: list):
        super().__init__()


        # Add room and element subgraphs with "layer" attribute
        self._add_layer_attribute(room_graph, 'room')
        self._add_layer_attribute(element_graph, 'element')

        # Add room and element subgraphs
        self.graph.add_nodes_from(room_graph.nodes(data=True))
        self.graph.add_nodes_from(element_graph.nodes(data=True))
        self.graph.add_edges_from(room_graph.edges(data=True))
        self.graph.add_edges_from(element_graph.edges(data=True))

        # Add bipartite edges (edges between room and elements)
        self.graph.add_edges_from(bipartite_edges)

    def _add_layer_attribute(self, graph, layer_name):
        """ Add layer attribute to all nodes in a graph. """
        for node in graph.nodes:
            graph.nodes[node]['layer'] = layer_name

    def get_elements_subgraph(self, room_id):
        """ Return a subgraph of elements associated with a specific room. """
        neighbors = list(self.graph.neighbors(room_id))
        element_nodes = [n for n in neighbors if self.graph.nodes[n]['layer'] == 'building']
        return self.graph.subgraph(element_nodes + [room_id])