
from core_module.default.match_config import internali2internalt_discipline
from core_module.graph.myGraph import MyGraph
from core_module.utils_general.general_functions import invert_dict_simple


class LayeredGraph(MyGraph):
    """ Point cloud graph.
    returns subgraph of elements by room """
    def __init__(self, graph=None):
        super().__init__()


    def getroomwise_subgraph(self, room_id):
        """ Return a subgraph of elements associated with a specific room. """
        neighbors = list(self.graph.neighbors(room_id))
        element_nodes = [n for n in neighbors if self.graph.nodes[n]['node_type'] != 'space']
        return self.graph.subgraph(element_nodes)

    def get_graph_without_spaces(self):
        """ Return a subgraph of elements only. """
        element_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]['node_type'] != 'space']
        return self.graph.subgraph(element_nodes)

    def get_discipline_wise_subgraph(self, discipline, room_id=None):
        """ Return a subgraph of elements associated with a specific discipline and room. """
        discipline_i = invert_dict_simple(internali2internalt_discipline)[discipline]
        if room_id is None:
            element_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n].get('discipline') == discipline_i]
            return self.graph.subgraph(element_nodes)
        else:
            room_sub_graph = self.getroomwise_subgraph(room_id)
            element_nodes = [n for n in room_sub_graph.nodes() if room_sub_graph.nodes[n].get('discipline') == discipline_i]
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
        room_nodes = [n for n in neighbors if self.graph.nodes[n]['node_type'] == 'space']
        return room_nodes
