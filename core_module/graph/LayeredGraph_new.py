
from core_module.graph.myGraph import MyGraph


class LayeredGraph_new(MyGraph):
    """ this graph is a special version of my graph. it optionally contains room nodes, and all nodes have a discipline
    attribute. methods are: get roomswise subgraph, get discipline wise subgraph,  what_room_am_i_in, a streching
    function, that moves the centroid of certain elements up or downwards.
    input a regular my graph object including room nodes. a mapping table for element label to discipline. """

    def __init__(self, flat_graph: MyGraph):
        super().__init__()
        self.graph = flat_graph

