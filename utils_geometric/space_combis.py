from core_module.graph.PCGraph import PCGraph
from itertools import combinations
import networkx as nx
import itertools
from core_module.default_config.config import transition_element_types, internali2internalt
from core_module.utils_general.general_functions import invert_dict_simple


def get_possible_room_combinations(cfg, nb=2, limit_room_repetition=True):
    # load graph and get adjacent rooms
    PCGraphObject = PCGraph()
    PCGraphObject.load_graph_from_pickle(cfg.space_graph)

    g = PCGraphObject.graph

    room_labels = invert_dict_simple(internali2internalt)["Space"]
    transition_labels = [invert_dict_simple(internali2internalt)[i_type] for i_type in transition_element_types]

    transition_nodes = [node for node, attrs in g.nodes(data=True) if attrs.get('label') in transition_labels]
    for door in transition_nodes:
        # Find all rooms (space nodes) connected to this door
        connected_rooms = [neighbor for neighbor in g.neighbors(door) if g.nodes[neighbor].get('label') == room_labels]
        # Connect all pairs of connected rooms directly
        for i in range(len(connected_rooms)):
            for j in range(i + 1, len(connected_rooms)):
                g.add_edge(connected_rooms[i], connected_rooms[j])
        # Remove the door node from the graph
        g.remove_node(door)

    room_nodes = [node for node, attrs in g.nodes(data=True) if attrs.get('label') == room_labels]

    room_combinations = list(combinations(room_nodes, nb))

    valid_combinations = []
    for room_combination in room_combinations:
        room_combination = room_combination + (room_combination[0],)
        sub_graph = nx.Graph()
        sub_graph.add_nodes_from(room_combination)
        possible_edges = list(itertools.combinations(room_combination, 2))
        for edge in possible_edges:
            if g.has_edge(edge[0], edge[1]):
                sub_graph.add_edge(edge[0], edge[1])
        # check if the graph is fully connected
        if nx.is_connected(sub_graph):
            valid_combinations.append(room_combination[:-1])

    # make int
    valid_combinations = [list(map(int, combi)) for combi in valid_combinations]

    # limit room repetition
    # if a combination is discarted if there exists one that contains at least two same rooms
    if limit_room_repetition:
        valid_combinations = filter_combinations(valid_combinations)



    return valid_combinations

def filter_combinations(valid_combinations):
    def check_validity(combi, valid_combi):
        return len(set(combi).intersection(valid_combi)) < 2

    # unique values in combinationa
    all_numbers = set(itertools.chain(*valid_combinations))

    valid_valid_combinations = []
    included_numbers = set()

    for combi in valid_combinations:
        if all(check_validity(combi, valid_combi) for valid_combi in valid_valid_combinations):
            valid_valid_combinations.append(combi)
            included_numbers.update(combi)

    # Ensure every number appears at least once
    missing_numbers = all_numbers - included_numbers
    for num in missing_numbers:
        for combi in valid_combinations:
            if num in combi:
                valid_valid_combinations.append(combi)
                included_numbers.update(combi)
                break
            else:
                # Continue if the inner loop wasn't broken.
                continue
            break


    return valid_valid_combinations