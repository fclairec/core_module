import pandas as pd
from archive.spatial_instance_queries import get_containment_ray_trace
from core_module.graph.DesignGraph import DesignGraph
from core_module.default_config.config import enrichment_feature_dict, transition_element_types
from core_module.pem.io import load_pem
from pathlib import Path
from visualisation.SurfaceModel import SurfaceModel
from default_config.ifc_parsing_config import ifc_parsing_dict, get_relevant_ifc_classes

def helios_prep_model(cfg):
    #### Generate second surface model to easily select waypoints in cloud compare - requirement: guids of ceiling elements)
    disciplines = cfg.disciplines.copy()
    disciplines.remove("Rest")
    parsed_ifc_classes = get_relevant_ifc_classes(ifc_parsing_dict, disciplines)
    parsed_ifc_classes_names = [x[0] for x in parsed_ifc_classes]
    model = SurfaceModel(Path(cfg.waypoint_selection_file), type_s="b")
    model.ifcconvert(cfg.ifc_file_path, parsed_ifc_classes_names, cfg.ceiling_elements_guid)
    model.load_surface_file()
    model.modify_surface_model(cfg.pem_file, ['viz'])


def helios_waypoint_assignment(cfg, space_instances):
    waypoints = pd.read_csv(cfg.waypoint_file, index_col=False, header=0)
    space_containment_list = get_containment_ray_trace(space_instances, waypoints)
    waypoints["space_id"] = space_containment_list
    waypoints.to_csv(cfg.waypoint_file, index=False, header=True)


def get_space_graph(cfg):

    pem = load_pem(cfg.pem_file, mode="design")
    # select only spaces and transition elements
    condition = pem['type_txt'].isin(["Space"] + transition_element_types)
    selected_guid_ints = pem.index[condition].tolist()

    SpaceGraph = DesignGraph()

    SpaceGraph.assemble_graph_files(cfg, adjacency_type="element", faces=False, by_guid_int=selected_guid_ints, feats=False)
    SpaceGraph.enrich_graph(cfg.pem_file, enrichment_feature_dict, cfg.b_node_color_legend_file)
    SpaceGraph.graph_to_pkl(cfg.space_graph)
    # plot a networkx graph
    SpaceGraph.plot_graph("space graph", cfg.space_graph_viz)
    a=0