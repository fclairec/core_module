import pandas as pd
from archive.spatial_instance_queries import get_containment_ray_trace
from core_module.graph.DesignGraph import DesignGraph
from core_module.default_config.config import enrichment_feature_dict, transition_element_types
from pathlib import Path
from visualisation.SurfaceModel import SurfaceModel
from default_config.ifc_parsing_config import get_ifc_class_parsing_details
from instance_classes.InstanceCollection import InstanceCollection
from core_module.pem.IfcPEM import IfcPEM

def helios_prep_model(cfg):
    #### Generate second surface model to easily select waypoints in cloud compare - requirement: guids of ceiling elements)
    print("preparing surface model for waypoint selection in cloud compare")
    disciplines = cfg.disciplines.copy()
    disciplines.remove("Rest")
    parsed_ifc_classes = get_ifc_class_parsing_details(disciplines)
    parsed_ifc_classes_names = [x[0] for x in parsed_ifc_classes]
    model = SurfaceModel(Path(cfg.waypoint_selection_file), type_s="b")
    model.ifcconvert(cfg.ifc_file_path, parsed_ifc_classes_names, cfg.ceiling_elements_guid)
    model.load_surface_file()
    model.modify_surface_model(cfg.pem_file, ['viz'])


def helios_waypoint_assignment(cfg, instance_collection: InstanceCollection):
    print("assigning waypoints to spaces")
    waypoints = pd.read_csv(cfg.waypoint_file, index_col=False, header=0)
    space_instances = list(instance_collection.space_instances.values())
    space_containment_list = get_containment_ray_trace(space_instances, waypoints)
    waypoints["space_id"] = space_containment_list
    waypoints.to_csv(cfg.waypoint_file, index=False, header=True)


def get_space_graph(cfg, instance_collection: InstanceCollection):
    print("assembling space graph with transition elements")
    pem = IfcPEM()
    pem.load_pem(cfg.pem_file)
    transit = [list(pem.get_instance_guids_by_attribute_condition("type_txt", t)) for t in transition_element_types]
    transit_flat = [item for sublist in transit for item in sublist]
    selected_guid_ints = list(instance_collection.space_instances.keys())
    selected_guid_ints += transit_flat

    SpaceGraph = DesignGraph()

    SpaceGraph.assemble_graph_files(cfg, "element", selected_guid_ints, False, False)

    enrichment_feature_dict.pop("edge_length", None)
    SpaceGraph.enrich_graph(cfg.pem_file, enrichment_feature_dict, cfg.node_color_legend_file)
    SpaceGraph.graph_to_pkl(cfg.space_graph)
    # plot a networkx graph
    SpaceGraph.plot_graph("space graph", cfg.space_graph_viz)
    a=0