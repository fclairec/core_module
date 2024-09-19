import copy

import pandas as pd
from src.core_module.graph.DesignGraph import DesignGraph
from src.core_module.default.match_config import enrichment_feature_dict, transition_element_types
from pathlib import Path
from src.visualisation.SurfaceModel import SurfaceModel
from src.ifc_module.ifc_parsing_utils import get_ifc_class_parsing_details
from src.instance_classes.InstanceCollection import InstanceCollection
from src.instance_classes.BuildingElement import BuildingElement
from src.core_module.pem.IfcPEM import IfcPEM
from typing import List
import numpy as np


def helios_waypoint_assignment(cfg, instance_collection: InstanceCollection):
    print("assigning waypoints to spaces")
    waypoints = pd.read_csv(cfg.waypoint_file, index_col=False, header=0)
    space_instances = list(instance_collection.space_instances.values())
    space_containment_list = get_containment_ray_trace(space_instances, waypoints)
    waypoints["space_id"] = space_containment_list
    waypoints.to_csv(cfg.waypoint_file, index=False, header=True)


def get_containment_ray_trace(potential_host_instances: List[BuildingElement],
                              potential_guest_instances: pd.DataFrame) -> np.ndarray:
    """ takes a list of building instances, and a dataframe of points and checks for containment with the trimesh ray trace method
    :param potential_host_instances: list of building instances that are potentially hosts
    :param potential_guest_instances: dataframe of points that are potentially guests in """
    host_mesh_list = {}
    for instance in potential_host_instances:
        i = instance.guid_int
        host_mesh_list[i] = instance.triangulate()
    waypoints = potential_guest_instances[["x", "y", "z"]] + np.array([0, 0, 0.1])
    host_per_point = np.zeros(waypoints.shape[0], dtype=int)
    for id, mesh in host_mesh_list.items():
        result = mesh.ray.contains_points(waypoints)
        host_per_point[result] = id
    return host_per_point


def get_space_graph(cfg, instance_collection: InstanceCollection):
    """ assembles a simple space graph with transition elements."""
    print("assembling space graph with transition elements")
    pem = IfcPEM()
    pem.load_pem(cfg.pem_file)
    transit = [list(pem.get_instance_guids_by_attribute_condition("type_txt", t)) for t in transition_element_types]
    transit_flat = [item for sublist in transit for item in sublist]
    selected_guid_ints = list(instance_collection.space_instances.keys())
    selected_guid_ints += transit_flat

    SpaceGraph = DesignGraph()

    SpaceGraph.assemble_graph_files(cfg, "element", selected_guid_ints, False, False)

    enrichment_f = copy.deepcopy(enrichment_feature_dict)
    enrichment_f.pop("edge_length", None)
    SpaceGraph.enrich_graph(cfg.pem_file, enrichment_f, cfg.node_color_legend_file)
    SpaceGraph.graph_to_pkl(cfg.space_graph)
    # plot a networkx graph
    SpaceGraph.plot_graph("space graph", cfg.space_graph_viz)
    a=0