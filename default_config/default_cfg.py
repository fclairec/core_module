import os.path as osp
from yacs.config import CfgNode as CN
from core_module.utils import common
import yaml



_C = CN()


_C.seed = 42
_C.root_root_dir = '/home/appuser/input_data/experiments'
_C.experiment_name = "CCP"
_C.experiment_dir = osp.join(_C.root_root_dir, _C.experiment_name)
_C.building_projects = ["A", "B", "C"]

_C.project_name = "building_A"
_C.root_dir = osp.join(_C.experiment_dir, _C.project_name)
_C.log_dir = osp.join(_C.root_dir, 'data_log')

_C.design = CN()
_C.design.root_dir = osp.join(_C.root_dir, "d")
_C.design.ifc_file = "design.ifc"
_C.design.ifc_file_path = osp.join(_C.root_dir, "d", _C.design.ifc_file)
_C.design.disciplines = ["ARC"]
_C.design.default_spanning_types = ["Wall", "Ceiling", 'Floor', 'Column']

_C.design.final_adjacency_file = osp.join(_C.design.root_dir, 'd_adjacencies_final.csv')

_C.design.d_tol = CN()
_C.design.d_tol.elements = 0.2
_C.design.d_tol.face = 0.02 # 2cm
_C.design.d_tol.merges = 0.02 # 2cm

_C.built = CN()
_C.built.root_dir = osp.join(_C.root_dir, "b")
_C.built.ifc_file = "built.ifc"
_C.built.ifc_file_path = osp.join(_C.root_dir, "b", _C.built.ifc_file)
_C.built.disciplines = ["ARC"]


# extract_data_from_ifc
_C.design.viz_file = osp.join(_C.design.root_dir, "d_my_graph.ply")
_C.design.adjacency_file = osp.join(_C.design.root_dir, 'd_adjacencies.csv')
_C.design.containment_file = osp.join(_C.design.root_dir, 'd_containments.csv')
_C.design.features_file = osp.join(_C.design.root_dir, 'd_features.csv')
_C.design.pem_file = osp.join(_C.design.root_dir, 'd_project_element_map.csv')
_C.design.surface_file_path = osp.join(_C.design.root_dir, 'd_surface.obj')
_C.design.instances_pickle = osp.join(_C.design.root_dir, 'd_instances.pkl')
_C.design.shapes_pkl = osp.join(_C.design.root_dir, 'd_geometries.pkl')
_C.design.sampled_pcd_filename = osp.join(_C.design.root_dir, 'd_sampled_pcd.ply')
_C.design.class_colors_fp = osp.join(_C.design.root_dir, f"_colors.json")
_C.design.faces_filename = osp.join(_C.design.root_dir, 'd_faces.obj')

_C.design.sampling_density = 400
_C.design.voxel_size = 0.1

# from elements to faces
_C.design.face_adjacency_file = osp.join(_C.design.root_dir, "d_face_adjacency.csv")
_C.design.f_instances_pickle = osp.join(_C.design.root_dir, 'd_f_instances.pkl')
_C.design.f_shapes_pkl = osp.join(_C.design.root_dir, 'd_f_geometries.pkl')

# assemble graph
_C.design.face_graph_file = osp.join(_C.design.root_dir, "d_my_graph_faces.graphml")
_C.design.face_graph_viz_file = osp.join(_C.design.root_dir, "d_my_graph_faces_viz.ply")
_C.design.d_node_color_legend_file = osp.join(_C.design.root_dir, "d_node_color_legend.png")


# prepare helios
_C.built.waypoint_file = osp.join(_C.built.root_dir, "b_waypoints.csv")
_C.built.pem_file = osp.join(_C.built.root_dir, "b_project_element_map.csv")
_C.built.waypoint_selection_file = osp.join(_C.built.root_dir, 'b_waypoint_selection.obj')
_C.built.instances_pickle = osp.join(_C.built.root_dir, 'b_instances.pkl')
_C.built.shapes_pkl = osp.join(_C.built.root_dir, 'b_geometries.pkl')
_C.built.helios_simulation_file = osp.join(_C.built.root_dir,'b_simulation_file.obj')
_C.built.faces_filename = osp.join(_C.built.root_dir, 'b_faces.obj')
_C.built.adjacency_file = osp.join(_C.built.root_dir, 'b_adjacencies.csv')
_C.built.containment_file = osp.join(_C.built.root_dir, 'b_containments.csv')
_C.built.space_adj_file = osp.join(_C.built.root_dir, 'b_space_adj.csv')
_C.built.space_graph = osp.join(_C.built.root_dir, 'b_space_graph.graphml')
_C.built.space_graph_viz = osp.join(_C.built.root_dir, 'b_space_graph.png')
_C.built.face_adjacency_file = osp.join(_C.built.root_dir, "b_face_adjacency.csv")
_C.built.default_spanning_types = ["Wall", "Ceiling", 'Floor', 'Column']

_C.built.ceiling_elements_guid = []

_C.built.d_tol = CN()
_C.built.d_tol.elements = 0.2
_C.built.d_tol.face = 0.02 # 2cm
_C.built.d_tol.merges = 0.02 # 2cm
_C.built.final_adjacency_file = osp.join(_C.design.root_dir, 'd_adjacencies_final.csv')



# manual way point conversion
_C.built.waypoints = "b_waypoints.txt"
_C.built.manual_waypoints_selection = osp.join(_C.built.root_dir, _C.built.waypoints)



# helios simulation
_C.built.simulation = CN()
_C.built.simulation.test_pcd = True
_C.built.simulation.directory = osp.join(_C.built.root_dir, "helios")
_C.built.simulation.results_directory = osp.join(_C.built.simulation.directory, "results")
_C.built.simulation.total_pcd_filename = osp.join(_C.built.root_dir, "b_simulated_pcd.xyz")
_C.built.simulation.total_trajectory_filename = osp.join(_C.built.root_dir, "full_trajectory.txt")


# pcd_graph
_C.built.spg_file = osp.join(_C.built.root_dir, "spg.h5")
_C.built.pc_graph_file = osp.join(_C.built.root_dir, "my_graph.graphml")
_C.built.pc_graph_viz_file = osp.join(_C.built.root_dir, "b_my_graph.ply")
_C.built.b_node_color_legend_file = osp.join(_C.built.root_dir, "b_node_color_legend.png")
_C.built.b_downsampled_pcd = osp.join(_C.built.root_dir, "b_downsampled_pcd.ply")
_C.built.features_file = osp.join(_C.built.root_dir,"b_features.csv")
_C.built.voxel_size = 0.01
_C.built.d_max = 0.02 # can not be lower than voxel size

# make dataset splits
_C.built.splits = CN()
_C.built.splits.dir_template = osp.join(_C.built.root_dir, "split_{}_{}_{}_{}") #subset key, room_nb, rotation(x30/y30), translation(x10,x-10)
_C.built.splits.dirs = []
_C.built.splits.split_pcd = "pcd.las"
_C.built.splits.split_pem = "project_element_map.csv"
_C.built.splits.split_feat = "features.csv"
_C.built.splits.split_spg = "spg.h5"
_C.built.splits.split_viz = "my_graph.ply"
_C.built.splits.split_graph = "my_graph.graphml"
_C.built.splits.split_cfg_log = "cfg.log"

# default experiemental setup
_C.built.experiment_setup = CN()
_C.built.experiment_setup.split_type = ["wp"]
_C.built.experiment_setup.room_nb = [2]
_C.built.experiment_setup.rotations = [[0, 0, 0]]
_C.built.experiment_setup.translations = [[0, 0, 0]]


def update_config(cfg, cfg_args, ensure_dir=True):
    cfg.defrost()
    cfg.merge_from_file(cfg_args)
    cfg = update_dependent_paths(cfg)
    #update_paths(cfg)

    if ensure_dir:

        #general_classes.ensure_dir(cfg.event_dir)
        common.ensure_dir(cfg.log_dir)
    cfg.freeze()

    return cfg


def update_dependent_paths(cfg):
    cfg.experiment_dir = osp.join(_C.root_root_dir, _C.experiment_name)

    cfg.root_dir = osp.join(_C.experiment_dir, _C.project_name)
    cfg.log_dir = osp.join(_C.root_dir, 'data_log')

    cfg.design.root_dir = osp.join(_C.root_dir, "d")
    cfg.design.ifc_file_path = osp.join(_C.root_dir, "d", _C.design.ifc_file)

    cfg.built.root_dir = osp.join(_C.root_dir, "b")
    cfg.built.ifc_file_path = osp.join(_C.root_dir, "b", _C.built.ifc_file)

    # extract_data_from_ifc
    cfg.design.viz_file = osp.join(_C.design.root_dir, "d_my_graph.ply")
    cfg.design.adjacency_file = osp.join(_C.design.root_dir, 'd_adjacencies.csv')
    cfg.design.containment_file = osp.join(_C.design.root_dir, 'd_containments.csv')
    cfg.design.features_file = osp.join(_C.design.root_dir, 'd_features.csv')
    cfg.design.pem_file = osp.join(_C.design.root_dir, 'd_project_element_map.csv')
    cfg.design.surface_file_path = osp.join(_C.design.root_dir, 'd_surface.obj')
    cfg.design.instances_pickle = osp.join(_C.design.root_dir, 'd_instances.pkl')
    cfg.design.shapes_pkl = osp.join(_C.design.root_dir, 'd_geometries.pkl')
    cfg.design.sampled_pcd_filename = osp.join(_C.design.root_dir, 'd_sampled_pcd.ply')
    cfg.design.class_colors_fp = osp.join(_C.design.root_dir, f"_colors.json")
    cfg.design.faces_filename = osp.join(_C.design.root_dir, 'd_faces.obj')

    # from elements to faces
    cfg.design.face_adjacency_file = osp.join(_C.design.root_dir, "d_face_adjacency.csv")
    cfg.design.f_instances_pickle = osp.join(_C.design.root_dir, 'd_f_instances.pkl')
    cfg.design.f_shapes_pkl = osp.join(_C.design.root_dir, 'd_f_geometries.pkl')

    cfg.design.final_adjacency_file = osp.join(_C.design.root_dir, 'd_adjacencies_final.csv')

    # assemble graph
    cfg.design.face_graph_file = osp.join(_C.design.root_dir, "d_my_graph_faces.graphml")
    cfg.design.face_graph_viz_file = osp.join(_C.design.root_dir, "d_my_graph_faces_viz.ply")
    cfg.design.d_node_color_legend_file = osp.join(_C.design.root_dir, "d_node_color_legend.png")


    # prepare helios
    cfg.built.waypoint_file = osp.join(_C.built.root_dir, "b_waypoints.csv")
    cfg.built.pem_file = osp.join(_C.built.root_dir, "b_project_element_map.csv")
    cfg.built.waypoint_selection_file = osp.join(_C.built.root_dir, 'b_waypoint_selection.obj')
    cfg.built.instances_pickle = osp.join(_C.built.root_dir, 'b_instances.pkl')
    cfg.built.shapes_pkl = osp.join(_C.built.root_dir, 'b_geometries.pkl')
    cfg.built.helios_simulation_file = osp.join(_C.built.root_dir,'b_simulation_file.obj')
    cfg.built.faces_filename = osp.join(_C.built.root_dir, 'b_faces.obj')
    cfg.built.adjacency_file = osp.join(_C.built.root_dir, 'b_adjacencies.csv')
    cfg.built.containment_file = osp.join(_C.built.root_dir, 'b_containments.csv')
    cfg.built.space_adj_file = osp.join(_C.built.root_dir, 'b_space_adj.csv')
    cfg.built.space_graph = osp.join(_C.built.root_dir, 'b_space_graph.graphml')
    cfg.built.space_graph_viz = osp.join(_C.built.root_dir, 'b_space_graph.png')
    cfg.built.face_adjacency_file = osp.join(_C.built.root_dir, "b_face_adjacency.csv")

    # manual way point conversion
    cfg.built.manual_waypoints_selection = osp.join(_C.built.root_dir, _C.built.waypoints)
    cfg.built.final_adjacency_file = osp.join(_C.design.root_dir, 'd_adjacencies_final.csv')

    # helios simulation
    cfg.built.simulation.directory = osp.join(_C.built.root_dir, "helios")
    cfg.built.simulation.results_directory = osp.join(_C.built.simulation.directory, "results")
    cfg.built.simulation.total_pcd_filename = osp.join(_C.built.root_dir, "b_simulated_pcd.xyz")
    cfg.built.simulation.total_trajectory_filename = osp.join(_C.built.root_dir, "full_trajectory.txt")


    # pcd_graph
    cfg.built.spg_file = osp.join(_C.built.root_dir, "spg.h5")
    cfg.built.pc_graph_file = osp.join(_C.built.root_dir, "my_graph.graphml")
    cfg.built.pc_graph_viz_file = osp.join(_C.built.root_dir, "b_my_graph.ply")
    cfg.built.b_node_color_legend_file = osp.join(_C.built.root_dir, "b_node_color_legend.png")
    cfg.built.b_downsampled_pcd = osp.join(_C.built.root_dir, "b_downsampled_pcd.ply")
    cfg.built.b_features_file = osp.join(_C.built.root_dir,"b_features.csv")


    # make dataset splits
    cfg.built.splits.dir_template = osp.join(_C.built.root_dir, "split_{}_{}_{}_{}") #subset key, room_nb, rotation(x30/y30), translation(x10,x-10)

    return cfg








def update_config_value(cfg, cfg_args, ensure_dir=True):
    cfg.defrost()
    cfg.merge_from_list(cfg_args)
    cfg = update_dependent_paths(cfg)
    cfg.freeze()
    common.ensure_dir(cfg.root_dir)
    common.ensure_dir(cfg.log_dir)
    return cfg


def update_paths(cfg):
    cfg.root_dir = osp.join('/home/appuser/input_data/experiments', cfg.project_name)
    cfg.data.bim.root_dir = osp.join(cfg.root_dir, 'd')
    cfg.data.bim.features = osp.join(cfg.data.bim.root_dir, 'd_features.csv')
    cfg.data.bim.pem = osp.join(cfg.data.bim.root_dir, 'd_project_element_map.csv')
    cfg.data.bim.pcd = osp.join(cfg.data.bim.root_dir, 'd_sampled_pcd.ply')

    cfg.data.pcd.root_dir = osp.join(cfg.root_dir, 'b')
    cfg.data.pcd.folders = [subset_path for subset_path in osld(cfg.data.pcd.root_dir) if
                            osp.isdir(osp.join(cfg.data.pcd.root_dir, subset_path)) and subset_path.startswith('split')]
    cfg.data.pcd.ids = [d.split("_")[-2] for d in cfg.data.pcd.folders]
    cfg.data.pcd.nb_rooms = [d.split("_")[-1] for d in cfg.data.pcd.folders]
    cfg.data.pcd.pems = [osp.join(cfg.data.pcd.root_dir, fd, f"{id}_project_element_map.csv") for fd, id in
                         zip(cfg.data.pcd.folders, cfg.data.pcd.ids)]
    cfg.data.pcd.pcds = [osp.join(cfg.data.pcd.root_dir, fd, f"{id}.ply") for fd, id in
                         zip(cfg.data.pcd.folders, cfg.data.pcd.ids)]
    cfg.data.pcd.features = [osp.join(cfg.data.pcd.root_dir, fd, f"{id}_features.csv") for fd, id in
                             zip(cfg.data.pcd.folders, cfg.data.pcd.ids)]

    cfg.output_dir = osp.join(cfg.root_dir, 'output_aligner')
    cfg.log_dir = osp.join(cfg.output_dir, 'log')

    common.ensure_dir(cfg.output_dir)
    common.ensure_dir(cfg.log_dir)


def save_cfg_to_yaml(cfg, file_path):
    with open(file_path, 'w') as f:
        # Convert CfgNode to dictionary and dump as YAML
        yaml.safe_dump(cfg.dump(), f, default_flow_style=False)


def load_cfg_from_yaml(file_path):
    with open(file_path) as f:
        # Load YAML file and initialize CfgNode
        cfg_dict = yaml.safe_load(f)
        cfg = CN.load_cfg_from_yaml(cfg_dict)
    return cfg



d_node_attributes = ['cp_x', 'cp_y', 'cp_z', 'pca1x', 'pca1y', 'pca1z', 'pca2x', 'pca2y', 'pca2z',
                   'pca3x', 'pca3y', 'pca3z', 'extent_pca1', 'extent_pca2', 'extent_pca3', 'label', 'color', 'has_face', 'ifc_guid', 'node_type']
b_node_attributes = ['cp_x', 'cp_y', 'cp_z', 'pca1x', 'pca1y', 'pca1z', 'pca2x', 'pca2y', 'pca2z',
                   'pca3x', 'pca3y', 'pca3z', 'extent_pca1', 'extent_pca2', 'extent_pca3', 'label', 'color']
# , 'has_face'