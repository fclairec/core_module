import os.path as osp
from yacs.config import CfgNode as CN
from core_module.utils_general import common
import yaml
import shutil




def setup_paths(_C, ensure_paths=True):

    _C.root_dir = osp.join(_C.root_root_dir, _C.building_project)
    _C.log_dir = osp.join(_C.root_dir, 'data_log')

    _c = getattr(_C, _C.mode)

    _c.root_dir = osp.join(_C.root_dir, _c.setup_name)
    _c.instance_dir = osp.join(_c.root_dir, 'instance_geometry')
    _c.ifc_file_path = CN()
    _c.ifc_file_path.ALL = osp.join(_c.root_dir, _c.ifc_file.ALL)
    _c.ifc_file_path.ARC = osp.join(_c.root_dir, _c.ifc_file.ARC)
    _c.ifc_file_path.VTL = osp.join(_c.root_dir, _c.ifc_file.VTL)
    _c.ifc_file_path.PLB = osp.join(_c.root_dir, _c.ifc_file.PLB)
    _c.ifc_file_path.EL = osp.join(_c.root_dir, _c.ifc_file.EL)
    _c.ifc_file_path.FUR = osp.join(_c.root_dir, _c.ifc_file.FUR)
    _c.ifc_file_path.Rest = osp.join(_c.root_dir, _c.ifc_file.Rest)

    _c.surface_file_path = CN()
    _c.surface_file_path.ALL = osp.join(_c.root_dir, 'surface.obj')
    _c.surface_file_path.ARC = osp.join(_c.root_dir, 'surface_arc.obj')
    _c.surface_file_path.VTL = osp.join(_c.root_dir, 'surface_vtl.obj')
    _c.surface_file_path.PLB = osp.join(_c.root_dir, 'surface_plb.obj')
    _c.surface_file_path.EL = osp.join(_c.root_dir, 'surface_el.obj')
    _c.surface_file_path.FUR = osp.join(_c.root_dir, 'surface_fur.obj')
    _c.surface_file_path.Rest = osp.join(_c.root_dir, 'surface_rest.obj')

    _c.instances_pickle = osp.join(_c.root_dir, 'instances.pkl')
    _c.shapes_pkl = osp.join(_c.root_dir, 'geometries.pkl')

    _c.adjacency_file = osp.join(_c.root_dir, '_adjacencies.csv')
    _c.containment_file = osp.join(_c.root_dir, 'spatial_relationships.csv')
    _c.features_file = osp.join(_c.root_dir, 'features.csv')
    _c.pem_file = osp.join(_c.root_dir, 'pem.csv')



    # from elements to faces (instances_filename)
    _c.proc_segments = osp.join(_c.root_dir, 'proc_segments.obj')

    _c.face_adjacency_file = osp.join(_c.root_dir, "face_adjacencies.csv")
    _c.proc_instances_pickle = osp.join(_c.root_dir, 'proc_instances.pkl')
    _c.proc_shapes_pkl = osp.join(_c.root_dir, 'proc_geometries.pkl')
    _c.final_adjacency_file = osp.join(_c.root_dir, 'proc_adjacencies.csv')



    _c.space_adj_file = osp.join(_c.root_dir, '_space_adjacencies.csv')
    _c.space_graph = osp.join(_c.root_dir, 'only_space_graph.graphml')
    _c.space_graph_viz = osp.join(_c.root_dir, 'only_space_graph.png')

    setattr(_C, _C.mode, _c)  # Explicitly assign modified _c back to _C

    # design specific
    if _C.mode == "design":
        _C.design.graph_file = osp.join(_C.design.root_dir, "bim_element_graph.graphml")
        _C.design.graph_viz_file = osp.join(_C.design.root_dir, "bim_element_graph.ply")
        _C.design.sampled_pcd_file = osp.join(_C.design.root_dir, 'sampled_pcd_elements.las')
        _C.design.proc_sampled_pcd_file = osp.join(_C.design.root_dir, 'sampled_pcd_segments.las')
        _C.design.proc_graph_file = osp.join(_C.design.root_dir, "bim_room_graph.graphml")
        _C.design.proc_graph_viz_file = osp.join(_C.design.root_dir, "bim_room_graph.ply")
        _C.design.node_color_legend_file = osp.join(_C.design.root_dir, "color_legend.png")
    elif _C.mode == "built":

        # built specific
        _C.built.waypoint_file = osp.join(_C.built.root_dir, "waypoints.csv")
        _C.built.waypoint_selection_file = osp.join(_C.built.root_dir, 'waypoint_selection.obj')
        _C.built.manual_waypoints_selection = osp.join(_C.built.root_dir, _C.built.waypoints)

        _C.built.simulation.directory = osp.join(_C.built.root_dir, "helios")
        _C.built.simulation.results_directory = osp.join(_C.built.simulation.directory, "results")
        _C.built.simulation.total_pcd_filename = osp.join(_C.built.root_dir, "simulated_pcd.las")
        _C.built.simulation.total_trajectory_filename = osp.join(_C.built.root_dir, "full_trajectory.txt")

        _C.built.spg_file = osp.join(_C.built.root_dir, "spg.h5")
        _C.built.pc_graph_file = osp.join(_C.built.root_dir, "my_graph.graphml")
        _C.built.pc_graph_viz_file = osp.join(_C.built.root_dir, "my_graph.ply")
        _C.built.node_color_legend_file = osp.join(_C.built.root_dir, "node_color_legend.png")
        _C.built.b_downsampled_pcd = osp.join(_C.built.root_dir, "downsampled_pcd.ply")
        _C.built.features_file = osp.join(_C.built.root_dir, "features.csv")

        # make dataset splits
        _C.built.splits = CN()
        _C.built.splits.dir_template = osp.join(_C.built.root_dir, "split_{}_{}_{}_{}") #subset key, room_nb, rotation(x30/y30), translation(x10,x-10)
        _C.built.splits.dirs = []
        _C.built.splits.split_pcd = "pcd.las"
        _C.built.splits.split_pem = "pem.csv"
        _C.built.splits.split_feat = "features.csv"
        _C.built.splits.split_spg = "spg.h5"
        _C.built.splits.split_viz = "my_graph.ply"
        _C.built.splits.split_graph = "my_graph.graphml"
        _C.built.splits.split_cfg_log = "cfg.log"
    else:
        raise ValueError("Invalid mode specified. Must be 'design' or 'built'.")

    if ensure_paths:
        s = create_paths(_C)
        if s == 0:
            return None
    return _C

def create_paths(cfg):
    """ ensures directories are created. if already there, break"""
    """
    if not os.path.exists(cfg.experiment_dir):
        os.makedirs(cfg.experiment_dir)
    """
    if not osp.exists(cfg.root_dir):
        common.ensure_dir(cfg.root_dir)
    if not osp.exists(cfg.log_dir):
        common.ensure_dir(cfg.log_dir)


    #mode = "d" if cfg.design.setup_name != "setup_default" elif cfg.built.setup_name != "setup_default" else None
    mode = "design" if cfg.design.setup_name != "setup_default" else "built" if cfg.built.setup_name != "setup_default" else None
    if mode is None:
        print ("No setup name provided - skipping ")
        return 0
    cfg_s = getattr(cfg, mode)
    if osp.exists(cfg_s.root_dir):
        print(f"Directory {cfg_s.root_dir} already exists. Please rename")
        return 0
    else:
        common.ensure_dir(cfg_s.root_dir)
        common.ensure_dir(cfg_s.instance_dir)

        # copy ifc
        for ifc in cfg_s.ifc_file:
            file_name = getattr(cfg_s.ifc_file, ifc)
            if file_name == "setup_default":
                continue
            d_ifc = osp.join(cfg.ifc_pool, file_name)
            shutil.copy(d_ifc, getattr(cfg_s.ifc_file_path, ifc))


        if mode == "built":
            # copy waypoints
            waypoint_file = osp.join(cfg.waypoint_pool, cfg.built.waypoints)
            shutil.copy(waypoint_file, cfg.built.manual_waypoints_selection)







def update_config_value(cfg, cfg_args, ensure_dir=True):
    cfg.defrost()
    cfg.merge_from_list(cfg_args)
    cfg.freeze()
    return cfg





def save_cfg_to_yaml(cfg, file_path):
    with open(file_path, 'w') as f:
        cfg_str = yaml.safe_load(cfg.dump())
        # Convert CfgNode to dictionary and dump as YAML
        yaml.safe_dump(cfg_str, f, default_flow_style=False, sort_keys=False, allow_unicode=True, indent=2)


def load_cfg_from_yaml(file_path):
    with open(file_path) as f:
        # Load YAML file and initialize CfgNode
        cfg_dict = yaml.safe_load(f)
        cfg = CN.load_cfg_from_yaml(cfg_dict)
    return cfg


