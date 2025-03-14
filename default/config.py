import os.path as osp
from yacs.config import CfgNode as CN
import yaml
import copy



_C = CN()
_C.seed = 42

_C.root_root_dir = '/media/fio/Daten 1/scanbim/data'
_C.ifc_pool = '/media/fio/Daten 1/scanbim/ifc_models'
_C.waypoint_pool = '/media/fio/Daten 1/scanbim/waypoint_files'
_C.real_point_cloud_pool = '/media/fio/Daten 1/scanbim/real_point_clouds'
_C.building_project = "A"
_C.mode = "d"
_C.IFC_CONVERT = "/home/fio/special_pkgs/IfcConvert"


_C.design = CN()
_C.design.IFC_CONVERT = "/home/fio/special_pkgs/IfcConvert"
_C.design.setup_name = "setup_default"
_C.design.ifc_file_type = "single" # multiple if one file per discipline

_C.design.ifc_file = CN()
_C.design.ifc_file.ALL = "setup_default"
_C.design.ifc_file.ARC = "setup_default"
_C.design.ifc_file.VTL = "setup_default"
_C.design.ifc_file.PLB = "setup_default"
_C.design.ifc_file.EL = "setup_default"
_C.design.ifc_file.FUR = "setup_default"
_C.design.ifc_file.Rest = "setup_default"

_C.design.ifc_parsing_cfg = "ifc_parsing_dict_default"

_C.design.disciplines = []

_C.design.d_tol = CN()
_C.design.d_tol.elements = 0.02 #(smaller than thinnest wall)
_C.design.d_tol.face = 0.02  # 2cm
_C.design.d_tol.merges = 0.05  # 2cm
_C.design.n_tol = 0.9

_C.design.sampling_density = 400
_C.design.voxel_size = 0.1




_C.built = CN()
_C.built.IFC_CONVERT = "/home/fio/special_pkgs/IfcConvert"
_C.built.setup_name = "setup_default"

_C.built.ifc_file_type = "single" # multiple if one file per discipline

_C.built.ifc_file = CN()
_C.built.ifc_file.ALL = "setup_default"
_C.built.ifc_file.ARC = "setup_default"
_C.built.ifc_file.VTL = "setup_default"
_C.built.ifc_file.PLB = "setup_default"
_C.built.ifc_file.EL = "setup_default"
_C.built.ifc_file.FUR = "setup_default"
_C.built.ifc_file.Rest = "setup_default"

_C.built.ifc_parsing_cfg = "ifc_parsing_dict_default"

_C.built.disciplines = []

_C.built.d_tol = CN()
_C.built.d_tol.elements = 0.02
_C.built.d_tol.face = 0.02
_C.built.d_tol.merges = 0.02
_C.built.n_tol = 0.9

_C.built.pcd_type = "helios"
_C.built.point_cloud = "setup_default"


_C.built.waypoints = "general.txt"
_C.built.simulation = CN()
_C.built.simulation.overwrite_existing = False
_C.built.simulation.test = False
_C.built.simulation.accuracy = 0.005
_C.built.simulation.output_legs = False

_C.built.voxel_size = 0.01
_C.built.d_max = 0.02

_C.built.subsets = CN()
_C.built.subsets.split_type = ["wp"]
_C.built.subsets.room_nb = [1]


_C.built.ceiling_elements_guid = ["3kOisf8iHDQ9b60gokQk6c", "2NoCROD1P8KxbahrIQpFr7", "3L2L5EHTz56ef40SrZlEr6",
                        "3L2L5EHTz56ef40SrZlEr5", "3BmeJtEDj3AQO77Os2w6ZR", "0P7YxrKvv9pAandyAj8CEC",
                        "08Kju84ofEuAMc6WrE1A36"]




def dict_to_flat_list(d, mode):
    flat_list = []
    for key, value in d.items():
        if isinstance(value, dict):
            sub_list = dict_to_flat_list(value, mode+"."+key)
            flat_list = flat_list + sub_list
        else:
            flat_list.append(mode+"."+key)
            flat_list.append(value)
    return flat_list

def update_exp_config(cfg_t, cfg_args, ensure_dir=True):

    # read yaml file normally

    import yaml
    with open(cfg_args, 'r') as file:
        prime_service = yaml.safe_load(file)
    list_cfgs = []
    modes = ["design", "built"]
    # set up d_1, d_2, b_1, b_2 is for all buildings
    for mode in modes:
        setups = copy.deepcopy(prime_service[mode])
        for setup_id, params in setups.items():
            params = copy.deepcopy(params)
            params["disciplines"] = unlock_d(params["disciplines"])
            setup_list = dict_to_flat_list(params, mode)
            setup_list = setup_list + ["mode", mode] + ["building_project", prime_service["building_projects"][0]]
            cfg = copy.deepcopy(cfg_t)
            cfg.defrost()
            cfg.merge_from_list(setup_list)
            list_cfgs.append(cfg)

    return list_cfgs



def unlock_d(d_short):
    if d_short[0] == "ARC":
        return d_short
    d_long = {
        "A": "ARC",
        "V": "VTL",
        "E": "EL",
        "P": "PLB",
        "F": "FUR",
        "R": "Rest"
    }
    d_long_list = [d_long[d] for d in d_short]
    return d_long_list
