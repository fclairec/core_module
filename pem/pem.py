from typing import List, Union, Dict
import pandas as pd
from pathlib import Path
import numpy as np
from bim_spatial_queries.utils.BuildingElement import BuildingElement
from bim_spatial_queries.utils.BuildingSubElement import BuildingSubElement
from bim_spatial_queries.utils.BuildingFace import BuildingFace

# TODO get rid of
from bim_spatial_queries.core_module.default_config.config import transition_element_types, internali2internalt
from bim_spatial_queries.core_module.utils.general_functions import invert_dict_simple


def sort_pairs(pairs: List[tuple], relevant_el_type: str, pem: pd.DataFrame) -> np.ndarray:
    """sort pairs such that the first element is of type relevant_el_type, discart if none of them are
    :param pairs: list of tuples with element_id, space_id (unsorted)
    :param relevant_el_type: type of the relevant element
    :param pem: project element map"""
    # add empty type columns
    pairs = pd.DataFrame(pairs, columns=["element1", "element2"])

    pairs["type1"] = ""
    pairs["type2"] = ""

    # add type txt to adjacencies
    for i, pair in pairs.iterrows():
        pair["type1"] = pem.loc[pair["element1"]]["type_txt"]
        pair["type2"] = pem.loc[pair["element1"]]["type_txt"]
        # add to adj_df
        pairs.loc[i] = pair

    # keep only rows where one of both is a relevant element
    adj_df = pairs[(pairs["type1"] == relevant_el_type) | (pairs["type2"] == relevant_el_type)]

    mask = adj_df['type1'] != relevant_el_type
    adj_df.loc[mask, ['element1', 'element2']] = adj_df.loc[mask, ['element2', 'element1']].values
    adj_df.loc[mask, ['type1', 'type2']] = adj_df.loc[mask, ['type2', 'type1']].values
    # make cols element1 and element2 to ints
    adj_df.loc[:, 'element1'] = adj_df['element1'].astype(int)
    adj_df.loc[:, 'element2'] = adj_df['element2'].astype(int)

    # get rid of the type columns
    adj_df = adj_df[["element1", "element2"]]
    return adj_df.values


def update_associated_spaces(containments: List[tuple], adjacencies: List[tuple], instances: List[BuildingElement],
                             pem_file: Path):
    """ function that updates the pem with element containment and adjacencies.

    :param containments: list of tuples with element_id, space_id (unsorted)
    :param adjacencies: list of tuples with element_id, space_id (unsorted)
    :param instances: list of BuildingElement instances
    :param pem_file: path to the project element map
    """

    pem = load_pem(pem_file, mode="design")

    adj_formatted = sort_pairs(adjacencies, "Space", pem)
    cont_formatted = sort_pairs(containments, "Space", pem)

    instance_wise_adjacencies = {instance.guid_int: [] for instance in instances}
    instance_wise_containments = {instance.guid_int: [] for instance in instances}

    for space_id, element_id in adj_formatted:
        instance_wise_adjacencies[element_id].append(space_id)

    for space_id, element_id in cont_formatted:
        instance_wise_containments[element_id].append(space_id)

    # format the space associations
    instance_wise_adjacencies = {key: ';'.join(map(str, value)) for key, value in instance_wise_adjacencies.items()}
    instance_wise_containments = {key: ';'.join(map(str, value)) for key, value in instance_wise_containments.items()}

    # fill pem["space_id_ajd"] with instance_wise_adjacencies
    pem['space_id_adjacency'] = pem.index.map(instance_wise_adjacencies)
    pem['space_id_containment'] = pem.index.map(instance_wise_containments)

    # fill the rest with -1
    pem['space_id_adjacency'] = pem['space_id_adjacency'].apply(lambda x: -1 if len(x) == 0 else x)
    pem['space_id_containment'] = pem['space_id_containment'].apply(lambda x: -1 if len(x) == 0 else x)

    # print out ifc_guid for the rows that have -1 in space_id_adjacency and space_id_containment
    no_space_association = pem[(pem['space_id_adjacency'] == -1) & (pem['space_id_containment'] == -1)]["ifc_guid"]
    print("Warning: the following elements have no space association:", no_space_association)

    # save updated project element map
    save_pem(pem_file, pem)

    return


def define_spanning_elements(cfg) -> List[int]:
    """ function defines spanning elements from pem file using both containments and adjacencies and default types.
    The pem file is updated and the guid_ints are returned.
    spanning element = building elements that are associated to multiple spaces (transition elements exempted)
    :param cfg: config object
    :return: list of guid_ints of elements that are associated to multiple spaces"""


    pem_file = cfg.pem_file


    pem = load_pem(pem_file, mode="design")

    internalt2internalini = invert_dict_simple(internali2internalt)
    transition_elements_classes = [internalt2internalini[class_type] for class_type in transition_element_types]
    default_spanning_types = [internalt2internalini[class_type] for class_type in cfg.default_spanning_types]

    # get columns of pem with "space_id_adjacency" and "space_id_containment"
    pem_space_assoc = pem[["space_id_adjacency", "space_id_containment"]].copy()
    pem_space_assoc = pem_space_assoc.applymap(lambda x: x.split(";") if type(x) == str else [])
    pem_space_assoc = pd.concat([pem_space_assoc, pem["type_int"].copy()], axis=1)
    # merge the two columns, if space is double associated, it will appear once
    pem_space_assoc['merged_space_assoc'] = pem_space_assoc.apply(
        lambda row: list(set(row['space_id_adjacency'] + row['space_id_containment'])), axis=1)
    pem_space_assoc['merged_space_assoc'] = pem_space_assoc['merged_space_assoc'].apply(
        lambda x: [i for i in x if i != "-1"])
    pem_space_assoc = pem_space_assoc.drop(columns=["space_id_adjacency", "space_id_containment"])

    # no transition elements
    pem_space_assoc = pem_space_assoc[~pem_space_assoc["type_int"].isin(transition_elements_classes)]

    # decide if spanning based on the length of space associations and default types
    bool_spanning_element = pem_space_assoc.apply(
        lambda row: (len(row['merged_space_assoc']) > 1) or (row["type_int"] in default_spanning_types), axis=1)

    # make column in pem and enter if an element is spanning or not
    pem["spanning_element"] = bool_spanning_element
    pem.loc[pem["type_int"].isin(transition_elements_classes), "spanning_element"] = False

    spanning_guids = pem[pem["spanning_element"]].index.tolist()

    # save updated project element map
    save_pem(pem_file, pem)

    return spanning_guids


def update_new_instances_to_pem(pem_file: str, instances: Dict[str, List[Union[BuildingFace, BuildingSubElement]]],
                                mode) -> Dict[
    str, List[Union[BuildingFace, BuildingSubElement]]]:
    """opens, existing PEM with building instances, adds faces enries to PEM and save it to disk.
    the faces_id get converted from local numbering to global
    :param instances: list of BuildingFace instances
    :param project_element_map_filename: path to the project element map
    """
    pem = load_pem(pem_file, mode="design")

    # remove previously split and merged items and start from scratch
    if mode == "split":
        pem = pem[~pem["instance_type"].isin(["face", "subelement", "aggregate"])]
    elif mode == "merge":
        pem = pem[~pem["instance_type"].isin(["aggregate"])]

    # initialize a new guid that does not exist in the pem file.
    new_guid_int = pem.index.max() + 1

    # add a new column called "parent_element" to the project element map
    if "parent_element" not in pem.columns:
        pem["parent_element"] = np.nan
        pem["composing_elements"] = np.nan

    if mode == "split":
        for key in ["faces", "subelements"]:
            for i, single_room_instance in enumerate(instances[key]):
                # initialize a new entry in the pem file.
                new_face_row = pd.DataFrame(columns=pem.columns)
                new_face_row.loc[0, "guid_int"] = new_guid_int

                space_id = single_room_instance.parent_space
                new_face_row.loc[0, "space_id"] = space_id

                parent_entry = pem.loc[single_room_instance.parent_mre]
                new_face_row.loc[0, "type_int"] = parent_entry["type_int"]
                new_face_row.loc[0, "type_txt"] = parent_entry["type_txt"]
                new_face_row.loc[0, "ifc_guid"] = parent_entry["ifc_guid"]

                # we need a guid_txt to easily recongnize the origin of the element (WIP)
                spanning_id_txt = parent_entry["ifc_guid"]
                space_id_txt = pem.loc[space_id]["ifc_guid"]
                new_instance_guid_txt = spanning_id_txt + ";" + space_id_txt + ";" + str(single_room_instance.guid_int)
                new_face_row.loc[0, "guid_txt"] = new_instance_guid_txt

                new_face_row.loc[0, "instance_type"] = single_room_instance.instance_type
                new_face_row.loc[0, "spanning_element"] = 0

                new_face_row.loc[0, "parent_element"] = single_room_instance.parent_mre

                # change interface instance_guid_int to a global one
                single_room_instance.update_guid_int(new_guid_int)
                new_face_row.set_index("guid_int", inplace=True)

                # update pem:
                # 1. add new element as new row
                pem = pd.concat([pem, new_face_row], ignore_index=True)
                # 2. change instance_type from element to rewritten in parent_entry
                pem.loc[single_room_instance.parent_mre, "instance_type"] = "rewritten"

                new_guid_int += 1


    elif mode == "merge":
        for i, aggregate_instance in enumerate(instances["aggregates"]):
            # initialize a new entry in the pem file.
            new_face_row = pd.DataFrame(columns=pem.columns)
            new_face_row.loc[0, "guid_int"] = new_guid_int

            space_ids = ';'.join(str(x) for x in aggregate_instance.parent_spaces)
            new_face_row.loc[0, "space_id"] = space_ids

            parent_entries = pem.loc[aggregate_instance.parent_mres]

            types_int_parents = [parent["type_int"] for id, parent in parent_entries.iterrows()]
            if len(set(types_int_parents)) == 1:
                # we juist access the first parent
                types_int = types_int_parents[0]
                types_txt = parent_entries.iloc[0]["type_txt"]
            else:
                # this is the case of wall, column
                types_int = 1
                types_txt = "Wall"
            # take the majority type, if there is a tie, take the first one
            new_face_row.loc[0, "type_int"] = types_int
            new_face_row.loc[0, "type_txt"] = types_txt

            mre_id_txt_lst = [parents["ifc_guid"] for ids, parents in parent_entries.iterrows()]
            spanning_id_txt = ";".join(mre_id_txt_lst)
            new_face_row.loc[0, "ifc_guid"] = spanning_id_txt

            mre_id_int_lst = aggregate_instance.parent_mres
            mre_id_int = ";".join(str(x) for x in mre_id_int_lst)
            new_face_row.loc[0, "parent_element"] = mre_id_int

            composing_elements = aggregate_instance.composing_elements
            new_face_row.loc[0, "composing_elements"] = ";".join(str(x) for x in composing_elements)

            space_guid_txt_lst = [pem.loc[space_id_i]["ifc_guid"] for space_id_i in aggregate_instance.parent_spaces]
            space_id_txt = ";".join(space_guid_txt_lst)

            new_instance_guid_txt = spanning_id_txt + ";" + space_id_txt + ";" + str(aggregate_instance.guid_int)
            new_face_row.loc[0, "guid_txt"] = new_instance_guid_txt

            new_face_row.loc[0, "instance_type"] = "aggregate"
            new_face_row.loc[0, "spanning_element"] = 0

            # change interface instance_guid_int to a global one
            aggregate_instance.update_guid_int(new_guid_int)

            # update pem:
            new_face_row.set_index("guid_int", inplace=True)
            # 1. add new element as new row
            pem = pd.concat([pem, new_face_row], ignore_index=True)
            # 2. change instance_type from element to rewritten in parent_entry

            new_guid_int += 1

        for rewritten_instance in instances["rewritten"]:
            pem.loc[rewritten_instance.guid_int, "instance_type"] = "rewritten"

    if "guid_txt" not in pem.columns:
        pem["guid_txt"] = pem["ifc_guid"]
    else:
        pem["guid_txt"] = pem["guid_txt"].fillna(pem["ifc_guid"])

    save_pem(pem_file, pem)
    return instances


def save_pem(pem_file, pem):
    pem.index.name = "guid_int"
    pem = pem.reset_index()
    pem.to_csv(pem_file, index=False)


def load_pem(pem_file, mode="design"):
    """load project element map from file
    :param pem_file: path to the project element map
    :param mode: design or built"""
    if mode == "design" or mode == "d":
        pem = pd.read_csv(pem_file, sep=',', header=0, index_col="guid_int")
    elif mode == "built" or mode == "b":
        # we can not set spg_label as index because ...
        pem = pd.read_csv(str(pem_file), sep=',', header=0, index_col=0)
    else:
        raise ValueError("mode must be either 'design' or 'built'")
    return pem
