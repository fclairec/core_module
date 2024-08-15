from core_module.pem.PEM import PEM
import ast
import numpy as np
from core_module.default_config.config import transition_element_types
from core_module.utils_general.general_functions import invert_dict_list
class IfcPEM(PEM):
    def __init__(self):
        super().__init__(mode="d")

    def add_instance_entry(self, instance_attributes: dict):
        self.update(**instance_attributes)

    def update_associated_spaces(self, relationship_df, filename):
        """ Update the PEM with the adjacencies and containments.
        input: adjacencies, containments - list of tuples (guid_int, guid_int) unsorted
        """
        relationship_df["assigned"] = False
        relations_dict = relationship_df[["pair", "relationship"]].set_index("pair").to_dict()["relationship"]
        relations_dict_inverted = invert_dict_list(relations_dict)
        room_assignment = {guid_int: [] for guid_int in self.guid_int}
        if "contained" in relations_dict_inverted.keys():
            containments = invert_dict_list(relations_dict)["contained"]
            space_instance_ids = self.get_instance_guids_by_type("space")

            for containment in containments:

                space_loc = 0 if containment[0] in space_instance_ids else 1 if containment[1] in space_instance_ids else None
                if space_loc is None:
                    continue
                else:
                    space_el = containment[space_loc]
                    non_space_el = containment[1 - space_loc]
                    room_assignment[non_space_el].append(space_el)
                    relationship_df.loc[relationship_df["pair"] == containment, "assigned"] = True
        if "touching" in relations_dict_inverted.keys():
            touching = invert_dict_list(relations_dict)["touching"]
            space_instance_ids = self.get_instance_guids_by_type("space")

            for touch in touching:
                space_loc = 0 if touch[0] in space_instance_ids else 1 if touch[1] in space_instance_ids else None
                if space_loc is None:
                    continue
                else:
                    space_el = touch[space_loc]
                    non_space_el = touch[1 - space_loc]
                    non_space_el_type = self.get_instance_entry(non_space_el)["type_txt"]
                    if non_space_el_type in transition_element_types:
                        room_assignment[non_space_el].append(space_el)
                        relationship_df.loc[relationship_df["pair"] == touch, "assigned"] = True
                    else:
                        continue



        update_dict = {"room_id": list(room_assignment.values())}
        self.update_attribute(**update_dict)

        # save relationship_df
        # seperate pair column (1,2) into two columns
        """relationship_df["1"] = relationship_df["pair"].apply(lambda x: x[0])
        relationship_df["2"] = relationship_df["pair"].apply(lambda x: x[1])
        relationship_df.drop(columns=["pair"], inplace=True)"""
        relationship_df.to_csv(filename, index=False)




    def add_splitmerge_results(self, instance_collection, step: str, failed_pairs=None):
        """ function that adds new instances from splitmerge to the PEM and updates the rewritten ones in the
        corresponding attribute"""

        if step == "split":
            for instance in instance_collection.unindexed_faces + instance_collection.unindexed_segments:
                instance.update_match_id()
                guid_int = self.assign_new_guid()
                instance.index_instance(guid_int)
                self.add_instance_entry(instance.output_instance_map())
                instance_collection.add_instance(instance)
            instance_collection.unindexed_faces = []
            instance_collection.unindexed_segments = []

            for _, instance in instance_collection.rewritten_instances.items():
                self.update_instance_attribute(instance.guid_int, **{"instance_type": "rewritten"})
                instance.instance_type = "rewritten"

            # the elements that have not been split and assigned a room_id but were also not assigned to a room by
            # rel_type = "contained" are assigned here according to a custom logic here. (doors, and small composite
            # elements s.a. alarms,

            # loop over failed_pairs,
            # get elements from pem that are not assigned to a room check if they are in failed_pairs, if yes. check if
            # door -->

        else:
            for instance in instance_collection.unindexed_aggregates:
                composing_el_id = instance_collection.get_instances(instance.composing_elements)
                match_id = "_".join([el.match_id for el in composing_el_id])
                instance.update_match_id(match_id)
                guid_int = self.assign_new_guid()
                instance.index_instance(guid_int)
                self.add_instance_entry(instance.output_instance_map())
                instance_collection.add_instance(instance)
            instance_collection.unindexed_aggregates = []

            for _, instance in instance_collection.rewritten_instances.items():
                self.update_instance_attribute(instance.guid_int, **{"instance_type": "rewritten"})
                instance.instance_type = "rewritten"

        return instance_collection

    def assign_remaining_elements(self, remaining_elements, relations):
        # for remaining element pairs, check if they have not already lead to an assignment
        # if not, assign based on rules: door to space, switch to space.
        A=0
        for pair, _ in remaining_elements.items():
            pair_tuple = ast.literal_eval(pair)
            if self.get_instance_entry(pair_tuple[0])["type_txt"] == "Space":
                space = pair_tuple[0]
                other = pair_tuple[1]
            elif self.get_instance_entry(pair_tuple[1])["type_txt"] == "Space":
                space = pair_tuple[1]
                other = pair_tuple[0]
            else:
                continue
            if not bool(relations[relations["pair"] == pair_tuple]["assigned"].values[0]):
                if type(self.room_id[self.guid_int.index(other)]) == str:
                    self.room_id[self.guid_int.index(other)] = ast.literal_eval(self.room_id[self.guid_int.index(other)])
                self.room_id[self.guid_int.index(other)].append(space)

            a=0






    def update_room_affiliation(self, updated_instances):
        for _, instance in updated_instances.element_instances.items():
            pos = self.guid_int.index(instance.guid_int)

            self.update_instance_attribute(pos, **{"room_id": rel_spaces})
            """if len(rel_spaces) == 1:
                self.update_instance_attribute(instance.guid_int, **{"room_id": rel_spaces[0]})
            else:
                integ = updated_instances.space_instances.keys()
                new_id = max(integ) + 1
                self.update_instance_attribute(instance.guid_int, **{"room_id": new_id})"""

        #elif stage == "split_merge"
        for _, instance in updated_instances.space_instances.items():
            self.update_instance_attribute(instance.guid_int, **{"room_id": instance.guid_int})



