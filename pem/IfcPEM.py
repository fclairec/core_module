from core_module.pem.PEM import PEM
from instance_classes.InstanceCollection import InstanceCollection
import ast
import numpy as np
from core_module.default_config.config import transition_element_types
from core_module.utils_general.general_functions import invert_dict_list
class IfcPEM(PEM):
    def __init__(self):
        super().__init__(mode="d")

    def add_instance_entry(self, instance_attributes: dict):
        self.update(**instance_attributes)

    def update_associated_spaces(self, relationship_df):
        """ Update the PEM with the adjacencies and containments.
        input: adjacencies, containments - list of tuples (guid_int, guid_int) unsorted
        """
        relationship_df["pairs"] = relationship_df.apply(lambda x: (int(x["1"]), int(x["2"])), axis=1)
        relations_dict = relationship_df[["pairs", "relationship"]].set_index("pairs").to_dict()["relationship"]
        room_assignment = {guid_int: [] for guid_int in self.guid_int}
        if "contained" in relations_dict.keys():
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
        if "touching" in relations_dict.keys():
            touching = invert_dict_list(relations_dict)["touching"]
            space_instance_ids = self.get_instance_guids_by_type("space")

            for touch in touching:
                space_loc = 0 if touch[0] in space_instance_ids else 1 if touch[1] in space_instance_ids else None
                if space_loc is None:
                    continue
                else:
                    space_el = touching[space_loc]
                    non_space_el = touching[1 - space_loc]
                    non_space_el_type = self.get_instance_entry(non_space_el)["type_txt"]
                    if non_space_el_type in transition_element_types:
                        room_assignment[non_space_el].append(space_el)
                    else:
                        continue



        update_dict = {"room_id": list(room_assignment.values())}
        self.update_attribute(**update_dict)

    def add_splitmerge_results(self, instance_collection: InstanceCollection, step: str):
        """ function that adds new instances from splitmerge to the PEM and updates the rewritten ones in the
        corresponding attribute"""

        if step == "split":
            for instance in instance_collection.unindexed_faces + instance_collection.unindexed_segments:
                guid_int = self.assign_new_guid()
                instance.index_instance(guid_int)
                self.add_instance_entry(instance.output_instance_map())
                instance_collection.add_instance(instance)
            instance_collection.unindexed_faces = []
            instance_collection.unindexed_segments = []

            for _, instance in instance_collection.rewritten_instances.items():
                self.update_instance_attribute(instance.guid_int, **{"instance_type": "rewritten"})
                instance.instance_type = "rewritten"

        else:
            for instance in instance_collection.unindexed_aggregates:
                guid_int = self.assign_new_guid()
                instance.index_instance(guid_int)
                self.add_instance_entry(instance.output_instance_map())
                instance_collection.add_instance(instance)
            instance_collection.unindexed_aggregates = []

            for _, instance in instance_collection.rewritten_instances.items():
                self.update_instance_attribute(instance.guid_int, **{"instance_type": "rewritten"})
                instance.instance_type = "rewritten"

        return instance_collection

    def update_room_affiliation(self, updated_instances: InstanceCollection):
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



