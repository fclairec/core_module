from core_module.pem.PEM import PEM
from instance_classes.InstanceCollection import InstanceCollection
import ast
import numpy as np
from core_module.default_config.config import transition_element_types

class IfcPEM(PEM):
    def __init__(self):
        super().__init__(mode="d")

    def add_instance_entry(self, instance_attributes: dict):
        self.update(**instance_attributes)

    def update_associated_spaces(self, adjacencies, containments):
        """ Update the PEM with the adjacencies and containments.
        input: adjacencies, containments - list of tuples (guid_int, guid_int) unsorted
        """

        space_instance_ids = self.get_instance_guids_by_type("space")

        instance_wise_adjacencies = {guid_int: [] for guid_int in self.guid_int}
        instance_wise_space_ctn = {guid_int: [] for guid_int in self.guid_int}

        relations = {"adj": adjacencies,
                     "ctn": containments}

        for rel_type, relations in relations.items():
            for relation in relations:
                space_loc = 0 if relation[0] in space_instance_ids else 1 if relation[1] in space_instance_ids else None
                if space_loc is None:
                    continue
                else:
                    space_el = relation[space_loc]
                    non_space_el = relation[1 - space_loc]
                    if rel_type == "adj":
                        instance_wise_adjacencies[non_space_el].append(space_el)
                    else:
                        instance_wise_space_ctn[non_space_el].append(space_el)

        update_dict = {"space_id_adjacency": list(instance_wise_adjacencies.values()),
                       "space_id_containment": list(instance_wise_space_ctn.values())}

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

    def update_element_room_affiliation(self, updated_instances: InstanceCollection):
        for _, instance in updated_instances.element_instances.items():
            adj_space = self.space_id_adjacency[instance.guid_int]
            ctn_space = self.space_id_containment[instance.guid_int]
            rel_spaces = ast.literal_eval(adj_space) + ast.literal_eval(ctn_space)
            if len(rel_spaces) == 1:
                self.update_instance_attribute(instance.guid_int, **{"room_id": rel_spaces[0]})
            else:
                integ = updated_instances.space_instances.keys()
                new_id = max(integ) + 1
                self.update_instance_attribute(instance.guid_int, **{"room_id": new_id})

        for _, instance in updated_instances.space_instances.items():
            self.update_instance_attribute(instance.guid_int, **{"room_id": instance.guid_int})


    def get_spanning_elements(self, cfg):
        """ spanning elements are by default all walls and slabs plus all other elements that intersect with more than
        one space"""
        if np.isnan(self.spanning_element[0]):
            default_sp_el = [id for id in self.guid_int if self.get_instance_entry(id)["type_txt"] in cfg.default_spanning_types]

            space_id_adjacency = [ast.literal_eval(x) for x in self.space_id_adjacency]
            space_id_containment = [ast.literal_eval(x) for x in self.space_id_containment]
            merged_space_id = [list(set(x + y)) for x, y in zip(space_id_adjacency, space_id_containment)]
            spanning_elements_maks = np.array([(len(x) > 1) for x in merged_space_id])
            sp_el_guids = np.array(self.guid_int)[spanning_elements_maks].tolist() + default_sp_el
            for guid_int in sp_el_guids:
                if self.get_instance_entry(guid_int)["type_txt"] in transition_element_types:
                    spanning_elements_maks[self.guid_int.index(guid_int)] = False
                else:
                    spanning_elements_maks[self.guid_int.index(guid_int)] = True
            sp_el_guids = np.array(self.guid_int)[spanning_elements_maks].tolist()
            self.spanning_element = spanning_elements_maks
        else:
            sp_el_guids = [self.guid_int[i] for i in range(len(self.guid_int)) if self.spanning_element[i]]

        return sp_el_guids



