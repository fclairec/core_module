from copy import deepcopy
import matplotlib
import numpy as np
class InstanceCollection:
    def __init__(self):
        self.element_instances = {}
        self.face_instances = {}
        self.segment_instances = {}
        self.space_instances = {}
        self.aggregate_instances = {}
        self.rewritten_instances = {}
        self.spanning_instances = {}
        self.opening_instances = {}

        self.unindexed_faces = []
        self.unindexed_segments = []
        self.unindexed_aggregates = []

    def add_instances(self, instances: list):
        for instance in instances:
            self.add_instance(instance)


    def add_instance(self, instance):
        instance_type = instance.instance_type
        if instance_type == "element":
            self.element_instances[instance.guid_int] = instance
        elif instance_type == "opening":
            self.opening_instances[instance.guid_int] = instance
        elif instance_type == "face":
            if instance.guid_int is None:
                self.unindexed_faces.append(instance)  # before indexed in geometrical order
            else:
                self.face_instances[instance.guid_int] = instance
        elif instance_type == "segment":
            if instance.guid_int is None:
                self.unindexed_segments.append(instance)  # before indexed in geometrical order
            else:
                self.segment_instances[instance.guid_int] = instance
        elif instance_type == "space":
            self.space_instances[instance.guid_int] = instance
        elif instance_type == "aggregate":
            if instance.guid_int is None:
                self.unindexed_aggregates.append(instance)  # before indexed in geometrical order
            else:
                self.aggregate_instances[instance.guid_int] = instance
        else:
            raise ValueError(f"Instance type {instance_type} is not supported.")


    def get_flat_list(self, all=False):
        if all:
            return list(self.element_instances.values()) + list(self.face_instances.values()) \
            + list(self.segment_instances.values()) + list(self.space_instances.values()) \
            + list(self.aggregate_instances.values()) + list(self.rewritten_instances.values()) \
            + list(self.spanning_instances.values()) + list(self.opening_instances.values())
        else:
            return list(self.element_instances.values()) + list(self.face_instances.values()) \
            + list(self.aggregate_instances.values()) + list(self.segment_instances.values())

    def get_instance_dict(self, types):
        """ returns one single dict of instances with the given types (face, subemeneten, space, aggregate)"""
        instances = {}
        if "element" in types:
            instances.update(self.element_instances)
        if "face" in types:
            instances.update(self.face_instances)
        if "segment" in types:
            instances.update(self.segment_instances)
        if "space" in types:
            instances.update(self.space_instances)
        if "aggregate" in types:
            instances.update(self.aggregate_instances)
        if "rewritten" in types:
            instances.update(self.rewritten_instances)
        if "spanning" in types:
            instances.update(self.spanning_instances)
        if "opening" in types:
            instances.update(self.opening_instances)
        return instances

    def get_instances(self, guid_int_list: list):
        instances = []
        for guid_int in guid_int_list:
            if guid_int in self.element_instances:
                instances.append(self.element_instances[guid_int])
            elif guid_int in self.face_instances:
                instances.append(self.face_instances[guid_int])
            elif guid_int in self.segment_instances:
                instances.append(self.segment_instances[guid_int])
            elif guid_int in self.space_instances:
                instances.append(self.space_instances[guid_int])
            elif guid_int in self.aggregate_instances:
                instances.append(self.aggregate_instances[guid_int])
            elif guid_int in self.rewritten_instances:
                instances.append(self.rewritten_instances[guid_int])
            elif guid_int in self.spanning_instances:
                instances.append(self.spanning_instances[guid_int])
            elif guid_int in self.opening_instances:
                instances.append(self.opening_instances[guid_int])
            else:
                raise ValueError(f"Instance with guid_int {guid_int} not found.")
        return instances

    def set_spanning_instances(self, spanning_element_ids: list):
        """ function that reassigns the instances from the element_instances to the spanning_instances"""
        spanning_instances = {}
        for guid_int in spanning_element_ids:
            if guid_int in self.element_instances:
                spanning_instances[guid_int] = self.element_instances[guid_int]
                del self.element_instances[guid_int]
                spanning_instances[guid_int].instance_type = "spanning"
            else:
                raise ValueError(f"Instance with guid_int {guid_int} not found in element_instances.")
        self.spanning_instances = spanning_instances

    def set_rewritten_instances(self, ids: list):
        # move from spanning to rewritten
        for guid_int in ids:
            if guid_int in self.element_instances:
                self.rewritten_instances[guid_int] = self.element_instances[guid_int]
                del self.element_instances[guid_int]
                self.rewritten_instances[guid_int].instance_type = "rewritten"
            elif guid_int in self.face_instances:
                self.rewritten_instances[guid_int] = self.face_instances[guid_int]
                del self.face_instances[guid_int]
                self.rewritten_instances[guid_int].instance_type = "rewritten"
            else:
                raise ValueError(f"Instance with guid_int {guid_int} not found in spanning_instances.")

    def set_indexed_faces(self):
        for face in self.unindexed_faces:
            if face.guid_int is not None:
                self.face_instances[face.guid_int] = face
            else:
                raise ValueError(f"Instance with guid_int {face.guid_int} not found in face_ids.")
        self.unindexed_faces = []


    def color_instances(self, by, inst_types):
        cmap = matplotlib.colormaps.get_cmap("turbo")
        relevant_inst_ids = [i for type in inst_types for i in getattr(self, f"{type}_instances").keys()]

        if by == "instance":
            nb = len(relevant_inst_ids)
            color_map = cmap(np.linspace(0, 1, nb))
            instance_color_dict = {guid_int: color_map[i] for i, guid_int in enumerate(relevant_inst_ids)}
        else:
            raise ValueError(f"Coloring by {by} is not supported.")

        for inst_type in inst_types:
            for guid_int, instance in getattr(self, f"{inst_type}_instances").items():
                if by == "instance":
                    instance.color = instance_color_dict[guid_int]
                else:
                    raise ValueError(f"Coloring by {by} is not supported.")
        return self.get_instance_dict(inst_types)

    def get_element_instances_by_discipline(self, discipline):
        instances = {}
        if discipline == "ALL":
            return self.element_instances
        else:
            for guid_int, instance in self.element_instances.items():
                if instance.discipline_txt == discipline:
                    instances[guid_int] = instance

        return instances



