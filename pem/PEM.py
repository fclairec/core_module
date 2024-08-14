import pandas as pd
from core_module.utils_general.general_functions import invert_dict_list
import ast


class PEM:
    def __init__(self, mode="default"):
        self.mode = mode
        self.inst_types_all = ["element", "face", "aggregate", "space", "rewritten", "spanning", "segment"]
        self.guid_int = []
        self.type_int = []
        self.type_txt = []
        self.discipline_int = []
        self.discipline_txt = []
        self.identifier_txt = []
        self.geometry_type = []
        self.color = []
        self.instance_type = []
        self.ifc_guid = []
        self.space_id = []
        self.parent_element = []
        self.composing_elements = []
        self.room_id = []
        self.guid_txt = []
        self.pcd = []

    def update(self, **kwargs):
        self.check_minimum_attr(kwargs)

        for key, value in kwargs.items():
            if hasattr(self, key):
                attr = getattr(self, key)
                attr.append(value)
            else:
                raise AttributeError(f"{key} is not a valid attribute of {self.__class__.__name__}.")
        # add dummy values for missing attributes except for mode
        for key in self.__dict__.keys():
            if key in ["mode", "inst_types_all", "pc_type"]:
                continue
            elif key not in kwargs.keys():
                attr = getattr(self, key)
                attr.append(None)

    def update_attribute(self, **kwargs: dict):
        """ update attributes of the instance. Note that the attributes must be lists of the same length as the
        instance and ordered accordingly"""
        for key, value in kwargs.items():
            if len(value) != len(self.guid_int):
                raise ValueError(f"Length of {key} does not match the length of the instance.")
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"{key} is not a valid attribute of {self.__class__.__name__}.")

    def update_instance_attribute(self, guid_int, **kwargs):
        position = self.guid_int.index(guid_int)
        for key, value in kwargs.items():
            if hasattr(self, key):
                attr = getattr(self, key)
                attr[position] = value
                setattr(self, key, attr)
            else:
                raise AttributeError(f"{key} is not a valid attribute of {self.__class__.__name__}.")

    def check_minimum_attr(self, kwargs):
        minimum_attr = ["guid_int", "type_int", "discipline_txt", "identifier_txt"]
        missing = [attr for attr in minimum_attr if attr not in kwargs.keys()]
        if len(missing) > 0:
            raise ValueError(f"Missing attributes: {missing}, instance can not be created.")

    def get_instance_entry(self, guid_int):
        position = self.guid_int.index(guid_int)
        instance = {key: value[position] for key, value in self.__dict__.items() if key not in ["mode", "inst_types_all", "pc_type"]}
        return instance

    def get_instance_guids_by_type(self, instance_type):
        instance_guid2type = {self.guid_int[i]: self.instance_type[i] for i in range(len(self.guid_int))}
        type2instance_guid = invert_dict_list(instance_guid2type)
        for inst_t in self.inst_types_all:
            if inst_t not in type2instance_guid.keys():
                type2instance_guid[inst_t] = []
        return type2instance_guid[instance_type]

    def get_instance_guids_by_types(self, instance_types):
        all_guids = []
        for intst_type in instance_types:
            lst_guids = self.get_instance_guids_by_type(intst_type)
            all_guids += lst_guids
        return all_guids

    def get_instance_guids_excluding_type(self, instance_types):
        instance_guid2type = {self.guid_int[i]: self.instance_type[i] for i in range(len(self.guid_int)) if self.instance_type[i] not in instance_types}
        type2instance_guid = invert_dict_list(instance_guid2type)
        guids_excluding_type = [guid for key, value in type2instance_guid.items() if key not in instance_types for guid in value]
        return guids_excluding_type


    def get_instance_guids_by_attribute_condition(self, attr, condition):
        instance_attributes = {self.guid_int[i]: getattr(self, attr)[i] for i in range(len(self.guid_int))}
        instance_subset = {key: value for key, value in instance_attributes.items() if value==condition}
        return instance_subset.keys()

    def get_physical_instances(self, attr):
        # exclude instance_type space
        instance_guids = self.get_instance_guids_excluding_type(["space", "rewritten", "spanning"])
        idx = [self.guid_int.index(guid) for guid in instance_guids]
        instance_attributes = {self.guid_int[i]: getattr(self, attr)[i] for i in idx}
        return instance_attributes

    def assign_new_guid(self):
        if len(self.guid_int) == 0:
            return 0
        else:
            return max(self.guid_int) + 1


    def save_pem(self, pem_file):
        pem_dict = {key: value for key, value in self.__dict__.items() if key not in ["mode", "inst_types_all"]}
        pem = pd.DataFrame(pem_dict)
        if self.mode == "d":
            pem.to_csv(pem_file, index=False)

        elif self.mode == "b":
            pem.to_csv(pem_file, index=True)

    def get_feature_vector(self, guids, feature_name):
        feature_vector = [getattr(self, feature_name)[self.guid_int.index(guid)] for guid in guids]
        return feature_vector

    def load_pem(self, pem_file):
        """load project element map from file
        :param pem_file: path to the project element map
        :param mode: design or built"""
        if self.mode == "d":
            pem = pd.read_csv(pem_file, sep=',', header=0)

        elif self.mode == "b":
            # we can not set spg_label as index because ...
            pem = pd.read_csv(str(pem_file), sep=',', header=0)
        else:
            raise ValueError("mode must be either 'd' or 'b'")

        # add pem to attributes
        self.__dict__.update(pem.to_dict(orient='list'))

    def __str__(self):
        return f"ProjectElementMap)"
