from core_module.pem.PEM import PEM
import warnings

class PcPEM(PEM):
    def __init__(self, pcd_type):
        if pcd_type == "bim_sampled":
            mode = "d"
        elif pcd_type == "helios":
            mode = "b"
        elif pcd_type == "real":
            mode = "b"
        super().__init__(mode)
        self.pc_type = pcd_type
        self.archived_guid_int = []

    def add_instance_entry(self, instance_attributes: dict):
        self.update(**instance_attributes)

    def get_subset_of_pem(self, guids):
        """ returns a subset of the pem based on the provided guids. returns a new instance of PcPEM"""
        subset_pem = PcPEM(self.pc_type)
        for guid in guids:
            instance = self.get_instance_entry(guid)
            subset_pem.add_instance_entry(instance)
        subset_pem.archived_guid_int = subset_pem.guid_int.copy()

        return subset_pem

    def reindex_spg_label(self, old_new_dict):
        # if id_name_to_replace == spg_id --> replace spg_id
        # remove nan?
        for old_id, new_id in old_new_dict.items():
            if old_id in self.guid_int:
                position = self.guid_int.index(old_id)
                self.guid_int[position] = new_id
            else:
                warnings.warn(f"old_id {old_id} not found in spg_int")

        # TODO remove spg_labels that are not on old_new

    def __str__(self):
        base_str = super().__str__()
        return f"{base_str}, derived_from_ifc={self.derived_from_ifc}"

