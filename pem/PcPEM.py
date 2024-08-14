from core_module.pem.PEM import PEM

class PcPEM(PEM):
    def __init__(self, pcd_type):
        mode = "d" if pcd_type == "bim_sampled" else "b"
        super().__init__(mode)
        self.pc_type = pcd_type

    def add_instance_entry(self, instance_attributes: dict):
        self.update(**instance_attributes)

    def get_subset_of_pem(self, guids):
        """ returns a subset of the pem based on the provided guids. returns a new instance of PcPEM"""
        subset_pem = PcPEM(self.pc_type)
        for guid in guids:
            instance = self.get_instance_entry(guid)
            subset_pem.add_instance_entry(instance)
        return subset_pem

    def __str__(self):
        base_str = super().__str__()
        return f"{base_str}, derived_from_ifc={self.derived_from_ifc}"


