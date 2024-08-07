from core_module.pem.PEM import PEM

class PcPEM(PEM):
    def __init__(self, pcd_type):
        mode = "d" if pcd_type == "bim_sampled" else "p"
        super().__init__(mode)
        self.pc_type = pcd_type

    def __str__(self):
        base_str = super().__str__()
        return f"{base_str}, derived_from_ifc={self.derived_from_ifc}"


