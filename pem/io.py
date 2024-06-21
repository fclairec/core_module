import pandas as pd


def save_pem(pem_file, pem, mode="design"):
    if mode == "design" or mode == "d":
        pem.index.name = "guid_int"
        pem = pem.reset_index()
        pem.to_csv(pem_file, index=False)

    else:
        pem.to_csv(pem_file, index=True)


def load_pem(pem_file, mode="design"):
    """load project element map from file
    :param pem_file: path to the project element map
    :param mode: design or built"""
    if mode == "design" or mode == "d":
        pem = pd.read_csv(pem_file, sep=',', header=0, index_col="guid_int")
    elif mode == "built" or mode == "b":
        # we can not set spg_label as index because ...
        pem = pd.read_csv(str(pem_file), sep=',', header=0)
    else:
        raise ValueError("mode must be either 'design' or 'built'")
    return pem
