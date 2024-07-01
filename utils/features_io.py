from core_module.default_config.config import feature_tasks
from core_module.utils_geometric.geometric_features import get_geometric_features
import pandas as pd
from typing import List, Dict, Union
from utils.BuildingFace import BuildingFace
from utils.BuildingElement import BuildingElement
from core_module.default_config.config import sp_feature_translation_dict


def geom_features_to_file(cfg, instances: Dict[str, List[Union[BuildingFace, BuildingElement]]], exists=True):
    # calculation of the geometric features (not to store the whole geometry (difficult to pickle, not stable))
    # d
    features_file = cfg.design.features_file
    # flatten dict to list
    instances = [inst for key in instances for inst in instances[key] if key!="rewritten"]
    features_df = pd.DataFrame(index=[inst.guid_int for inst in instances])

    for i_ in instances:
        sampled_points = i_.sample_points(density=100)
        geometric_features = get_geometric_features(feature_tasks, sampled_points, bbox=i_.shape.bbox)
        for feature_name, value in geometric_features.items():
            column_names = sp_feature_translation_dict[feature_name]["myGraph"]
            features_df.loc[i_.guid_int, column_names] = value

    if exists:

        existing_features_frame = pd.read_csv(features_file, sep=',', header=0, index_col=0)
        # chekc if indeces from features_df are already in existing_features_frame
        if not features_df.index.isin(existing_features_frame.index).all():
            features_df = pd.concat([existing_features_frame, features_df])
        else:
            existing_features_frame.update(features_df)
            features_df = existing_features_frame


    features_df.reset_index(drop=False, inplace=True)
    # rename column Index to guid_int
    features_df.rename(columns={"index": "guid_int"}, inplace=True)
    features_df.to_csv(features_file, index=True, header=True)
