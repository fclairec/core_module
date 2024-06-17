feature_tasks = ["centroid", "pca_and_extents"]
mre_element_types = ["Wall", "Ceiling", 'Floor', 'Column', 'mech_plumb']  # "Floor", "Ceiling",
transition_element_types = ['Door', 'Window', 'Stair']
# excluded'IfcOpeningElement',

# current ifc models
current_models = {
"A" : "building_A_003_m.ifc",
"B" : "building_B_cms.ifc",
"C" : "building_C_dhub_eg_selection.ifc",
"A_test": "building_A_test.ifc",
"A_chairs": "building_A_chairs.ifc",
"B_test": "building_B_cms_test.ifc",
"Y": "building_Y.ifc",
"C_test": "building_C_dhub_eg_selection_test.ifc",
"A_l-test": "building_A_003_m_l-shape_test.ifc",
"A_graph-test": "building_A_003_m_graph-test.ifc",
}



#         'Openings': {'IfcOpeningElement': {}},

internali2internalt = {
    1: 'Wall',
    2: 'Floor',
    3: 'Ceiling',
    4: 'Door',
    5: 'Window',
    6: 'Column',
    7: 'Stair',
    8: 'Pipe',
    9: 'Sink',
    10: 'Toilet',
    11: 'Sprinkler',
    12: 'Tank',
    13: 'Duct',
    14: 'Air terminal',
    15: 'Light',
    16: 'Alarm',
    17: 'Sensor',
    18: 'Outlet',
    19: 'Switch',
    20: 'Table',
    21: 'Chair',
    22: 'Bookshelf',
    23: 'Appliance',
    24: 'Space',
    25: 'Proxy',
    999: 'other'
}

# geometric features are calculated from a cluster of points and become attributes of the spg. The following dictionary translates the
# geometric feature names from the spg to the myGraph format
sp_feature_translation_dict = {
    "centroid": {"SPG": ["sp_centroids"], "myGraph": ["cp_x", "cp_y", "cp_z"]},
    "pca": {"SPG": ["pca1", "pca2", "pca3"],
            "myGraph": ["pca1x", "pca1y", "pca1z", "pca2x", "pca2y", "pca2z", "pca3x", "pca3y", "pca3z"]},
    "extents": {"SPG": ["sp_extents"], "myGraph": ["extent_pca1", "extent_pca2", "extent_pca3"]},
}

enrichment_feature_dict = {
    "label": "type_int",
    "color": "color"
}

int2color = {
    1: [112, 48, 160, 1.0],
    2: [0, 0, 255, 1.0],
    3: [124, 130, 0, 1.0],
    4: [255, 192, 0, 1.0],
    5: [124, 255, 0, 1.0],
    6: [197, 90, 17, 1.0],
    7: [255, 128, 0, 1.0],
    8: [255, 2, 0, 1.0],
    9: [0, 255, 255, 1.0],
    10: [0, 255, 50, 1.0],
    11: [0, 50, 255, 1.0],
    12: [255, 0, 255, 1.0],
    13: [255, 0, 50, 1.0],
    14: [255, 255, 0, 1.0],
    15: [0, 255, 0, 1.0],
    16: [200, 193, 209, 1.0],
    17: [44, 160, 21, 1.0],
    18: [158, 124, 137, 1.0],
    19: [134, 114, 157, 1.0],
    20: [87, 240, 51, 1.0],
    21: [13, 63, 34, 1.0],
    22: [99, 193, 98, 1.0],
    23: [39, 120, 164, 1.0],
    24: [200, 162, 89, 1.0],
    25: [121, 107, 0, 1.0],
    999: [128, 60, 21, 1.0]
}

discipline_colors = {
    'ARC': [0, 0, 255, 1.0],
    'PLB': [255, 0, 0, 1.0],
    'VTL': [0, 255, 0, 1.0],
    'EL': [255, 255, 0, 1.0],
    'FUR': [0, 255, 255, 1.0]
}

""" 'ControlElement': ['IfcActuator', 'IfcController', 'IfcSensor', 'IfcUnitaryControlElement', 'IfcUnitaryEquipment'],
 'HVACComponent': ['IfcAirTerminal', 'IfcDuctFitting', 'IfcDuctSegment', 'IfcDuctSilencer', 'IfcSpaceHeater'],
 'SafetyDevice': ['IfcAlarm', 'IfcFireSuppressionTerminal', 'IfcProtectiveDevice'],
 'Cabling': ['IfcCableCarrierFitting', 'IfcCableCarrierSegment', 'IfcCableFitting', 'IfcJunctionBox'],
 'HeatingDevice': ['IfcCoil'],
 'FlowControl': ['IfcDamper', 'IfcValve'],
 'Accessory': ['IfcDiscreteAccessory'],
 'ElectricalDevice': ['IfcElectricDistributionBoard', 'IfcElectricTimeControl', 'IfcOutlet', 'IfcSwitchingDevice'],
 'StructuralElement': ['IfcFooting'],
 'Lighting': ['IfcLightFixture'],
 'Plumbing': ['IfcSanitaryTerminal', 'IfcWasteTerminal'],
 'BuildingComponent': ['IfcBuilding', 'IfcBuildingStorey', 'IfcSite']"""











def get_int2discipline(ifc_parsing_dict):
    out = {}
    for expert_model_name, int2ifc_map in ifc_parsing_dict.items():
        for internal_class, ifc_classes in int2ifc_map.items():
            out[internal_class] = expert_model_name

    return out

