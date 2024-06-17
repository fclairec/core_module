import pandas as pd


def get_element_space_face_map(pem):
    columns_needed = ['guid_int', 'ifc_guid', 'instance_type', 'space_id']
    pem_filtered = pem[columns_needed]

    # Separate the elements and faces based on the instance_type column.
    elements_df = pem_filtered[pem_filtered['instance_type'] == 'element']
    faces_df = pem_filtered[pem_filtered['instance_type'] == 'face']
    faces_df = faces_df.copy()
    faces_df['space_id'] = faces_df['space_id'].astype(int)
    # Since faces are children of elements and are linked by the same ifc_guid, we will use this to map faces to elements.
    # Each element has associated spaces which can be a list in the space_id, whereas each face is associated with a single space.

    # We need to explode the elements_df on space_id to handle the list of spaces.
    elements_df = elements_df.copy()
    elements_df = elements_df.assign(space_id=elements_df['space_id'].str.split(',')).explode('space_id')
    # make space id col to ints
    elements_df['space_id'] = elements_df['space_id'].astype(int)
    # Now, we will map faces to elements based on shared ifc_guid and associated space.
    # First, merge elements with faces on ifc_guid to establish the parent-child relationship.
    element_face_map = pd.merge(elements_df, faces_df, on='ifc_guid', suffixes=('_element', '_face'))

    # We only keep the rows where the face's space_id matches the element's space_id to maintain the correct space-face association.
    element_face_map = element_face_map[element_face_map['space_id_element'] == element_face_map['space_id_face']]

    element_face_map = element_face_map[['guid_int_element', 'guid_int_face', 'space_id_element']]

    return element_face_map



