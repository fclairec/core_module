import json
import pandas as pd
import numpy as np

def get_classes_colors(class_int, classes):
    class_colors = {}

    # Generate a random color for each class
    for class_idx, class_name in zip(class_int, classes):
        # Generate random RGB values between 0 and 255
        r = np.random.randint(0, 256) / 255.0
        g = np.random.randint(0, 256) / 255.0
        b = np.random.randint(0, 256) / 255.0

        # Assign the RGB values as a tuple to the class in the dictionary
        class_colors[class_idx] = (r, g, b)

    class_colors[-1] = (0, 0, 0)

    return class_colors


def create_or_load_colors(design_pem_file, class_colors_fp):
    # get the classes, should be saved when making the dataset
    pem = pd.read_csv(design_pem_file, sep=",", header=0)
    classes = pem["type_txt"]
    unique_classes = classes.unique()
    unique_class_int = pem["type_int"].unique()


    # create the class colors, or load them if they exist
    class_colors = None
    if class_colors_fp.exists():
        with open(class_colors_fp, "r") as f:
            class_colors = json.load(f)
        print("Loaded class colors from ", class_colors_fp)
    else:
        class_colors = get_classes_colors(unique_class_int, unique_classes)
        class_colors = {str(k): v for k, v in class_colors.items()}
        with open(class_colors_fp, "w") as f:
            json.dump(class_colors, f)
        print("Saved class colors to ", class_colors_fp)
    return classes, class_colors