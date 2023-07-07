import numpy as np


class A9Meta:
    image_width = 1920
    image_height = 1200

    class_id_colors = {
        "CAR": [0, 0.8, 0.964705882],
        "TRUCK": [0.337254902, 1, 0.71372549],
        "TRAILER": [0.352941176, 1, 0.494117647],
        "VAN": [0.921568627, 0.811764706, 0.211764706],
        "MOTORCYCLE": [0.725490196, 0.643137255, 0.329411765],
        "BUS": [0.850980392, 0.541176471, 0.525490196],
        "PEDESTRIAN": [0.91372549, 0.462745098, 0.976470588],
        "BICYCLE": [0.694117647, 0.549019608, 1],
        "EMERGENCY_VEHICLE": [0.4, 0.419607843, 0.980392157],
        "OTHER": [0.780392157, 0.780392157, 0.780392157],
    }

    lidar2s1image = np.asarray(
        [
            [7.04216073e02, -1.37317442e03, -4.32235765e02, -2.03369364e04],
            [-9.28351327e01, -1.77543929e01, -1.45629177e03, 9.80290034e02],
            [8.71736000e-01, -9.03453000e-02, -4.81574000e-01, -2.58546000e00],
        ],
        dtype=np.float32,
    )

    lidar2s2image = np.asarray(
        [
            [1546.63215008, -436.92407115, -295.58362676, 1319.79271737],
            [93.20805656, 47.90351592, -1482.13403199, 687.84781276],
            [0.73326062, 0.59708904, -0.32528854, -1.30114325],
        ],
        dtype=np.float32,
    )

    class_name_to_id_mapping = {
        "CAR": 0,
        "TRUCK": 1,
        "TRAILER": 2,
        "VAN": 3,
        "MOTORCYCLE": 4,
        "BUS": 5,
        "PEDESTRIAN": 6,
        "BICYCLE": 7,
        "EMERGENCY_VEHICLE": 8,
        "OTHER": 9,
        "LICENSE_PLATE_LOCATION": 10,
    }

    id_to_class_name_mapping = {
        "0": {
            "class_label_de": "PKW",
            "class_label_en": "Car",
            "color_hex": "#00ccf6",
            "color_rgb": (0, 204, 246),
            "color_rgb_normalized": (0, 0.8, 0.96),
        },
        "1": {
            "class_label_de": "LKW",
            "class_label_en": "Truck",
            "color_hex": "#3FE9B9",
            "color_rgb": (63, 233, 185),
            "color_rgb_normalized": (0.25, 0.91, 0.72),
        },
        "2": {
            "class_label_de": "Anhänger",
            "class_label_en": "Trailer",
            "color_hex": "#5AFF7E",
            "color_rgb": (90, 255, 126),
            "color_rgb_normalized": (0.35, 1, 0.49),
        },
        "3": {
            "class_label_de": "Van",
            "class_label_en": "Van",
            "color_hex": "#EBCF36",
            "color_rgb": (235, 207, 54),
            "color_rgb_normalized": (0.92, 0.81, 0.21),
        },
        "4": {
            "class_label_de": "Motorrad",
            "class_label_en": "Motorcycle",
            "color_hex": "#B9A454",
            "color_rgb": (185, 164, 84),
            "color_rgb_normalized": (0.72, 0.64, 0.33),
        },
        "5": {
            "class_label_de": "Bus",
            "class_label_en": "Bus",
            "color_hex": "#D98A86",
            "color_rgb": (217, 138, 134),
            "color_rgb_normalized": (0.85, 0.54, 0.52),
        },
        "6": {
            "class_label_de": "Person",
            "class_label_en": "Pedestrian",
            "color_hex": "#E976F9",
            "color_rgb": (233, 118, 249),
            "color_rgb_normalized": (0.91, 0.46, 0.97),
        },
        "7": {
            "class_label_de": "Fahrrad",
            "class_label_en": "Bicycle",
            "color_hex": "#B18CFF",
            "color_rgb": (177, 140, 255),
            "color_rgb_normalized": (0.69, 0.55, 1),
        },
        "8": {
            "class_label_de": "Einsatzfahrzeug",
            "class_label_en": "Emergency_Vehicle",
            "color_hex": "#666bfa",
            "color_rgb": (102, 107, 250),
            "color_rgb_normalized": (0.4, 0.42, 0.98),
        },
        "9": {
            "class_label_de": "Unbekannt",
            "class_label_en": "Other",
            "color_hex": "#C7C7C7",
            "color_rgb": (199, 199, 199),
            "color_rgb_normalized": (0.78, 0.78, 0.78),
        },
        "10": {
            "class_label_de": "Nummernschild",
            "class_label_en": "License_Plate",
            "color_hex": "#000000",
            "color_rgb": (0, 0, 0),
            "color_rgb_normalized": (0, 0, 0),
        },
    }
