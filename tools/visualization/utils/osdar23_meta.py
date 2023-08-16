from copy import deepcopy
from itertools import product

import numpy as np


class OSDAR23Meta:
    rgb_width: int = 2464
    rgb_height: int = 1600

    rgb_highres_width: int = 4112
    rgb_highres_height: int = 2504

    class_colors = [
        ("person", [0.91372549, 0.462745098, 0.976470588]),
        ("bicycle", [0.694117647, 0.549019608, 1]),
        ("signal", [0, 0.8, 0.964705882]),
        ("catenary_pole", [0.337254902, 1, 0.71372549]),
        ("buffer_stop", [0.352941176, 1, 0.494117647]),
        ("train", [0.921568627, 0.811764706, 0.211764706]),
        ("road_vehicle", [0.4, 0.419607843, 0.980392157]),
        ("signal_pole", [0.725490196, 0.643137255, 0.329411765]),
        ("animal", [0.780392157, 0.780392157, 0.780392157]),
        ("switch", [0.850980392, 0.541176471, 0.525490196]),
        ("crowd", [0.97647059, 0.43529412, 0.36470588]),
        ("wagons", [0.98431373, 0.94901961, 0.75294118]),
        ("signal_bridge", [0.42745098, 0.27058824, 0.29803922]),
        ("flame", [0.0, 0.0, 0.0]),  # BBOX ONLY
        ("drag_shoe", [0.0, 0.0, 0.0]),  # BBOX ONLY
    ]

    sensor_ids = [
        "lidar__cuboid__",
        "rgb_center__bbox__",
        "rgb_left__bbox__",
        "rgb_right__bbox__",
        "rgb_highres_center__bbox__",
        "rgb_highres_left__bbox__",
        "rgb_highres_right__bbox__",
    ]

    class_id_colors = {f"{x}{y[0]}": y[1] for x, y in product(sensor_ids, class_colors)}

    lidar2rgb_center = np.asarray(
        [
            [1.28155029e03, -4.58555989e03, -1.02967448e01, -6.88197425e01],
            [8.46420682e02, 2.22640290e01, -4.58648857e03, 9.36547118e03],
            [9.99968067e-01, 5.41023377e-03, 5.88174706e-03, -7.90217233e-02],
        ],
        dtype=np.float32,
    )
    lidar2rgb_highres_center = np.asarray(
        [
            [2.11443983e03, -7.18523532e03, -1.89987995e02, -1.89435698e03],
            [4.14791337e02, -3.75379653e01, -7.29679085e03, 2.55646787e04],
            [9.93614086e-01, 9.29010435e-03, -1.12448838e-01, 3.18121585e-01],
        ],
        dtype=np.float32,
    )
    lidar2rgb_left = np.asarray(
        [
            [2.73611511e03, -3.90615474e03, 2.24407451e01, 6.00999552e02],
            [8.82958038e02, 3.26109605e02, -4.58687996e03, 9.34507256e03],
            [9.39295629e-01, 3.42440329e-01, 2.14089134e-02, -1.36006017e-01],
        ],
        dtype=np.float32,
    )
    lidar2rgb_highres_left = np.asarray(
        [
            [4.60709278e03, -5.91552604e03, -2.53021286e02, -1.99736624e02],
            [2.98383823e02, 1.12462909e02, -7.29581083e03, 2.56435774e04],
            [9.21559517e-01, 3.68083432e-01, -1.23461101e-01, 4.28171490e-01],
        ],
        dtype=np.float32,
    )
    lidar2rgb_right = np.asarray(
        [
            [-5.15649958e02, -4.73024432e03, -6.19729395e01, -7.77740065e02],
            [8.44808580e02, -2.35699427e02, -4.57878001e03, 9.33346125e03],
            [9.31990264e-01, -3.61951168e-01, 1.96341615e-02, -1.59558444e-01],
        ],
        dtype=np.float32,
    )
    lidar2rgb_highres_right = np.asarray(
        [
            [-6.60454783e02, -7.46284041e03, -4.18444286e02, -2.41693304e03],
            [2.89901993e02, 3.94107215e01, -7.30206100e03, 2.56819982e04],
            [9.25346812e-01, -3.54911295e-01, -1.33308850e-01, 2.34974919e-01],
        ],
        dtype=np.float32,
    )

    @staticmethod
    def get_projection_matrix_to_image(id: str) -> np.ndarray:
        if id == "rgb_center":
            return OSDAR23Meta.lidar2rgb_center
        elif id == "rgb_left":
            return OSDAR23Meta.lidar2rgb_left
        elif id == "rgb_right":
            return OSDAR23Meta.lidar2rgb_right
        elif id == "rgb_highres_center":
            return OSDAR23Meta.lidar2rgb_highres_center
        elif id == "rgb_highres_left":
            return OSDAR23Meta.lidar2rgb_highres_left
        elif id == "rgb_highres_right":
            return OSDAR23Meta.lidar2rgb_highres_right
