from copy import deepcopy
from itertools import product

import numpy as np


class OSDAR23Meta:
    rgb_width: int = 2464
    rgb_height: int = 1600

    rgb_highres_width: int = 4112
    rgb_highres_height: int = 2504

    class_colors = [
        ("person", [0, 0.8, 0.964705882]),
        ("catenary_pole", [0.337254902, 1, 0.71372549]),
        ("signal", [0.352941176, 1, 0.494117647]),
        ("signal_pole", [0.921568627, 0.811764706, 0.211764706]),
        ("train", [0.725490196, 0.643137255, 0.329411765]),
        ("switch", [0.850980392, 0.541176471, 0.525490196]),
        ("buffer_stop", [0.91372549, 0.462745098, 0.976470588]),
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
            # [1.22951075e03, -4.60330749e03, 2.01307849e01, -2.88889572e03],
            # [8.01732765e02, -2.96884835e01, -4.59948985e03, -1.67953802e03],
            # [9.99975285e-01, -5.88174706e-03, -3.85152637e-03, -2.05432000e00]
            # [1.22951075e03, -4.60330749e03, 2.01307849e01, 2.88889572e03],
            # [8.01732765e02, -2.96884835e01, -4.59948985e03, 1.67953802e03],
            # [9.99975285e-01, -5.88174706e-03, -3.85152637e-03, 2.05432000e00],
            # [1.23011244e03, -4.61671899e03, 2.02010093e01, -7.54825862e03],
            # [8.02129308e02, -2.97616460e01, -4.61252812e03, -1.68503013e03],
            # [9.99975285e-01, -5.88174706e-03, -3.85152637e-03, -2.04801963e00],
            [1.23011244e03, -4.61671899e03, 2.02010093e01, 2.94630185e03],
            [8.02129308e02, -2.97616460e01, -4.61252812e03, 1.71487925e03],
            [9.99975285e-01, -5.88174706e-03, -3.85152637e-03, 2.05388041e00],
        ],
        dtype=np.float32,
    )
    lidar2rgb_left = np.asarray(
        [
            [1.14148874e03, 4.36786869e03, -1.58215698e03, -3.16172510e03],
            [8.05576582e02, 1.60082949e03, 4.34315174e03, -2.50332491e03],
            [9.99770435e-01, 2.14089134e-02, 8.58207043e-04, -2.05378655e00],
        ],
        dtype=np.float32,
    )

    @staticmethod
    def get_projection_matrix_to_image(id: str) -> np.ndarray:
        if id == "rgb_center":
            return OSDAR23Meta.lidar2rgb_center
        elif id == "rgb_left":
            return OSDAR23Meta.lidar2rgb_left
