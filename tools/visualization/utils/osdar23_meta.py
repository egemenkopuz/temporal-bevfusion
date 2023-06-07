import numpy as np


class OSDAR23Meta:
    rgb_width: int = 2464
    rgb_height: int = 1600

    rgb_highres_width: int = 4112
    rgb_highres_height: int = 2504

    class_colors = {
        "lidar__cuboid__person": [0, 0.8, 0.964705882],
        "lidar__cuboid__catenary_pole": [0.337254902, 1, 0.71372549],
        "lidar__cuboid__signal": [0.352941176, 1, 0.494117647],
        "lidar__cuboid__signal_pole": [0.921568627, 0.811764706, 0.211764706],
        "rgb_center__bbox__person": [0, 0.8, 0.964705882],
        "rgb_center__bbox__catenary_pole": [0.337254902, 1, 0.71372549],
        "rgb_center__bbox__signal": [0.352941176, 1, 0.494117647],
        "rgb_center__bbox__signal_pole": [0.921568627, 0.811764706, 0.211764706],
        "rgb_left__bbox__person": [0, 0.8, 0.964705882],
        "rgb_left__bbox__catenary_pole": [0.337254902, 1, 0.71372549],
        "rgb_left__bbox__signal": [0.352941176, 1, 0.494117647],
        "rgb_left__bbox__signal_pole": [0.921568627, 0.811764706, 0.211764706],
        "rgb_right__bbox__person": [0, 0.8, 0.964705882],
        "rgb_right__bbox__catenary_pole": [0.337254902, 1, 0.71372549],
        "rgb_right__bbox__signal": [0.352941176, 1, 0.494117647],
        "rgb_right__bbox__signal_pole": [0.921568627, 0.811764706, 0.211764706],
        "rgb_highres_center__bbox__person": [0, 0.8, 0.964705882],
        "rgb_highres_center__bbox__catenary_pole": [0.337254902, 1, 0.71372549],
        "rgb_highres_center__bbox__signal": [0.352941176, 1, 0.494117647],
        "rgb_highres_center__bbox__signal_pole": [0.921568627, 0.811764706, 0.211764706],
        "rgb_highres_left__bbox__person": [0, 0.8, 0.964705882],
        "rgb_highres_left__bbox__catenary_pole": [0.337254902, 1, 0.71372549],
        "rgb_highres_left__bbox__signal": [0.352941176, 1, 0.494117647],
        "rgb_highres_left__bbox__signal_pole": [0.921568627, 0.811764706, 0.211764706],
        "rgb_highres_right__bbox__person": [0, 0.8, 0.964705882],
        "rgb_highres_right__bbox__catenary_pole": [0.337254902, 1, 0.71372549],
        "rgb_highres_right__bbox__signal": [0.352941176, 1, 0.494117647],
        "rgb_highres_right__bbox__signal_pole": [0.921568627, 0.811764706, 0.211764706],
    }

    lidar2rgb_center = np.asarray(
        [
            [1.22951075e03, -4.60330749e03, 2.01307849e01, -2.88889572e03],
            [8.01732765e02, -2.96884835e01, -4.59948985e03, -1.67953802e03],
            [9.99975285e-01, -5.88174706e-03, -3.85152637e-03, -2.05432000e00]
            # [1.22951075e03, -4.60330749e03, 2.01307849e01, 2.88889572e03],
            # [8.01732765e02, -2.96884835e01, -4.59948985e03, 1.67953802e03],
            # [9.99975285e-01, -5.88174706e-03, -3.85152637e-03, 2.05432000e00],
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
