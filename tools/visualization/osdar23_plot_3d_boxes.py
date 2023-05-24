import ast
import json
import os
import sys
from argparse import ArgumentParser, Namespace
from glob import glob
from math import radians
from typing import List

import cv2
import matplotlib as mpl
import numpy as np
import open3d as o3d

from .utils import OSDAR23Meta
from .utils.geometry import add_open3d_axis, get_corners


def get_args() -> Namespace:
    """
    Parse given arguments for osdar23_plot_3d_boxes function.

    Returns:
        Namespace: parsed arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-p", "--point_clouds_folder_path", type=str, required=True)
    parser.add_argument("-d", "--detections_folder_path", type=str, required=True)
    parser.add_argument("-i", "--index", type=int, required=False, default=0)
    parser.add_argument("--point_size", type=int, required=False, default=2)
    parser.add_argument("--enable_coord_frame", type=bool, required=False, default=False)
    parser.add_argument("--enable_color_distance", type=bool, required=False, default=False)
    parser.add_argument("--enable_color_detection", type=bool, required=False, default=False)

    return parser.parse_args()


def osdar23_plot_3d_boxes(
    input_folder_path_point_clouds: str,
    input_folder_path_detections: str,
    index: int = 0,
    point_size: int = 2,
    show_coordinate_frame: bool = False,
    color_distance: bool = False,
    color_detection: bool = False,
):
    pass


if __name__ == "__main__":
    args = get_args()
    osdar23_plot_3d_boxes(
        input_folder_path_point_clouds=args.input_folder_path_point_clouds,
        input_folder_path_detections=args.input_folder_path_detections,
        index=args.index,
        point_size=args.point_size,
        show_coordinate_frame=args.show_coordinate_frame,
        color_distance=args.color_distance,
        color_detection=args.color_detection,
    )
