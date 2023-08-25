import copy
import os
from typing import List, Optional, Tuple

import cv2
import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt

from ..bbox import LiDARInstance3DBoxes

__all__ = ["visualize_camera", "visualize_lidar", "visualize_map", "visualize_bev_feature"]


OBJECT_PALETTE = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
}

TUMTRAF_OBJECT_PALETTE = {
    "CAR": (0, 204, 246),
    "TRUCK": (63, 233, 185),
    "BUS": (217, 138, 134),
    "TRAILER": (90, 255, 126),
    "VAN": (235, 207, 54),
    "MOTORCYCLE": (185, 164, 84),
    "BICYCLE": (177, 140, 255),
    "PEDESTRIAN": (233, 118, 249),
    "EMERGENCY_VEHICLE": (102, 107, 250),
    "OTHER": (199, 199, 199),
}

OSDAR23_OBJECT_PALETTE = [
    ("lidar__cuboid__person", [0.91372549, 0.462745098, 0.976470588]),
    ("lidar__cuboid__bicycle", [0.694117647, 0.549019608, 1]),
    ("lidar__cuboid__signal", [0, 0.8, 0.964705882]),
    ("lidar__cuboid__catenary_pole", [0.337254902, 1, 0.71372549]),
    ("lidar__cuboid__buffer_stop", [0.352941176, 1, 0.494117647]),
    ("lidar__cuboid__train", [0.921568627, 0.811764706, 0.211764706]),
    ("lidar__cuboid__road_vehicle", [0.4, 0.419607843, 0.980392157]),
    ("lidar__cuboid__signal_pole", [0.725490196, 0.643137255, 0.329411765]),
    ("lidar__cuboid__animal", [0.780392157, 0.780392157, 0.780392157]),
    ("lidar__cuboid__switch", [0.850980392, 0.541176471, 0.525490196]),
    ("lidar__cuboid__crowd", [0.97647059, 0.43529412, 0.36470588]),
    ("lidar__cuboid__wagons", [0.98431373, 0.94901961, 0.75294118]),
    ("lidar__cuboid__signal_bridge", [0.42745098, 0.27058824, 0.29803922]),
]
OSDAR23_OBJECT_PALETTE = {x[0]: np.asarray(x[1]) * 255 for x in OSDAR23_OBJECT_PALETTE}

MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "road_segment": (31, 120, 180),
    "road_block": (178, 223, 138),
    "lane": (51, 160, 44),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "road_divider": (202, 178, 214),
    "lane_divider": (106, 61, 154),
    "divider": (106, 61, 154),
}


def visualize_camera(
    fpath: str,
    image: np.ndarray,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    gt_bboxes: Optional[LiDARInstance3DBoxes] = None,
    gt_labels: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: float = 4,
    dataset: Optional[str] = None,
) -> None:
    canvas = image.copy()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    if dataset == "TUMTrafIntersectionDataset":
        object_palette = TUMTRAF_OBJECT_PALETTE
    elif dataset == "OSDAR23Dataset":
        object_palette = OSDAR23_OBJECT_PALETTE
    else:
        object_palette = OBJECT_PALETTE

    if bboxes is not None and len(bboxes) > 0:
        corners = bboxes.corners
        num_bboxes = corners.shape[0]

        coords = np.concatenate([corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1)

        if dataset in ["TUMTrafIntersectionDataset", "OSDAR23Dataset"]:
            transform = np.vstack([transform, np.asarray([0, 0, 0, 1])])
        else:
            transform = copy.deepcopy(transform).reshape(4, 4)

        coords = coords @ transform.T
        coords = coords.reshape(-1, 8, 4)

        indices = np.all(coords[..., 2] > 0, axis=1)
        coords = coords[indices]
        labels = labels[indices]

        indices = np.argsort(-np.min(coords[..., 2], axis=1))
        coords = coords[indices]
        labels = labels[indices]

        coords = coords.reshape(-1, 4)
        coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]

        coords = coords[..., :2].reshape(-1, 8, 2)
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            for start, end in [
                (0, 1),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 5),
                (3, 2),
                (3, 7),
                (4, 5),
                (4, 7),
                (2, 6),
                (5, 6),
                (6, 7),
            ]:
                cv2.line(
                    canvas,
                    coords[index, start].astype(np.int),
                    coords[index, end].astype(np.int),
                    color or object_palette[name],
                    thickness,
                    cv2.LINE_AA,
                )
        canvas = canvas.astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)


def visualize_camera_combined(
    fpath: str,
    image: np.ndarray,
    pred_bboxes: Optional[LiDARInstance3DBoxes] = None,
    gt_bboxes: Optional[LiDARInstance3DBoxes] = None,
    transform: Optional[np.ndarray] = None,
    thickness: float = 4,
    dataset: Optional[str] = None,
) -> None:
    canvas = image.copy()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    if (
        pred_bboxes is not None
        and len(pred_bboxes) > 0
        and gt_bboxes is not None
        and len(gt_bboxes) > 0
    ):
        if dataset in ["TUMTrafIntersectionDataset", "OSDAR23Dataset"]:
            transform = np.vstack([transform, np.asarray([0, 0, 0, 1])])
        else:
            transform = copy.deepcopy(transform).reshape(4, 4)

        # fmt: off
        def draw_lines(bboxes, color):
            corners = bboxes.corners
            num_bboxes = corners.shape[0]
            coords = np.concatenate([corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1)
            coords = coords @ transform.T
            coords = coords.reshape(-1, 8, 4)
            indices = np.all(coords[..., 2] > 0, axis=1)
            coords = coords[indices]
            indices = np.argsort(-np.min(coords[..., 2], axis=1))
            coords = coords[indices]
            coords = coords.reshape(-1, 4)
            coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
            coords[:, 0] /= coords[:, 2]
            coords[:, 1] /= coords[:, 2]
            coords = coords[..., :2].reshape(-1, 8, 2)
            for index in range(coords.shape[0]):
                for start, end in [(0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7), (4, 5), (4, 7), (2, 6), (5, 6), (6, 7)]:
                    cv2.line(canvas, coords[index, start].astype(np.int), coords[index, end].astype(np.int), color, thickness, cv2.LINE_AA)
        # fmt: on

        draw_lines(gt_bboxes, (0, 255, 0))
        draw_lines(pred_bboxes, (255, 0, 0))

        canvas = canvas.astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)


def visualize_bev_feature(
    fpath: str,
    bev_feature: np.ndarray,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    dataset: Optional[str] = None,
) -> None:
    if bev_feature.ndim == 4:
        bev_feature = bev_feature[0]
    bev_feature = bev_feature / np.linalg.norm(bev_feature)
    bev_feature = np.sum(bev_feature, axis=0)

    if dataset == "TUMTrafIntersectionDataset":
        bev_feature = np.rot90(bev_feature)
    elif dataset == "OSDAR23Dataset":
        bev_feature = np.rot90(np.rot90(bev_feature))

    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    ax = plt.gca()
    ax.set_aspect(1)
    ax.set_axis_off()

    plt.imshow(bev_feature, cmap="magma")

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        fpath,
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def visualize_lidar(
    fpath: str,
    lidar: Optional[np.ndarray] = None,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    color: Optional[Tuple[int, int, int]] = None,
    radius: float = 15,
    thickness: float = 25,
    dataset: Optional[str] = None,
) -> None:
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    if dataset == "TUMTrafIntersectionDataset":
        object_palette = TUMTRAF_OBJECT_PALETTE
    elif dataset == "OSDAR23Dataset":
        object_palette = OSDAR23_OBJECT_PALETTE
    else:
        object_palette = OBJECT_PALETTE

    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c="white",
        )

    if bboxes is not None and len(bboxes) > 0:
        coords = bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=thickness,
                color=np.array(color or object_palette[name]) / 255,
            )

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        fpath,
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def visualize_lidar_combined(
    fpath: str,
    lidar: Optional[np.ndarray] = None,
    pred_bboxes: Optional[LiDARInstance3DBoxes] = None,
    gt_bboxes: Optional[LiDARInstance3DBoxes] = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    radius: float = 15,
    thickness: float = 15,
) -> None:
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c="white",
        )

    if (
        pred_bboxes is not None
        and len(pred_bboxes) > 0
        and gt_bboxes is not None
        and len(gt_bboxes) > 0
    ):
        pred_coords = pred_bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(pred_coords.shape[0]):
            plt.plot(
                pred_coords[index, :, 0],
                pred_coords[index, :, 1],
                linewidth=thickness,
                color=np.array((255, 0, 0)) / 255,
            )

        gt_coords = gt_bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(gt_coords.shape[0]):
            plt.plot(
                gt_coords[index, :, 0],
                gt_coords[index, :, 1],
                linewidth=thickness,
                color=np.array((0, 255, 0)) / 255,
            )

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        fpath,
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def visualize_map(
    fpath: str,
    masks: np.ndarray,
    classes: List[str],
    background: Tuple[int, int, int] = (240, 240, 240),
) -> None:
    assert masks.dtype == np.bool, masks.dtype

    canvas = np.zeros((*masks.shape[-2:], 3), dtype=np.uint8)
    canvas[:] = background

    for k, name in enumerate(classes):
        if name in MAP_PALETTE:
            canvas[masks[k], :] = MAP_PALETTE[name]
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)
    mmcv.imwrite(canvas, fpath)
