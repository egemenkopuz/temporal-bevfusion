# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Union

import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor

from mmdet3d.core.bbox import BaseInstance3DBoxes
from mmdet3d.core.points import BasePoints


@PIPELINES.register_module()
class DefaultFormatBundle3D:
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __init__(
        self,
        classes,
        with_gt: bool = True,
        with_label: bool = True,
    ) -> None:
        super().__init__()
        self.class_names = classes
        self.with_gt = with_gt
        self.with_label = with_label

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Format 3D data
        if "points" in data:
            assert isinstance(data["points"], BasePoints)
            data["points"] = DC(data["points"].tensor)

        for key in ["voxels", "coors", "voxel_centers", "num_points"]:
            if key not in data:
                continue
            data[key] = DC(to_tensor(data[key]), stack=False)

        if self.with_gt:
            # Clean GT bboxes in the final
            if "gt_bboxes_3d_mask" in data:
                gt_bboxes_3d_mask = data["gt_bboxes_3d_mask"]
                data["gt_bboxes_3d"] = data["gt_bboxes_3d"][gt_bboxes_3d_mask]
                if "gt_names_3d" in data:
                    data["gt_names_3d"] = data["gt_names_3d"][gt_bboxes_3d_mask]
                if "centers2d" in data:
                    data["centers2d"] = data["centers2d"][gt_bboxes_3d_mask]
                if "depths" in data:
                    data["depths"] = data["depths"][gt_bboxes_3d_mask]
            if "gt_bboxes_mask" in data:
                gt_bboxes_mask = data["gt_bboxes_mask"]
                if "gt_bboxes" in data:
                    data["gt_bboxes"] = data["gt_bboxes"][gt_bboxes_mask]
                data["gt_names"] = data["gt_names"][gt_bboxes_mask]
            if self.with_label:
                if "gt_names" in data and len(data["gt_names"]) == 0:
                    data["gt_labels"] = np.array([], dtype=np.int64)
                    data["attr_labels"] = np.array([], dtype=np.int64)
                elif "gt_names" in data and isinstance(data["gt_names"][0], list):
                    # gt_labels might be a list of list in multi-view setting
                    data["gt_labels"] = [
                        np.array([self.class_names.index(n) for n in res], dtype=np.int64)
                        for res in data["gt_names"]
                    ]
                elif "gt_names" in data:
                    data["gt_labels"] = np.array(
                        [self.class_names.index(n) for n in data["gt_names"]],
                        dtype=np.int64,
                    )
                # we still assume one pipeline for one frame LiDAR
                # thus, the 3D name is list[string]
                if "gt_names_3d" in data:
                    data["gt_labels_3d"] = np.array(
                        [self.class_names.index(n) for n in data["gt_names_3d"]],
                        dtype=np.int64,
                    )
        if "img" in data:
            data["img"] = DC(torch.stack(data["img"]), stack=True)

        for key in [
            "proposals",
            "gt_bboxes",
            "gt_bboxes_ignore",
            "gt_labels",
            "gt_labels_3d",
            "attr_labels",
            "centers2d",
            "depths",
        ]:
            if key not in data:
                continue
            if isinstance(data[key], list):
                data[key] = DC([to_tensor(res) for res in data[key]])
            else:
                data[key] = DC(to_tensor(data[key]))
        if "gt_bboxes_3d" in data:
            if isinstance(data["gt_bboxes_3d"], BaseInstance3DBoxes):
                data["gt_bboxes_3d"] = DC(data["gt_bboxes_3d"], cpu_only=True)
            else:
                data["gt_bboxes_3d"] = DC(to_tensor(data["gt_bboxes_3d"]))
        return data

    def apply_temporal(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for i, frame_data in enumerate(data):
            data[i] = self.apply(frame_data)
        return data

    def __call__(
        self, data: Union[dict, List[Dict[str, Any]]]
    ) -> Union[dict, List[Dict[str, Any]]]:
        """Call function to transform and format common fields in results.

        Args:
            data (dict, list): Result dict contains the data to convert.
        """
        if isinstance(data, list):
            return self.apply_temporal(data)
        else:
            return self.apply(data)


@PIPELINES.register_module()
class Collect3D:
    def __init__(
        self,
        keys,
        meta_keys=(
            "camera_intrinsics",
            "camera2ego",
            "img_aug_matrix",
            "lidar_aug_matrix",
        ),
        meta_lis_keys=(
            "filename",
            "timestamp",
            "ori_shape",
            "img_shape",
            "lidar2image",
            "depth2img",
            "cam2img",
            "pad_shape",
            "scale_factor",
            "flip",
            "pcd_horizontal_flip",
            "pcd_vertical_flip",
            "box_mode_3d",
            "box_type_3d",
            "img_norm_cfg",
            "pcd_trans",
            "token",
            "scene_token",  # for temporal
            "prev",  # for temporal
            "next",  # for temporal
            "frame_idx",  # for temporal
            "pcd_scale_factor",
            "pcd_rotation",
            "lidar_path",
            "transformation_3d_flow",
            "sample_mode",
        ),
    ):
        self.keys = keys
        self.meta_keys = meta_keys
        # [fixme] note: need at least 1 meta lis key to perform training.
        self.meta_lis_keys = meta_lis_keys

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for key in self.keys:
            if key not in self.meta_keys:
                out[key] = data[key]
        for key in self.meta_keys:
            if key in data:
                val = np.array(data[key])
                if isinstance(data[key], list):
                    out[key] = DC(to_tensor(val), stack=True)
                else:
                    out[key] = DC(to_tensor(val), stack=True, pad_dims=1)

        metas = {}
        for key in self.meta_lis_keys:
            if key in data:
                metas[key] = data[key]

        out["metas"] = DC(metas, cpu_only=True)
        return out

    def apply_temporal(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for i, frame_data in enumerate(data):
            data[i] = self.apply(frame_data)
        return data

    def __call__(
        self, data: Union[dict, List[Dict[str, Any]]]
    ) -> Union[dict, List[Dict[str, Any]]]:
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.

        Args:
            data (dict, list): Result dict contains the data to collect.
        """
        if isinstance(data, list):
            return self.apply_temporal(data)
        else:
            return self.apply(data)
