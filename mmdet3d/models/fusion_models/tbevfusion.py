import os
from typing import Any, Dict, List, Optional

import torch
from mmcv.runner import auto_fp16

from mmdet3d.core.utils.visualize import visualize_bev_feature
from mmdet3d.models import FUSIONMODELS
from mmdet3d.models.builder import build_temporal

from .bevfusion import BEVFusion

__all__ = ["TBEVFusion"]


class Queue:
    def __init__(self, max_length: int = 1):
        self._elements = []
        self._max_length = max_length
        self._frame_indices = []  # for debugging

    def enqueue(self, element, frame_index: Optional[int] = None) -> None:
        self._elements.append(element)
        self._frame_indices.append(frame_index)
        if len(self._elements) > self._max_length:
            self._elements.pop(0)
            self._frame_indices.pop(0)

    def dequeue(self) -> Any:
        self._frame_indices.pop(0)
        return self._elements.pop(0)

    def is_empty(self) -> bool:
        return len(self._elements) == 0

    def clear(self) -> None:
        self._elements = []
        self._frame_indices = []

    def get(self, index: int) -> Any:
        return self._elements[index]

    def get_idx(self, index: int) -> int:
        return self._frame_indices[index]

    def __len__(self) -> int:
        return len(self._elements)


@FUSIONMODELS.register_module()
class TBEVFusion(BEVFusion):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        temporal: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        max_queue_length: int = 3,
        save_bev_features: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(encoders, fuser, decoder, heads, save_bev_features, **kwargs)

        assert temporal is not None
        self.temporal_fuser = build_temporal(temporal)
        self.max_queue_length = max_queue_length
        self.cache_queue = Queue(max_queue_length)
        global _frame_indices_iter
        _frame_indices_iter = 0

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        # fmt: off
        if len(img.shape) == 6:  # B, T, I, C, H, W
            return self._forward_single(img, points, camera2ego, lidar2ego, lidar2camera, lidar2image, camera_intrinsics, camera2lidar, img_aug_matrix, lidar_aug_matrix, metas, gt_masks_bev, gt_bboxes_3d, gt_labels_3d)
        else:  # B, I, C, H, W
            return self._forward_single_online(img, points, camera2ego, lidar2ego, lidar2camera, lidar2image, camera_intrinsics, camera2lidar, img_aug_matrix, lidar_aug_matrix, metas, gt_masks_bev, gt_bboxes_3d, gt_labels_3d)
        # fmt: on

    def _forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
    ):
        len_queue = img.size(1)

        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_points = [[x[:][i] for x in points] for i in range(len_queue - 1)]
        points = [x[:][-1] for x in points]

        prev_metas = [[x[i] for x in metas] for i in range(len_queue - 1)]
        metas = [x[-1] for x in metas]

        gt_bboxes_3d = [x[:][-1] for x in gt_bboxes_3d]
        gt_labels_3d = [x[:][-1] for x in gt_labels_3d]

        prev_features = self._get_bev_history(
            prev_img,
            prev_points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            prev_metas,
        )

        features = []
        for sensor in self.encoders if self.training else list(self.encoders.keys())[::-1]:
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                )
            elif sensor == "lidar":
                feature = self.extract_lidar_features(points)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")

            # basename = metas[0]["lidar_path"].split("/")[-1][:-4]
            # self._save_bev_feat(feature, f"feat-bev-{sensor}", basename)

            features.append(feature)

        if not self.training:
            # avoid OOM
            features = features[::-1]

        if self.fuser is not None:
            features = self.fuser(features)

        elif len(features) == 1:
            features = features[0]

        if isinstance(prev_features[0], list):
            raise NotImplementedError(
                "prev_features' item is a list, but temporal_fuser is not None; therefore, prior fusing must be done"
            )
        else:
            # prev_features: -> (batch_size, queue_length - 1, channels, height, width)
            prev_features = torch.stack(prev_features, dim=1)
            # features: -> (batch_size, 1, channels, height, width)
            features = torch.unsqueeze(features, dim=1)
            # cat along the first dimension
            x = torch.cat([prev_features, features], dim=1)
            x = self.temporal_fuser(x)

        batch_size = x.shape[0]

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        if self.training:
            outputs = {}
            for h_type, head in self.heads.items():
                if h_type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif h_type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {h_type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{h_type}/{name}"] = val * self.loss_scale[h_type]
                    else:
                        outputs[f"stats/{h_type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for h_type, head in self.heads.items():
                if h_type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif h_type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {h_type}")
            return outputs

    def _forward_single_online(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        features = []
        for sensor in self.encoders if self.training else list(self.encoders.keys())[::-1]:
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                )
            elif sensor == "lidar":
                feature = self.extract_lidar_features(points)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")

            if self.save_bev_features is not None and "out_dir" in self.save_bev_features:
                # assuming batch size = 1
                basename = metas[0]["lidar_path"].split("/")[-1][:-4]
                self._save_bev_feat(feature, f"feat-bev-{sensor}", basename)

            features.append(feature)

        if not self.training:
            # avoid OOM
            features = features[::-1]

        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        if len(features) > 1:
            if self.save_bev_features is not None and "out_dir" in self.save_bev_features:
                # assuming batch size = 1
                basename = metas[0]["lidar_path"].split("/")[-1][:-4]
                self._save_bev_feat(x, "feat-bev-fused", basename)

        if len(x.shape) == 4:  # -> (batch_size, 1, channels, height, width)
            x = torch.unsqueeze(x, dim=1)

        self.cache_queue.enqueue(x, metas[0]["frame_idx"])

        x = [x]
        for i in range(len(self.cache_queue) - 1):
            x.insert(0, self.cache_queue.get(i))

        if len(x) > 1:
            x = torch.cat(x, dim=1)  # (batch_size, [STACK], channels, height, width)
        else:
            x = x[0]

        x = self.temporal_fuser(x)

        if self.save_bev_features is not None and "out_dir" in self.save_bev_features:
            # assuming batch size = 1
            basename = metas[0]["lidar_path"].split("/")[-1][:-4]
            self._save_bev_feat(x, "feat-bev-temporal-fused", basename)

        # global _frame_indices_iter
        # print(f"{_frame_indices_iter} frame_indices", self.cache_queue._frame_indices)
        # _frame_indices_iter += 1

        # delete cache if the current frame is the last frame in the sequence
        if not metas[0]["next"]:
            self.cache_queue.clear()

        batch_size = x.shape[0]

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        if self.training:
            outputs = {}
            for h_type, head in self.heads.items():
                if h_type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif h_type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {h_type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{h_type}/{name}"] = val * self.loss_scale[h_type]
                    else:
                        outputs[f"stats/{h_type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for h_type, head in self.heads.items():
                if h_type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif h_type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {h_type}")
            return outputs

    def _get_bev_history(
        self,
        prev_img,
        prev_points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        prev_metas,
    ) -> List[List[torch.Tensor]]:
        """
        Get BEV history features from previous frames.
        """
        prev_all_features = []
        is_train = self.training

        if is_train:
            self.eval()

        with torch.no_grad():
            for i in range(prev_img.shape[1]):
                prev_features = []

                for sensor in self.encoders if self.training else list(self.encoders.keys())[::-1]:
                    if sensor == "camera":
                        prev_feature = self.extract_camera_features(
                            prev_img[:, i, ...],
                            prev_points[i],
                            camera2ego,
                            lidar2ego,
                            lidar2camera,
                            lidar2image,
                            camera_intrinsics,
                            camera2lidar,
                            img_aug_matrix,
                            lidar_aug_matrix,
                            prev_metas[i],
                        )
                    elif sensor == "lidar":
                        prev_feature = self.extract_lidar_features(prev_points[i])
                    else:
                        raise ValueError(f"unsupported sensor: {sensor}")

                    # basename = prev_metas[i][0]["lidar_path"].split("/")[-1][:-4]
                    # self._save_bev_feat(prev_feature, f"feat-bev-prev{i}-{sensor}", basename)

                    prev_features.append(prev_feature.detach())

                if not self.training:
                    # avoid OOM
                    prev_features = prev_features[::-1]

                if self.fuser is not None:  # fuse camera and lidar features
                    x = self.fuser(prev_features).detach()
                    prev_all_features.append(x)
                elif len(prev_features) == 1:  # there is only one sensor, either camera or lidar
                    prev_all_features.append(prev_features[0])
                else:  # no fusing, a list of features
                    prev_all_features.append(prev_features)

        if is_train:
            self.train()

        return prev_all_features
