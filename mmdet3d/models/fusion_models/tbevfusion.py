import os
from itertools import chain
from typing import Any, Dict, List, Optional

import torch
from mmcv.runner import auto_fp16

from mmdet3d.core.utils.visualize import visualize_bev_feature
from mmdet3d.models import FUSIONMODELS
from mmdet3d.models.builder import build_fuser

from .bevfusion import BEVFusion

__all__ = ["TBEVFusion"]


class Queue:
    def __init__(self):
        self._elements = []
        self._frame_indices = []  # for debugging

    def enqueue(self, element, frame_index: Optional[int] = None) -> None:
        self._elements.append(element)
        self._frame_indices.append(frame_index)

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
        temporal_fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        save_bev_features: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(encoders, fuser, decoder, heads, save_bev_features, **kwargs)

        assert temporal_fuser is not None

        self.temporal_fuser = build_fuser(temporal_fuser)
        self.cache_queue = Queue()
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
        assert len(img.shape) == 6, img.shape
        online = True if not self.training else False

        len_queue = img.size(1)

        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_points = [[x[:][i] for x in points] for i in range(len_queue - 1)]
        points = [x[:][-1] for x in points]

        prev_metas = [[x[i] for x in metas] for i in range(len_queue - 1)]
        metas = [x[-1] for x in metas]

        if online:
            if self.cache_queue.is_empty():  # only at the first frame of a sequence
                prev_features = self.get_bev_history(
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
                for i, x in enumerate(prev_features):  # add (queue_length - 1) features
                    self.cache_queue.enqueue(x, prev_metas[i][0]["frame_idx"])
            else:
                prev_features = [self.cache_queue.dequeue()]
                for rem in range(len_queue - 2):
                    prev_features.append(self.cache_queue.get(rem))
        else:
            prev_features = self.get_bev_history(
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
            features.append(feature)

            if self.save_bev_features is not None and "out_dir" in self.save_bev_features:
                # assuming batch size = 1
                visualize_bev_feature(
                    os.path.join(
                        self.save_bev_features["out_dir"],
                        f"bev-feat-{sensor}",
                        f"{metas['timestamp']}.png",
                    ),
                    feature.clone().detach().cpu().numpy().squeeze(),
                    self.save_bev_features["xlim"],
                    self.save_bev_features["ylim"],
                    True,
                )

        if not self.training:
            # avoid OOM
            features = features[::-1]

        if self.fuser is not None:
            features = self.fuser(features)

        elif len(features) == 1:
            features = features[0]

        if self.save_bev_features is not None and "out_dir" in self.save_bev_features:
            visualize_bev_feature(
                os.path.join(
                    self.save_bev_features["out_dir"],
                    "bev-feat-fused",
                    f"{metas[0]['timestamp']}.png",
                ),
                features.clone().detach().cpu().numpy().squeeze(),
                self.save_bev_features["xlim"],
                self.save_bev_features["ylim"],
                True,
            )
        # else:
        #     visualize_bev_feature(
        #         os.path.join(
        #             "test_visuals",
        #             "bev-feat-fused",
        #             f"{metas[0]['timestamp']}.png",
        #         ),
        #         features.clone().detach().cpu().numpy().squeeze(),
        #         [-20, 140],
        #         [-80, 80],
        #         True,
        #     )

        if online:
            self.cache_queue.enqueue(features, metas[0]["frame_idx"])

        if len(prev_features) != 0:
            features = [features] + [x for x in prev_features]
            x = self.temporal_fuser(features)
        else:
            x = features[0]

        if self.save_bev_features is not None and "out_dir" in self.save_bev_features:
            visualize_bev_feature(
                os.path.join(
                    self.save_bev_features["out_dir"],
                    "bev-feat-temporal-fused",
                    f"{metas[0]['timestamp']}.png",
                ),
                x.clone().detach().cpu().numpy().squeeze(),
                self.save_bev_features["xlim"],
                self.save_bev_features["ylim"],
                True,
            )
        # else:
        #     visualize_bev_feature(
        #         os.path.join(
        #             "test_visuals",
        #             "bev-feat-temporal-fused",
        #             f"{metas[0]['timestamp']}.png",
        #         ),
        #         x.clone().detach().cpu().numpy().squeeze(),
        #         [-20, 140],
        #         [-80, 80],
        #         True,
        #     )

        # if online:
        #     global _frame_indices_iter
        #     print(f"{_frame_indices_iter} frame_indices", self.cache_queue._frame_indices)
        #     _frame_indices_iter += 1

        # delete cache if the current frame is the last frame in the sequence
        if online and not metas[0]["next_bev_exists"]:
            self.cache_queue.clear()

        batch_size = x.shape[0]

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
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
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs

    def get_bev_history(
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

                    if self.save_bev_features is not None and "out_dir" in self.save_bev_features:
                        # assuming batch size = 1
                        visualize_bev_feature(
                            os.path.join(
                                self.save_bev_features["out_dir"],
                                f"bev-feat-prev{i}-{sensor}",
                                f"{prev_metas[i][0]['timestamp']}.png",
                            ),
                            prev_feature.clone().detach().cpu().numpy().squeeze(),
                            self.save_bev_features["xlim"],
                            self.save_bev_features["ylim"],
                            True,
                        )
                    # else:
                    #     visualize_bev_feature(
                    #         os.path.join(
                    #             "test_visuals",
                    #             f"bev-feat-prev{i}-{sensor}",
                    #             f"{prev_metas[i][0]['timestamp']}.png",
                    #         ),
                    #         prev_feature.clone().detach().cpu().numpy().squeeze(),
                    #         [-20, 140],
                    #         [-80, 80],
                    #         True,
                    #     )

                    prev_features.append(prev_feature.detach())

                if not self.training:
                    # avoid OOM
                    prev_features = prev_features[::-1]

                if self.fuser is not None:
                    x = self.fuser(prev_features).detach()
                    prev_all_features.append(x)
                elif len(prev_features) == 1:
                    prev_all_features.append(prev_features[0])
                else:
                    prev_all_features.append(prev_features)

        if is_train:
            self.train()

        return prev_all_features
