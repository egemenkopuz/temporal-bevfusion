import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import mmcv
import numpy as np
import torch
import torchvision
from mmcv import is_tuple_of
from mmcv.utils import build_from_cfg
from mmdet.datasets.builder import PIPELINES
from numpy import random
from PIL import Image

from mmdet3d.core import VoxelGenerator
from mmdet3d.core.bbox import (
    CameraInstance3DBoxes,
    DepthInstance3DBoxes,
    LiDARInstance3DBoxes,
    box_np_ops,
)
from mmdet3d.core.points import LiDARPoints

from ..builder import OBJECTSAMPLERS
from .utils import box_collision_test, noise_per_object_v3_


@PIPELINES.register_module()
class ImageAug3D:
    def __init__(
        self,
        final_dim,
        resize_lim,
        bot_pct_lim,
        rot_lim,
        rand_flip,
        is_train,
        apply_same_aug_to_seq: bool = False,
    ):
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train
        self.apply_same_aug_to_seq = apply_same_aug_to_seq
        self.sample_aug = None

    def sample_augmentation(self, results):
        W, H = results["ori_shape"]
        fH, fW = self.final_dim
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = np.mean(self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform(self, img, rotation, translation, resize, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor(
            [
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)],
            ]
        )
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b

        return img, rotation, translation

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        imgs = data["img"]
        new_imgs = []
        transforms = []

        if self.apply_same_aug_to_seq:
            if self.sample_aug is None:
                self.sample_aug = [None] * len(imgs)
                for i, img in enumerate(imgs):
                    resize, resize_dims, crop, flip, rotate = self.sample_augmentation(data)
                    self.sample_aug[i] = (resize, resize_dims, crop, flip, rotate)

        for i, img in enumerate(imgs):
            if self.apply_same_aug_to_seq:
                resize, resize_dims, crop, flip, rotate = self.sample_aug[i]
            else:
                resize, resize_dims, crop, flip, rotate = self.sample_augmentation(data)

            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            new_img, rotation, translation = self.img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            new_imgs.append(new_img)
            transforms.append(transform.numpy())
        data["img"] = new_imgs
        # update the calibration matrices
        data["img_aug_matrix"] = transforms
        return data

    def apply_temporal(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for i, frame_data in enumerate(data):
            data[i] = self.apply(frame_data)
        self.sample_aug = None
        return data

    def __call__(
        self, data: Union[dict, List[Dict[str, Any]]]
    ) -> Union[dict, List[Dict[str, Any]]]:
        if isinstance(data, list):  # temporal
            return self.apply_temporal(data)
        else:  # non-temporal
            return self.apply(data)


@PIPELINES.register_module()
class GlobalRotScaleTrans:
    def __init__(
        self,
        resize_lim,
        rot_lim,
        trans_lim,
        is_train,
        apply_same_aug_to_seq: bool = False,
    ):
        self.resize_lim = resize_lim
        self.rot_lim = rot_lim
        self.trans_lim = trans_lim
        self.is_train = is_train
        self.apply_same_aug_to_seq = apply_same_aug_to_seq
        self.sample_aug = None

    def sample_augmentation(self):
        scale = random.uniform(*self.resize_lim)
        theta = random.uniform(*self.rot_lim)
        translation = np.array([random.normal(0, self.trans_lim) for i in range(3)])
        return scale, theta, translation

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        transform = np.eye(4).astype(np.float32)

        if self.is_train:
            if self.apply_same_aug_to_seq:
                if self.sample_aug is None:
                    scale, theta, translation = self.sample_augmentation()
                    self.sample_aug = (scale, theta, translation)
                else:
                    scale, theta, translation = self.sample_aug
            else:
                scale, theta, translation = self.sample_augmentation()

            rotation = np.eye(3)

            if "points" in data:
                data["points"].rotate(-theta)
                data["points"].translate(translation)
                data["points"].scale(scale)

            gt_boxes = data["gt_bboxes_3d"]
            rotation = rotation @ gt_boxes.rotate(theta).numpy()
            gt_boxes.translate(translation)
            gt_boxes.scale(scale)
            data["gt_bboxes_3d"] = gt_boxes

            transform[:3, :3] = rotation.T * scale
            transform[:3, 3] = translation * scale

        data["lidar_aug_matrix"] = transform
        return data

    def apply_temporal(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for i, frame_data in enumerate(data):
            data[i] = self.apply(frame_data)
        self.sample_aug = None
        return data

    def __call__(
        self, data: Union[dict, List[Dict[str, Any]]]
    ) -> Union[dict, List[Dict[str, Any]]]:
        if isinstance(data, list):
            return self.apply_temporal(data)
        else:
            return self.apply(data)


@PIPELINES.register_module()
class GridMask:
    def __init__(
        self,
        use_h,
        use_w,
        max_epoch,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=0,
        prob=1.0,
        fixed_prob=False,
        apply_same_aug_to_seq: bool = False,
    ):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.epoch = None
        self.max_epoch = max_epoch
        self.fixed_prob = fixed_prob
        self.apply_same_aug_to_seq = apply_same_aug_to_seq
        self.sample_aug = None

    def set_epoch(self, epoch):
        self.epoch = epoch
        if not self.fixed_prob:
            self.set_prob(self.epoch, self.max_epoch)

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * self.epoch / self.max_epoch

    def sample_augmentation(self, h, w):
        if np.random.rand() > self.prob:
            return None

        d1 = 2
        d2 = min(h, w)
        d = np.random.randint(d1, d2)

        if self.ratio == 1:
            l = np.random.randint(1, d)
        else:
            l = min(max(int(d * self.ratio + 0.5), 1), d - 1)

        st_h = np.random.randint(d)
        st_w = np.random.randint(d)

        r = np.random.randint(self.rotate)

        offset_r = np.random.rand(h, w)

        return d, l, st_h, st_w, r, offset_r

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        imgs = data["img"]

        h = imgs[0].shape[1]
        w = imgs[0].shape[2]

        if self.apply_same_aug_to_seq:
            if self.sample_aug is None:
                aug_vars = self.sample_augmentation(h, w)
                self.sample_aug = aug_vars
            else:
                aug_vars = self.sample_aug
        else:
            aug_vars = self.sample_augmentation(h, w)

        if aug_vars is None:
            return data

        d, l, st_h, st_w, r, offset_r = aug_vars

        hh = int(1.5 * h)
        ww = int(1.5 * w)
        mask = np.ones((hh, ww), np.float32)

        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + l, ww)
                mask[:, s:t] *= 0

        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2 : (hh - h) // 2 + h, (ww - w) // 2 : (ww - w) // 2 + w]

        mask = mask.astype(np.float32)
        mask = mask[None, :, :]
        if self.mode == 1:
            mask = 1 - mask

        # mask = mask.expand_as(imgs[0])
        if self.offset:
            offset = torch.from_numpy(2 * (offset_r - 0.5)).float()
            offset = (1 - mask) * offset
            imgs = [x * mask + offset for x in imgs]
        else:
            imgs = [x * mask for x in imgs]

        data.update(img=imgs)
        return data

    def apply_temporal(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for i, frame_data in enumerate(data):
            data[i] = self.apply(frame_data)
        self.sample_aug = None
        return data

    def __call__(
        self, data: Union[dict, List[Dict[str, Any]]]
    ) -> Union[dict, List[Dict[str, Any]]]:
        if isinstance(data, list):
            return self.apply_temporal(data)
        else:
            return self.apply(data)


@PIPELINES.register_module()
class RandomFlip3D:
    def __init__(
        self,
        flip_horizontal: bool = True,
        flip_vertical: bool = True,
        is_train: bool = False,
        apply_same_aug_to_seq: bool = False,
    ):
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.is_train = is_train
        self.apply_same_aug_to_seq = apply_same_aug_to_seq
        self.sample_aug = None

    def sample_augmentation(self):
        if self.flip_horizontal:
            flip_horizontal = random.choice([0, 1])
        else:
            flip_horizontal = 0
        if self.flip_vertical:
            flip_vertical = random.choice([0, 1])
        else:
            flip_vertical = 0
        return flip_horizontal, flip_vertical

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.apply_same_aug_to_seq:
            if self.sample_aug is None:
                flip_horizontal, flip_vertical = self.sample_augmentation()
                self.sample_aug = (flip_horizontal, flip_vertical)
            else:
                flip_horizontal, flip_vertical = self.sample_aug
        else:
            flip_horizontal, flip_vertical = self.sample_augmentation()

        rotation = np.eye(3)

        if self.is_train:
            if flip_horizontal:
                rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ rotation
                if "points" in data:
                    data["points"].flip("horizontal")
                if "gt_bboxes_3d" in data:
                    data["gt_bboxes_3d"].flip("horizontal")
                if "gt_masks_bev" in data:
                    data["gt_masks_bev"] = data["gt_masks_bev"][:, :, ::-1].copy()

            if flip_vertical:
                rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ rotation
                if "points" in data:
                    data["points"].flip("vertical")
                if "gt_bboxes_3d" in data:
                    data["gt_bboxes_3d"].flip("vertical")
                if "gt_masks_bev" in data:
                    data["gt_masks_bev"] = data["gt_masks_bev"][:, ::-1, :].copy()

        data["lidar_aug_matrix"][:3, :] = rotation @ data["lidar_aug_matrix"][:3, :]
        return data

    def apply_temporal(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for i, frame_data in enumerate(data):
            data[i] = self.apply(frame_data)
        self.sample_aug = None
        return data

    def __call__(
        self, data: Union[dict, List[Dict[str, Any]]]
    ) -> Union[dict, List[Dict[str, Any]]]:
        if isinstance(data, list):
            return self.apply_temporal(data)
        else:
            return self.apply(data)


@PIPELINES.register_module()
class ObjectPaste:
    """Sample GT objects to the data.
    Args:
        db_sampler (dict): Config dict of the database sampler.
        sample_2d (bool): Whether to also paste 2D image patch to the images
            This should be true when applying multi-modality cut-and-paste.
            Defaults to False.
    """

    def __init__(
        self,
        db_sampler,
        sample_2d=False,
        stop_epoch=None,
        apply_temporal_forward=False,
        apply_collision_check=True,
        apply_same_aug_to_seq: bool = False,
    ):
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        if "type" not in db_sampler.keys():
            db_sampler["type"] = "DataBaseSampler"
        self.db_sampler = build_from_cfg(db_sampler, OBJECTSAMPLERS)
        self.epoch = -1
        self.stop_epoch = stop_epoch
        self.apply_temporal_forward = apply_temporal_forward
        self.apply_collision_check_gt = apply_collision_check
        self.apply_same_aug_to_seq = apply_same_aug_to_seq
        self.sample_aug = None

    def set_epoch(self, epoch):
        self.epoch = epoch

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes.
        Args:
            points (:obj:`BasePoints`): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.
        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    def sample(self, data: Dict[str, Any]):
        # change to float for blending operation
        if self.sample_2d:
            # Assume for now 3D & 2D bboxes are the same
            return self.db_sampler.sample_all(
                data["gt_bboxes_3d"].tensor.numpy(),
                data["gt_bboxes_3d"],
                gt_bboxes_2d=data["gt_bboxes"],
                img=data["img"],
            )
        else:
            return self.db_sampler.sample_all(
                data["gt_bboxes_3d"].tensor.numpy(), data["gt_labels_3d"], img=None
            )

    def apply_rot(
        self, bboxes_3d: np.ndarray, points: LiDARPoints, rot, indices: List[int] = None
    ) -> Tuple[np.ndarray, List[LiDARPoints]]:
        if indices is not None:
            bboxes_3d[indices, 6] += rot[indices]
        else:
            bboxes_3d[:, 6] += rot
        iter_range = range(rot.shape[0]) if indices is None else indices
        for x in iter_range:
            angle = rot[x]
            rot_sin = np.sin(angle)
            rot_cos = np.cos(angle)
            rot_mat_T = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
            # center the points first and then rotate the points, and then move back
            points[x].tensor[:, :3] -= bboxes_3d[x, :3]
            points[x].tensor[:, :3] = points[x].tensor[:, :3] @ rot_mat_T
            points[x].tensor[:, :3] += bboxes_3d[x, :3]
        return bboxes_3d, points

    def apply_trans(
        self, bboxes_3d: np.ndarray, points: LiDARPoints, trans, indices: List[int] = None
    ) -> Tuple[np.ndarray, List[LiDARPoints]]:
        yaws = bboxes_3d[:, 6]
        unit_vector = np.stack([np.cos(yaws), -np.sin(yaws)], axis=1)
        vector = unit_vector * trans.reshape(-1, 1)
        vector = np.concatenate([vector, np.zeros((len(yaws), 1))], axis=1)
        iter_range = range(vector.shape[0]) if indices is None else indices
        for x in iter_range:
            bboxes_3d[x, 0:2] += vector[x, 0:2]
            points[x].tensor[:, :3] += vector[x]
        return bboxes_3d, points

    def check_collision_gt(self, sampled_bboxes_3d, gt_bboxes_3d) -> List[int]:
        sampled = copy.deepcopy(sampled_bboxes_3d)
        num_gt = gt_bboxes_3d.shape[0]
        num_sampled = len(sampled)

        # print(f"num_gt: {num_gt}")
        # print(f"num_sampled: {num_sampled}")

        gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
            gt_bboxes_3d[:, 0:2], gt_bboxes_3d[:, 3:5], gt_bboxes_3d[:, 6]
        )

        boxes = np.concatenate([gt_bboxes_3d, sampled_bboxes_3d], axis=0).copy()

        sp_boxes_new = boxes[gt_bboxes_3d.shape[0] :]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6]
        )

        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
        coll_mat = box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples_indices = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples_indices.append(i - num_gt)
        return valid_samples_indices

    def apply(self, data: Dict[str, Any], temporal_index: Optional[int] = None) -> Dict[str, Any]:
        gt_bboxes_3d = data["gt_bboxes_3d"]
        gt_labels_3d = data["gt_labels_3d"]
        points = data["points"]
        sample_mode = [0] * len(gt_labels_3d)

        if self.apply_same_aug_to_seq:
            if self.sample_aug is None:
                sampled_dict = self.sample(data)
                self.sample_aug = {"sampled_dict": sampled_dict}
            else:
                sampled_dict = self.sample_aug["sampled_dict"]
        else:
            sampled_dict = self.sample(data)

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict["gt_bboxes_3d"]
            sampled_points = sampled_dict["points"]
            sampled_gt_labels = sampled_dict["gt_labels_3d"]
            sample_mode += [1] * len(sampled_gt_labels)

            if temporal_index is not None:
                if self.apply_temporal_forward:
                    sampled_rot = sampled_dict["sampled_rot"]
                    sampled_trans = sampled_dict["sampled_trans"]
                else:
                    sampled_rot = -sampled_dict["sampled_rot"]
                    sampled_trans = -sampled_dict["sampled_trans"]

                if not (
                    temporal_index == 0 and self.apply_same_aug_to_seq
                ):  # not applying to first frame in the queue
                    if sampled_rot is not None:
                        sampled_gt_bboxes_3d, sampled_points = self.apply_rot(
                            sampled_gt_bboxes_3d, sampled_points, sampled_rot
                        )
                    if sampled_trans is not None:
                        sampled_gt_bboxes_3d, sampled_points = self.apply_trans(
                            sampled_gt_bboxes_3d, sampled_points, sampled_trans
                        )

                # check collision with gt
                valid_sample_indices_w_gt = []
                not_valid_sample_indices_w_gt = []
                if self.apply_collision_check_gt:
                    valid_sample_indices_w_gt = self.check_collision_gt(
                        sampled_gt_bboxes_3d, gt_bboxes_3d.tensor.numpy()
                    )
                    not_valid_sample_indices_w_gt = list(
                        set(range(len(sampled_points))) - set(valid_sample_indices_w_gt)
                    )
                    # print(
                    #     f"valid samples: {len(valid_sample_indices_w_gt)} {valid_sample_indices_w_gt}"
                    # )
                    # print(
                    #     f"not valid samples: {len(not_valid_sample_indices_w_gt)} {not_valid_sample_indices_w_gt}"
                    # )
                    # revert invalid samples' rotation and translation
                    if temporal_index != 0 and len(not_valid_sample_indices_w_gt) > 0:
                        if sampled_trans is not None:
                            # print("reverting translation")
                            sampled_gt_bboxes_3d, sampled_points = self.apply_trans(
                                sampled_gt_bboxes_3d,
                                sampled_points,
                                -sampled_trans,
                                not_valid_sample_indices_w_gt,
                            )
                        if sampled_rot is not None:
                            # print("reverting rotation")
                            sampled_gt_bboxes_3d, sampled_points = self.apply_rot(
                                sampled_gt_bboxes_3d,
                                sampled_points,
                                -sampled_rot,
                                not_valid_sample_indices_w_gt,
                            )

            sampled_points = sampled_points[0].cat(sampled_points)
            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels], axis=0)
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate([gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d])
            )

            points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            # check the points dimension
            points = points.cat([sampled_points, points])

            if self.sample_2d:
                sampled_gt_bboxes_2d = sampled_dict["gt_bboxes_2d"]
                gt_bboxes_2d = np.concatenate([gt_bboxes_2d, sampled_gt_bboxes_2d]).astype(
                    np.float32
                )

                data["gt_bboxes"] = gt_bboxes_2d
                data["img"] = sampled_dict["img"]

        data["gt_bboxes_3d"] = gt_bboxes_3d
        data["gt_labels_3d"] = gt_labels_3d.astype(np.long)
        data["sample_mode"] = np.asarray(sample_mode).astype(np.long)
        data["points"] = points

        return data

    def apply_temporal(self, data) -> List[Dict[str, Any]]:
        if self.apply_temporal_forward:
            for i, frame_data in enumerate(data):
                # print(f"applying to frame {frame_data['frame_idx']}")
                data[i] = self.apply(frame_data, i)
        else:  # backwards
            for i, frame_data in enumerate(data[::-1]):
                # print(f"applying to frame {frame_data['frame_idx']}")
                data[i] = self.apply(frame_data, i)
        self.sample_aug = None
        return data

    def __call__(
        self, data: Union[dict, List[Dict[str, Any]]]
    ) -> Union[dict, List[Dict[str, Any]]]:
        """Call function to sample ground truth objects to the data.
        Args:
            data (dict, list): Result dict or list from loading pipeline
        """
        if self.stop_epoch is not None and self.epoch >= self.stop_epoch:
            return data

        if isinstance(data, list):  # temporal
            return self.apply_temporal(data)
        else:  # non-temporal
            return self.apply(data)


@PIPELINES.register_module()
class ObjectNoise:
    """Apply noise to each GT objects in the scene.
    Args:
        translation_std (list[float], optional): Standard deviation of the
            distribution where translation noise are sampled from.
            Defaults to [0.25, 0.25, 0.25].
        global_rot_range (list[float], optional): Global rotation to the scene.
            Defaults to [0.0, 0.0].
        rot_range (list[float], optional): Object rotation range.
            Defaults to [-0.15707963267, 0.15707963267].
        num_try (int, optional): Number of times to try if the noise applied is
            invalid. Defaults to 100.
    """

    def __init__(
        self,
        translation_std=[0.25, 0.25, 0.25],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.15707963267, 0.15707963267],
        num_try=100,
    ):
        self.translation_std = translation_std
        self.global_rot_range = global_rot_range
        self.rot_range = rot_range
        self.num_try = num_try

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        gt_bboxes_3d = data["gt_bboxes_3d"]
        points = data["points"]

        # TODO: check this inplace function
        numpy_box = gt_bboxes_3d.tensor.numpy()
        numpy_points = points.tensor.numpy()

        noise_per_object_v3_(
            numpy_box,
            numpy_points,
            rotation_perturb=self.rot_range,
            center_noise_std=self.translation_std,
            global_random_rot_range=self.global_rot_range,
            num_try=self.num_try,
        )

        data["gt_bboxes_3d"] = gt_bboxes_3d.new_box(numpy_box)
        data["points"] = points.new_point(numpy_points)
        return data

    def apply_temporal(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for i, frame_data in enumerate(data):
            data[i] = self.apply(frame_data)
        return data

    def __call__(
        self, data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Call function to apply noise to each ground truth in the scene.
        Args:
            data (dict, list): Result dict, list from loading pipeline.
        """
        if isinstance(data, list):  # temporal
            return self.apply_temporal(data)
        else:  # non-temporal
            return self.apply(data)


@PIPELINES.register_module()
class FrameDropout:
    def __init__(self, prob: float = 0.5, time_dim: int = -1) -> None:
        self.prob = prob
        self.time_dim = time_dim

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        offsets = []
        for offset in torch.unique(data["points"].tensor[:, self.time_dim]):
            if offset == 0 or random.random() > self.prob:
                offsets.append(offset)
        offsets = torch.tensor(offsets)

        points = data["points"].tensor
        indices = torch.isin(points[:, self.time_dim], offsets)
        data["points"].tensor = points[indices]
        return data

    def apply_temporal(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for i, frame_data in enumerate(data):
            data[i] = self.apply(frame_data)
        return data

    def __call__(
        self, data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        if isinstance(data, list):  # temporal
            return self.apply_temporal(data)
        else:  # non-temporal
            return self.apply(data)


@PIPELINES.register_module()
class PointShuffle:
    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["points"].shuffle()
        return data

    def apply_temporal(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for i, frame_data in enumerate(data):
            data[i] = self.apply(frame_data)
        return data

    def __call__(
        self, data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        if isinstance(data, list):  # temporal
            return self.apply_temporal(data)
        else:  # non-temporal
            return self.apply(data)


@PIPELINES.register_module()
class ObjectRangeFilter:
    """Filter objects by the range.
    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Check points instance type and initialise bev_range
        if isinstance(data["gt_bboxes_3d"], (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        elif isinstance(data["gt_bboxes_3d"], CameraInstance3DBoxes):
            bev_range = self.pcd_range[[0, 2, 3, 5]]
        else:
            raise TypeError(
                "Only support LiDARInstance3DBoxes, DepthInstance3DBoxes and CameraInstance3DBoxes"
            )

        gt_bboxes_3d = data["gt_bboxes_3d"]
        gt_labels_3d = data["gt_labels_3d"]

        mask = gt_bboxes_3d.in_range_bev(bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        if "sample_mode" in data:
            sample_mode = data["sample_mode"][mask]
            data["sample_mode"] = sample_mode

        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(np.bool)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        data["gt_bboxes_3d"] = gt_bboxes_3d
        data["gt_labels_3d"] = gt_labels_3d

        return data

    def apply_temporal(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for i, frame_data in enumerate(data):
            data[i] = self.apply(frame_data)
        return data

    def __call__(
        self, data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Call function to filter objects by the range.
        Args:
            data (dict, list): Result dict from loading pipeline.
        """
        if isinstance(data, list):  # temporal
            return self.apply_temporal(data)
        else:  # non-temporal
            return self.apply(data)

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(point_cloud_range={self.pcd_range.tolist()})"
        return repr_str


@PIPELINES.register_module()
class PointsRangeFilter:
    """Filter points by the range.
    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        points = data["points"]
        points_mask = points.in_range_3d(self.pcd_range)
        clean_points = points[points_mask]
        data["points"] = clean_points
        return data

    def apply_temporal(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for i, frame_data in enumerate(data):
            data[i] = self.apply(frame_data)
        return data

    def __call__(
        self, data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Call function to filter points by the range.
        Args:
            data (dict, list): Result dict from loading pipeline.
        """
        if isinstance(data, list):
            return self.apply_temporal(data)
        else:
            return self.apply(data)


@PIPELINES.register_module()
class ObjectNameFilter:
    """Filter GT objects by their names.
    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        gt_labels_3d = data["gt_labels_3d"]
        gt_bboxes_mask = np.array([n in self.labels for n in gt_labels_3d], dtype=np.bool_)
        data["gt_bboxes_3d"] = data["gt_bboxes_3d"][gt_bboxes_mask]
        data["gt_labels_3d"] = data["gt_labels_3d"][gt_bboxes_mask]
        if "sample_mode" in data:
            sample_mode = data["sample_mode"][gt_bboxes_mask]
            data["sample_mode"] = sample_mode
        return data

    def apply_temporal(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for i, frame_data in enumerate(data):
            data[i] = self.apply(frame_data)
        return data

    def __call__(
        self, data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Call function to filter GT objects by their names.
        Args:
            data (dict, list): Result dict from loading pipeline.
        """
        if isinstance(data, list):  # temporal
            return self.apply_temporal(data)
        else:  # non-temporal
            return self.apply(data)


@PIPELINES.register_module()
class PointSample:
    """Point sample.
    Sampling data to a certain number.
    Args:
        num_points (int): Number of points to be sampled.
        sample_range (float, optional): The range where to sample points.
            If not None, the points with depth larger than `sample_range` are
            prior to be sampled. Defaults to None.
        replace (bool, optional): Whether the sampling is with or without
            replacement. Defaults to False.
    """

    def __init__(self, num_points, sample_range=None, replace=False):
        self.num_points = num_points
        self.sample_range = sample_range
        self.replace = replace

    def _points_random_sampling(
        self,
        points,
        num_samples,
        sample_range=None,
        replace=False,
        return_choices=False,
    ):
        """Points random sampling.
        Sample points to a certain number.
        Args:
            points (np.ndarray | :obj:`BasePoints`): 3D Points.
            num_samples (int): Number of samples to be sampled.
            sample_range (float, optional): Indicating the range where the
                points will be sampled. Defaults to None.
            replace (bool, optional): Sampling with or without replacement.
                Defaults to None.
            return_choices (bool, optional): Whether return choice.
                Defaults to False.
        Returns:
            tuple[np.ndarray] | np.ndarray:
                - points (np.ndarray | :obj:`BasePoints`): 3D Points.
                - choices (np.ndarray, optional): The generated random samples.
        """
        if not replace:
            replace = points.shape[0] < num_samples
        point_range = range(len(points))
        if sample_range is not None and not replace:
            # Only sampling the near points when len(points) >= num_samples
            depth = np.linalg.norm(points.tensor, axis=1)
            far_inds = np.where(depth > sample_range)[0]
            near_inds = np.where(depth <= sample_range)[0]
            # in case there are too many far points
            if len(far_inds) > num_samples:
                far_inds = np.random.choice(far_inds, num_samples, replace=False)
            point_range = near_inds
            num_samples -= len(far_inds)
        choices = np.random.choice(point_range, num_samples, replace=replace)
        if sample_range is not None and not replace:
            choices = np.concatenate((far_inds, choices))
            # Shuffle points after sampling
            np.random.shuffle(choices)
        if return_choices:
            return points[choices], choices
        else:
            return points[choices]

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        points = data["points"]
        # Points in Camera coord can provide the depth information.
        # TODO: Need to suport distance-based sampling for other coord system.
        if self.sample_range is not None:
            from mmdet3d.core.points import CameraPoints

            assert isinstance(
                points, CameraPoints
            ), "Sampling based on distance is only appliable for CAMERA coord"
        points, choices = self._points_random_sampling(
            points,
            self.num_points,
            self.sample_range,
            self.replace,
            return_choices=True,
        )
        data["points"] = points
        return data

    def apply_temporal(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for i, frame_data in enumerate(data):
            data[i] = self.apply(frame_data)
        return data

    def __call__(
        self, data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Call function to sample points to in indoor scenes.
        Args:
            data (dict, list): Result dict from loading pipeline.
        """
        if isinstance(data, list):
            return self.apply_temporal(data)
        else:
            return self.apply(data)

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(num_points={self.num_points},"
        repr_str += f" sample_range={self.sample_range},"
        repr_str += f" replace={self.replace})"

        return repr_str


@PIPELINES.register_module()
class BackgroundPointsFilter:
    """Filter background points near the bounding box.
    Args:
        bbox_enlarge_range (tuple[float], float): Bbox enlarge range.
    """

    def __init__(self, bbox_enlarge_range):
        assert (
            is_tuple_of(bbox_enlarge_range, float) and len(bbox_enlarge_range) == 3
        ) or isinstance(
            bbox_enlarge_range, float
        ), f"Invalid arguments bbox_enlarge_range {bbox_enlarge_range}"

        if isinstance(bbox_enlarge_range, float):
            bbox_enlarge_range = [bbox_enlarge_range] * 3
        self.bbox_enlarge_range = np.array(bbox_enlarge_range, dtype=np.float32)[np.newaxis, :]

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        points = data["points"]
        gt_bboxes_3d = data["gt_bboxes_3d"]

        # avoid groundtruth being modified
        gt_bboxes_3d_np = gt_bboxes_3d.tensor.clone().numpy()
        gt_bboxes_3d_np[:, :3] = gt_bboxes_3d.gravity_center.clone().numpy()

        enlarged_gt_bboxes_3d = gt_bboxes_3d_np.copy()
        enlarged_gt_bboxes_3d[:, 3:6] += self.bbox_enlarge_range
        points_numpy = points.tensor.clone().numpy()
        foreground_masks = box_np_ops.points_in_rbbox(
            points_numpy, gt_bboxes_3d_np, origin=(0.5, 0.5, 0.5)
        )
        enlarge_foreground_masks = box_np_ops.points_in_rbbox(
            points_numpy, enlarged_gt_bboxes_3d, origin=(0.5, 0.5, 0.5)
        )
        foreground_masks = foreground_masks.max(1)
        enlarge_foreground_masks = enlarge_foreground_masks.max(1)
        valid_masks = ~np.logical_and(~foreground_masks, enlarge_foreground_masks)

        data["points"] = points[valid_masks]
        return data

    def apply_temporal(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for i, frame_data in enumerate(data):
            data[i] = self.apply(frame_data)
        return data

    def __call__(
        self, data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Call function to filter points by the range.
        Args:
            data (dict, list): Result dict from loading pipeline.
        """
        if isinstance(data, list):
            return self.apply_temporal(data)
        else:
            return self.apply(data)

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(bbox_enlarge_range={self.bbox_enlarge_range.tolist()})"
        return repr_str


@PIPELINES.register_module()
class VoxelBasedPointSampler:
    """Voxel based point sampler.
    Apply voxel sampling to multiple sweep points.
    Args:
        cur_sweep_cfg (dict): Config for sampling current points.
        prev_sweep_cfg (dict): Config for sampling previous points.
        time_dim (int): Index that indicate the time dimention
            for input points.
    """

    def __init__(self, cur_sweep_cfg, prev_sweep_cfg=None, time_dim=3):
        self.cur_voxel_generator = VoxelGenerator(**cur_sweep_cfg)
        self.cur_voxel_num = self.cur_voxel_generator._max_voxels
        self.time_dim = time_dim
        if prev_sweep_cfg is not None:
            assert prev_sweep_cfg["max_num_points"] == cur_sweep_cfg["max_num_points"]
            self.prev_voxel_generator = VoxelGenerator(**prev_sweep_cfg)
            self.prev_voxel_num = self.prev_voxel_generator._max_voxels
        else:
            self.prev_voxel_generator = None
            self.prev_voxel_num = 0

    def _sample_points(self, points, sampler, point_dim):
        """Sample points for each points subset.
        Args:
            points (np.ndarray): Points subset to be sampled.
            sampler (VoxelGenerator): Voxel based sampler for
                each points subset.
            point_dim (int): The dimention of each points
        Returns:
            np.ndarray: Sampled points.
        """
        voxels, coors, num_points_per_voxel = sampler.generate(points)
        if voxels.shape[0] < sampler._max_voxels:
            padding_points = np.zeros(
                [
                    sampler._max_voxels - voxels.shape[0],
                    sampler._max_num_points,
                    point_dim,
                ],
                dtype=points.dtype,
            )
            padding_points[:] = voxels[0]
            sample_points = np.concatenate([voxels, padding_points], axis=0)
        else:
            sample_points = voxels

        return sample_points

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        points = data["points"]
        original_dim = points.shape[1]

        # TODO: process instance and semantic mask while _max_num_points
        # is larger than 1
        # Extend points with seg and mask fields
        map_fields2dim = []
        start_dim = original_dim
        points_numpy = points.tensor.numpy()
        extra_channel = [points_numpy]
        for idx, key in enumerate(data["pts_mask_fields"]):
            map_fields2dim.append((key, idx + start_dim))
            extra_channel.append(data[key][..., None])

        start_dim += len(data["pts_mask_fields"])
        for idx, key in enumerate(data["pts_seg_fields"]):
            map_fields2dim.append((key, idx + start_dim))
            extra_channel.append(data[key][..., None])

        points_numpy = np.concatenate(extra_channel, axis=-1)

        # Split points into two part, current sweep points and
        # previous sweeps points.
        # TODO: support different sampling methods for next sweeps points
        # and previous sweeps points.
        cur_points_flag = points_numpy[:, self.time_dim] == 0
        cur_sweep_points = points_numpy[cur_points_flag]
        prev_sweeps_points = points_numpy[~cur_points_flag]
        if prev_sweeps_points.shape[0] == 0:
            prev_sweeps_points = cur_sweep_points

        # Shuffle points before sampling
        np.random.shuffle(cur_sweep_points)
        np.random.shuffle(prev_sweeps_points)

        cur_sweep_points = self._sample_points(
            cur_sweep_points, self.cur_voxel_generator, points_numpy.shape[1]
        )
        if self.prev_voxel_generator is not None:
            prev_sweeps_points = self._sample_points(
                prev_sweeps_points, self.prev_voxel_generator, points_numpy.shape[1]
            )

            points_numpy = np.concatenate([cur_sweep_points, prev_sweeps_points], 0)
        else:
            points_numpy = cur_sweep_points

        if self.cur_voxel_generator._max_num_points == 1:
            points_numpy = points_numpy.squeeze(1)
        data["points"] = points.new_point(points_numpy[..., :original_dim])

        # Restore the correspoinding seg and mask fields
        for key, dim_index in map_fields2dim:
            data[key] = points_numpy[..., dim_index]

        return data

    def apply_temporal(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for i, frame_data in enumerate(data):
            data[i] = self.apply(frame_data)
        return data

    def __call__(
        self, data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Call function to sample points from multiple sweeps.
        Args:
            data (dict, list): Result dict from loading pipeline.
        """
        if isinstance(data, list):
            return self.apply_temporal(data)
        else:
            return self.apply(data)

    def __repr__(self):
        """str: Return a string that describes the module."""

        def _auto_indent(repr_str, indent):
            repr_str = repr_str.split("\n")
            repr_str = [" " * indent + t + "\n" for t in repr_str]
            repr_str = "".join(repr_str)[:-1]
            return repr_str

        repr_str = self.__class__.__name__
        indent = 4
        repr_str += "(\n"
        repr_str += " " * indent + f"num_cur_sweep={self.cur_voxel_num},\n"
        repr_str += " " * indent + f"num_prev_sweep={self.prev_voxel_num},\n"
        repr_str += " " * indent + f"time_dim={self.time_dim},\n"
        repr_str += " " * indent + "cur_voxel_generator=\n"
        repr_str += f"{_auto_indent(repr(self.cur_voxel_generator), 8)},\n"
        repr_str += " " * indent + "prev_voxel_generator=\n"
        repr_str += f"{_auto_indent(repr(self.prev_voxel_generator), 8)})"
        return repr_str


@PIPELINES.register_module()
class ImagePad:
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [
                mmcv.impad(img, shape=self.size, pad_val=self.pad_val) for img in results["img"]
            ]
        elif self.size_divisor is not None:
            padded_img = [
                mmcv.impad_to_multiple(img, self.size_divisor, pad_val=self.pad_val)
                for img in results["img"]
            ]
        results["img"] = padded_img
        results["img_shape"] = [img.shape for img in padded_img]
        results["pad_shape"] = [img.shape for img in padded_img]
        results["pad_fixed_size"] = self.size
        results["pad_size_divisor"] = self.size_divisor

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self._pad_img(data)
        return data

    def apply_temporal(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for i, frame_data in enumerate(data):
            data[i] = self.apply(frame_data)
        return data

    def __call__(
        self, data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            data (dict, list): Result dict from loading pipeline.
        """
        if isinstance(data, list):
            return self.apply_temporal(data)
        else:
            return self.apply(data)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.size}, "
        repr_str += f"size_divisor={self.size_divisor}, "
        repr_str += f"pad_val={self.pad_val})"
        return repr_str


@PIPELINES.register_module()
class ImageNormalize:
    def __init__(self, mean, std, skip_normalize=False):
        self.mean = mean
        self.std = std
        if skip_normalize:
            self.compose = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        else:
            self.compose = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=mean, std=std),
                ]
            )

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["img"] = [self.compose(img) for img in data["img"]]
        data["img_norm_cfg"] = dict(mean=self.mean, std=self.std)
        return data

    def apply_temporal(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for i, frame_data in enumerate(data):
            data[i] = self.apply(frame_data)
        return data

    def __call__(
        self, data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        if isinstance(data, list):
            return self.apply_temporal(data)
        else:
            return self.apply(data)


@PIPELINES.register_module()
class ImageDistort:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        imgs = data["img"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the input image of dtype np.float32,"
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            )
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta, self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower, self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        data["img"] = new_imgs
        return data

    def apply_temporal(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for i, frame_data in enumerate(data):
            data[i] = self.apply(frame_data)
        return data

    def __call__(
        self, data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        if isinstance(data, list):
            return self.apply_temporal(data)
        else:
            return self.apply(data)
