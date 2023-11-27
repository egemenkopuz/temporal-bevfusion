import copy
import os
from typing import Dict, List, Optional

import mmcv
import numpy as np
from mmdet.datasets import PIPELINES

from mmdet3d.core.bbox import box_np_ops
from mmdet3d.core.points.lidar_points import LiDARPoints

from ..builder import OBJECTSAMPLERS
from .utils import box_collision_test


class BatchSampler:
    """Class for sampling specific category of ground truths.

    Args:
        sample_list (list[dict]): List of samples.
        name (str | None): The category of samples. Default: None.
        epoch (int | None): Sampling epoch. Default: None.
        shuffle (bool): Whether to shuffle indices. Default: False.
        drop_reminder (bool): Drop reminder. Default: False.
    """

    def __init__(
        self,
        sampled_list,
        name=None,
        epoch=None,
        shuffle=True,
        drop_reminder=False,
    ):
        self._sampled_list = sampled_list
        self._indices = np.arange(len(sampled_list))
        if shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0
        self._example_num = len(sampled_list)
        self._name = name
        self._shuffle = shuffle
        self._epoch = epoch
        self._epoch_counter = 0
        self._drop_reminder = drop_reminder

    def _sample(self, num):
        """Sample specific number of ground truths and return indices.

        Args:
            num (int): Sampled number.

        Returns:
            list[int]: Indices of sampled ground truths.
        """
        if self._idx + num >= self._example_num:
            ret = self._indices[self._idx :].copy()
            self._reset()
        else:
            ret = self._indices[self._idx : self._idx + num]
            self._idx += num
        return ret

    def _reset(self):
        """Reset the index of batchsampler to zero."""
        assert self._name is not None
        # print("reset", self._name)
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0

    def sample(self, num):
        """Sample specific number of ground truths.

        Args:
            num (int): Sampled number.

        Returns:
            list[dict]: Sampled ground truths.
        """
        indices = self._sample(num)
        return [self._sampled_list[i] for i in indices]


@OBJECTSAMPLERS.register_module()
class DataBaseSampler:
    """Class for sampling data from the ground truth database.

    Args:
        info_path (str): Path of groundtruth database info.
        dataset_root (str): Path of groundtruth database.
        rate (float): Rate of actual sampled over maximum sampled number.
        prepare (dict): Name of preparation functions and the input value.
        sample_groups (dict): Sampled classes and numbers.
        classes (list[str]): List of classes. Default: None.
        points_loader(dict): Config of points loader. Default: dict(
            type='LoadPointsFromFile', load_dim=4, use_dim=[0,1,2,3])
    """

    def __init__(
        self,
        info_path,
        dataset_root,
        rate,
        prepare,
        sample_groups,
        classes=None,
        points_loader=dict(
            type="LoadPointsFromFile",
            coord_type="LIDAR",
            load_dim=4,
            use_dim=[0, 1, 2, 3],
        ),
        reduce_points_by_distance: Optional[Dict[str, float]] = None,
        cls_trans_lim: Optional[Dict[str, float]] = None,
        cls_rot_lim: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.info_path = info_path
        self.rate = rate
        self.prepare = prepare
        self.classes = classes
        self.cat2label = {name: i for i, name in enumerate(classes)}
        self.label2cat = {i: name for i, name in enumerate(classes)}
        self.points_loader = mmcv.build_from_cfg(points_loader, PIPELINES)
        self.reduce_points_by_distance = reduce_points_by_distance

        self.cls_trans_lim = cls_trans_lim
        self.cls_rot_lim = cls_rot_lim

        self._trans_lambda_normal = lambda mean, std: abs(np.random.normal(mean, std))
        self._trans_lambda_uniform = lambda min, max: abs(np.random.uniform(min, max))
        self._rot_lambda_normal = lambda mean, std: np.random.normal(mean, std)
        self._rot_lambda_uniform = lambda min, max: np.random.uniform(min, max)

        # convert cls_trans_lim into labelled format
        if self.cls_trans_lim is not None:
            self.cls_trans_lim = {self.cat2label[x]: y for x, y in self.cls_trans_lim.items()}
            self.cls_trans_fn = {
                x: self._trans_lambda_normal
                if y[0] in ["normal", "gaussian"]
                else self._trans_lambda_uniform
                for x, y in self.cls_trans_lim.items()
            }
        if self.cls_rot_lim is not None:
            self.cls_rot_lim = {self.cat2label[x]: y for x, y in self.cls_rot_lim.items()}
            self.cls_rot_fn = {
                x: self._rot_lambda_normal
                if y[0] in ["normal", "gaussian"]
                else self._rot_lambda_uniform
                for x, y in self.cls_rot_lim.items()
            }

        db_infos = mmcv.load(info_path)

        # filter database infos
        from mmdet3d.utils import get_root_logger

        logger = get_root_logger()
        for k, v in db_infos.items():
            logger.debug(f"load {len(v)} {k} database infos")
        for prep_func, val in prepare.items():
            db_infos = getattr(self, prep_func)(db_infos, val)
        logger.debug("After filter database:")
        for k, v in db_infos.items():
            logger.debug(f"load {len(v)} {k} database infos")

        self.db_infos = db_infos

        # load sample groups
        # TODO: more elegant way to load sample groups
        self.sample_groups = []
        for name, num in sample_groups.items():
            self.sample_groups.append({name: int(num)})

        self.group_db_infos = self.db_infos  # just use db_infos
        self.sample_classes = []
        self.sample_max_nums = []
        for group_info in self.sample_groups:
            self.sample_classes += list(group_info.keys())
            self.sample_max_nums += list(group_info.values())

        self.sampler_dict = {}
        for k, v in self.group_db_infos.items():
            self.sampler_dict[k] = BatchSampler(v, k, shuffle=True)
        # TODO: No group_sampling currently

    @staticmethod
    def filter_by_difficulty(db_infos, removed_difficulty):
        """Filter ground truths by difficulties.

        Args:
            db_infos (dict): Info of groundtruth database.
            removed_difficulty (list): Difficulties that are not qualified.

        Returns:
            dict: Info of database after filtering.
        """
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            new_db_infos[key] = [
                info for info in dinfos if info["difficulty"] not in removed_difficulty
            ]
        return new_db_infos

    @staticmethod
    def filter_by_min_points(db_infos, min_gt_points_dict):
        """Filter ground truths by number of points in the bbox.

        Args:
            db_infos (dict): Info of groundtruth database.
            min_gt_points_dict (dict): Different number of minimum points
                needed for different categories of ground truths.

        Returns:
            dict: Info of database after filtering.
        """
        for name, min_num in min_gt_points_dict.items():
            min_num = int(min_num)
            if min_num > 0:
                filtered_infos = []
                for info in db_infos[name]:
                    if info["num_points_in_gt"] >= min_num:
                        filtered_infos.append(info)
                db_infos[name] = filtered_infos
        return db_infos

    @staticmethod
    def inv_lerp(a, b, v):
        """
        Inverse linear interpolation

        Args:
            a (float): min
            b (float): max
            v (float): value
        """
        return (v - a) / (b - a)

    def sample_rot(self, cls_label: int, sample_trans: float) -> float:
        # if sample_trans < 0.5:
        #     return 0.0
        val = self.cls_rot_lim[cls_label][1:]
        return self.cls_rot_fn[cls_label](*val)

    def sample_trans(self, cls_label: int) -> float:
        val = self.cls_trans_lim[cls_label][1:]
        return self.cls_trans_fn[cls_label](*val)

    def sample_all(self, gt_bboxes, gt_labels, img=None):
        """Sampling all categories of bboxes.

        Args:
            gt_bboxes (np.ndarray): Ground truth bounding boxes.
            gt_labels (np.ndarray): Ground truth labels of boxes.

        Returns:
            dict: Dict of sampled 'pseudo ground truths'.

                - gt_labels_3d (np.ndarray): ground truths labels \
                    of sampled objects.
                - gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): \
                    sampled ground truth 3D bounding boxes
                - points (np.ndarray): sampled points
                - group_ids (np.ndarray): ids of sampled ground truths
        """
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(self.sample_classes, self.sample_max_nums):
            class_label = self.cat2label[class_name]
            # sampled_num = int(max_sample_num -
            #                   np.sum([n == class_name for n in gt_names]))
            sampled_num = int(max_sample_num - np.sum([n == class_label for n in gt_labels]))
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled = []
        sampled_gt_bboxes = []
        avoid_coll_boxes = gt_bboxes

        for class_name, sampled_num in zip(self.sample_classes, sample_num_per_class):
            if sampled_num > 0:
                sampled_cls = self.sample_class_v2(class_name, sampled_num, avoid_coll_boxes)

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]["box3d_lidar"][np.newaxis, ...]
                    else:
                        sampled_gt_box = np.stack([s["box3d_lidar"] for s in sampled_cls], axis=0)

                    sampled_gt_bboxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate([avoid_coll_boxes, sampled_gt_box], axis=0)

        ret = None
        if len(sampled) > 0:
            sampled_gt_bboxes = np.concatenate(sampled_gt_bboxes, axis=0)
            # center = sampled_gt_bboxes[:, 0:3]

            s_num_points_in_gt_list = []
            s_distance_list = []
            s_points_list = []
            count = 0
            for info in sampled:
                file_path = (
                    os.path.join(self.dataset_root, info["path"])
                    if self.dataset_root
                    else info["path"]
                )
                results = dict(lidar_path=file_path)
                s_points = self.points_loader(results)["points"]
                s_points.translate(info["box3d_lidar"][:3])

                num_points_in_gt = info["num_points_in_gt"]
                distance = info["distance"]

                count += 1

                s_points_list.append(s_points)
                s_num_points_in_gt_list.append(num_points_in_gt)
                s_distance_list.append(distance)

            gt_labels = np.array([self.cat2label[s["name"]] for s in sampled], dtype=np.long)

            if self.cls_trans_lim is not None:
                sample_trans = np.array([self.sample_trans(x) for x in gt_labels], dtype=np.float32)
            else:
                sample_trans = None
            if self.cls_rot_lim is not None:
                if sample_trans is None:
                    sample_rot = np.array([self.sample_rot(x) for x in gt_labels], dtype=np.float32)
                else:
                    sample_rot = np.array(
                        [self.sample_rot(x, sample_trans[i]) for i, x in enumerate(gt_labels)],
                        dtype=np.float32,
                    )
            else:
                sample_rot = None

            # reduce points
            if (
                self.reduce_points_by_distance is not None
                and self.reduce_points_by_distance["prob"] > 0
            ):
                distance_threshold = self.reduce_points_by_distance["distance_threshold"]
                reduce_prob = self.reduce_points_by_distance["prob"]

                # trigger probability
                prob = np.random.uniform(0, 1)
                if prob < reduce_prob:
                    for i, (s_points, s_num_points_in_gt, s_distance) in enumerate(
                        zip(s_points_list, s_num_points_in_gt_list, s_distance_list)
                    ):
                        max_ratio = self.inv_lerp(0.0, distance_threshold, s_distance)
                        max_ratio = np.clip(
                            max_ratio, 0.0, self.reduce_points_by_distance["max_ratio"]
                        )

                        # trigger probability between 0 and max_ratio
                        ratio = np.random.uniform(0, max_ratio)

                        if ratio > 0:
                            s_points.shuffle()
                            num_points_to_keep = int(s_num_points_in_gt * (1 - ratio))
                            s_points.tensor = s_points.tensor[:num_points_to_keep]

            ret = {
                "gt_labels_3d": gt_labels,
                "gt_bboxes_3d": sampled_gt_bboxes,
                "points": s_points_list,  # s_points_list[0].cat(s_points_list),
                "group_ids": np.arange(gt_bboxes.shape[0], gt_bboxes.shape[0] + len(sampled)),
                "sampled_rot": sample_rot,
                "sampled_trans": sample_trans,
            }

        return ret

    def sample_class_v2(self, name, num, gt_bboxes):
        """Sampling specific categories of bounding boxes.

        Args:
            name (str): Class of objects to be sampled.
            num (int): Number of sampled bboxes.
            gt_bboxes (np.ndarray): Ground truth boxes.

        Returns:
            list[dict]: Valid samples after collision test.
        """
        sampled = self.sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_bboxes.shape[0]
        num_sampled = len(sampled)
        gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
            gt_bboxes[:, 0:2], gt_bboxes[:, 3:5], gt_bboxes[:, 6]
        )

        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)
        boxes = np.concatenate([gt_bboxes, sp_boxes], axis=0).copy()

        sp_boxes_new = boxes[gt_bboxes.shape[0] :]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6]
        )

        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
        coll_mat = box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])
        return valid_samples
