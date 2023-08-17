import copy
import json
import os
import random
import tempfile
import time
from collections import defaultdict
from os import path as osp
from typing import Any, Dict, List, Optional, Tuple

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from scipy.spatial.transform import Rotation

from ..core.bbox import LiDARInstance3DBoxes
from .custom_3d import Custom3DDataset


@DATASETS.register_module()
class OSDAR23Dataset(Custom3DDataset):
    CLASSES = [
        "lidar__cuboid__person",
        "lidar__cuboid__signal",
        "lidar__cuboid__catenary_pole",
        "lidar__cuboid__signal_pole",
        "lidar__cuboid__train",
        "lidar__cuboid__road_vehicle",
        "lidar__cuboid__buffer_stop",
        "lidar__cuboid__animal",
        "lidar__cuboid__switch",
        "lidar__cuboid__bicycle",
        "lidar__cuboid__crowd",
        "lidar__cuboid__wagons",
        "lidar__cuboid__signal_bridge",
    ]

    ErrNameMapping = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
    }

    # below ranges are used only to filter outliers in detections
    cls_range = {
        "lidar__cuboid__person": 400,
        "lidar__cuboid__signal": 400,
        "lidar__cuboid__catenary_pole": 400,
        "lidar__cuboid__signal_pole": 400,
        "lidar__cuboid__train": 400,
        "lidar__cuboid__road_vehicle": 400,
        "lidar__cuboid__buffer_stop": 400,
        "lidar__cuboid__animal": 400,
        "lidar__cuboid__switch": 400,
        "lidar__cuboid__bicycle": 400,
        "lidar__cuboid__crowd": 400,
        "lidar__cuboid__wagons": 400,
        "lidar__cuboid__signal_bridge": 400,
    }

    dist_fcn = "center_distance"
    dist_ths = [0.5, 1.0, 2.0, 4.0]
    dist_th_tp = 2.0
    min_recall = 0.1
    min_precision = 0.1
    max_boxes_per_sample = 500
    mean_ap_weight = 5

    eval_list = [
        "occlusion_0-25 %",  # occlusion
        "occlusion_25-50 %",  # occlusion
        "occlusion_50-75 %",  # occlusion
        "occlusion_75-99 %",  # occlusion
        "occlusion_100 %",  # occlusion
        "num_points_0-199",  # num_points
        "num_points_200-499",  # num_points
        "num_points_500-999",  # num_points
        "num_points_1000-1999",  # num_points
        "num_points_2000-2999",  # num_points
        "num_points_3000-inf",  # num_points
        "distance_0-49",  # distance
        "distance_50-99",  # distance
        "distance_100-149",  # distance
        "distance_150-199",  # distance
        "distance_200-inf",  # distance
        "overall",
    ]

    def __init__(
        self,
        ann_file,
        pipeline=None,
        dataset_root=None,
        object_classes=None,
        load_interval=1,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        use_valid_flag=False,
        eval_point_cloud_range: Optional[List[float]] = None,
        queue_length: int = 0,
        queue_range_threshold: int = 0,
        online: bool = False,
    ) -> None:
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag

        assert eval_point_cloud_range is None or len(eval_point_cloud_range) == 6
        assert isinstance(queue_length, int) and queue_length >= 0

        self.eval_point_cloud_range = eval_point_cloud_range
        self.queue_length = queue_length
        self.queue_range_threshold = queue_range_threshold
        self.online = online

        super().__init__(
            dataset_root=dataset_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=object_classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            online=online,
        )

        self.eval_detection_configs = {
            "class_range": self.cls_range,
            "point_cloud_range": eval_point_cloud_range,
            "queue_length": queue_length,
            "queue_range_threshold": queue_range_threshold,
            "dist_fcn": self.dist_fcn,
            "dist_ths": self.dist_ths,
            "dist_th_tp": self.dist_th_tp,
            "min_recall": self.min_recall,
            "min_precision": self.min_precision,
            "max_boxes_per_sample": self.max_boxes_per_sample,
            "mean_ap_weight": self.mean_ap_weight,
        }

        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info["valid_flag"]
            gt_names = set(info["gt_names"][mask])
        else:
            gt_names = set(info["gt_names"])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        return data_infos

    def get_data_info(self, index: int) -> Dict[str, Any]:
        info = self.data_infos[index]

        data = dict(
            lidar_path=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"],
        )

        # lidar to ego transform
        data["lidar2ego"] = info["lidar2ego"]

        # temporal
        if self.queue_length > 0:
            data["scene_token"] = info["scene_token"]
            data["token"] = info["token"]
            data["prev"] = info["prev"]
            data["next"] = info["next"]
            data["frame_idx"] = info["frame_idx"]

        if self.modality["use_camera"]:
            data["image_paths"] = []
            data["lidar2camera"] = []
            data["lidar2image"] = []
            data["camera2ego"] = []
            data["camera_intrinsics"] = []
            data["camera2lidar"] = []

            for _, camera_info in info["cams"].items():
                data["image_paths"].append(camera_info["data_path"])
                # lidar to camera transform
                camera2lidar = camera_info["sensor2lidar"]
                camera2lidar = np.vstack([camera2lidar, [0.0, 0.0, 0.0, 1.0]])
                lidar2camera = np.linalg.inv(camera2lidar)
                lidar2camera = lidar2camera[:-1, :]
                data["lidar2camera"].append(lidar2camera)
                # camera intrinsics
                data["camera_intrinsics"].append(camera_info["camera_intrinsics"])
                # lidar to image transform
                data["lidar2image"].append(camera_info["lidar2image"])
                # camera to ego transform
                data["camera2ego"].append(camera_info["sensor2ego"])
                # camera to lidar transform
                data["camera2lidar"].append(camera_info["sensor2lidar"])

        # if self.test_mode: # OSDAR23 has annotation for test split
        #     annos = None
        # else:
        annos = self.get_ann_info(index)
        data["ann_info"] = annos
        return data

    def union2one(self, queue):
        imgs_list = [each["img"].data for each in queue]
        points_list = [each["points"].data for each in queue]
        metas_map = [{}] * len(queue)

        prev_scene_token = None
        for i, each in enumerate(queue):
            metas_map[i] = each["metas"].data
            if metas_map[i]["scene_token"] != prev_scene_token:
                metas_map[i]["prev_bev_exists"] = False
                prev_scene_token = metas_map[i]["scene_token"]
            else:
                metas_map[i]["prev_bev_exists"] = True

            if metas_map[i]["next"] not in [None, ""]:
                metas_map[i]["next_bev_exists"] = True
            else:
                metas_map[i]["next_bev_exists"] = False

        queue[-1]["img"] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[-1]["points"] = DC(points_list, cpu_only=False)
        queue[-1]["metas"] = DC(metas_map, cpu_only=True)
        queue = queue[-1]

        return queue

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        if self.queue_length == 0:
            input_dict = self.get_data_info(index)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            if self.filter_empty_gt and (
                example is None or ~(example["gt_labels_3d"]._data != -1).any()
            ):
                return None
            return example
        else:  # temporal
            queue = []
            index_list = list(
                range(index - (self.queue_length - 1 + self.queue_range_threshold), index)
            )
            random.shuffle(index_list)  # to add randomness if range is greater
            index_list = sorted(index_list[-(self.queue_length - 1) :])
            index_list.append(index)
            latest_dict = None
            debug_frame_indices = []
            for i in index_list[::-1]:
                i = max(0, i)
                input_dict = self.get_data_info(i)
                if (
                    latest_dict is not None
                    and input_dict["scene_token"] != latest_dict["scene_token"]
                ):
                    input_dict = latest_dict
                else:
                    latest_dict = copy.deepcopy(input_dict)
                debug_frame_indices.append(input_dict["frame_idx"])
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                if self.filter_empty_gt and (
                    example is None or ~(example["gt_labels_3d"]._data != -1).any()
                ):
                    return None
                queue.append(example)
            # print(f"train {index} debug_frame_indices", debug_frame_indices[::-1])
            return self.union2one(queue[::-1])

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        if self.queue_length == 0:
            input_dict = self.get_data_info(index)
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            return example
        else:  # temporal
            queue = []
            index_list = list(
                range(index - (self.queue_length - 1 + self.queue_range_threshold), index)
            )
            random.shuffle(index_list)  # to add randomness if range is greater
            index_list = sorted(index_list[-(self.queue_length - 1) :])
            index_list.append(index)
            latest_dict = None
            debug_frame_indices = []
            for i in index_list[::-1]:
                i = max(0, i)
                input_dict = self.get_data_info(i)
                if (
                    latest_dict is not None
                    and input_dict["scene_token"] != latest_dict["scene_token"]
                ):
                    input_dict = latest_dict
                else:
                    latest_dict = copy.deepcopy(input_dict)
                debug_frame_indices.append(input_dict["frame_idx"])
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                queue.append(example)
            # print(f"test {index} debug_frame_indices", debug_frame_indices[::-1])
            return self.union2one(queue[::-1])

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0

        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_distance = info["distances"][mask]
        gt_occlusions = info["occlusions"][mask]
        gt_distance_level = info["distance_levels"][mask]
        gt_num_points_level = info["num_points_levels"][mask]
        gt_labels_3d = []

        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.5)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            distance=gt_distance,
            occlusion=gt_occlusions,
            distance_level=gt_distance_level,
            num_points_level=gt_num_points_level,
        )
        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")
        for index, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            ts = str(self.data_infos[index]["timestamp"])
            boxes = output_to_box_dict(det)
            boxes = filter_box_in_lidar_cs(boxes, mapped_class_names, self.eval_detection_configs)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box["label"]]

                nusc_anno = dict(
                    timestamp=ts,
                    translation=box["center"].tolist(),
                    size=box["wlh"].tolist(),
                    rotation=box["orientation"],
                    detection_name=name,
                    detection_score=box["score"],
                )
                annos.append(nusc_anno)
            nusc_annos[ts] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_osdar23.json")
        print("Results writes to", res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(
            self
        ), "The length of results is not equal to the dataset len: {} != {}".format(
            len(results), len(self)
        )

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        result_files = self._format_bbox(results, jsonfile_prefix)
        return result_files, tmp_dir

    # IDEA: Custom evaluation based on adapted NuScenes functions but not using nuscenes itself since it needs tokens
    # SEE:
    # https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/evaluate.py
    # https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/algo.py
    # https://github.com/nutonomy/nuscenes-devkit/blob/da3c9a977112fca05413dab4e944d911769385a9/python-sdk/nuscenes/eval/common/utils.py
    # https://github.com/nutonomy/nuscenes-devkit/blob/da3c9a977112fca05413dab4e944d911769385a9/python-sdk/nuscenes/eval/detection/data_classes.py

    def load_prediction(self, result_path: str, max_boxes_per_sample: int, verbose: bool = False):
        """
        Loads object predictions from file.
        :param result_path: Path to the .json result file provided by the user.
        :param max_boxes_per_sample: Maximum number of boxes allowed per sample.
        :param verbose: Whether to print messages to stdout.
        :return: The deserialized results and meta data.
        """

        # Load from file and check that the format is correct.
        with open(result_path) as f:
            data = json.load(f)
        assert "results" in data, "Error: No field `results` in result file."

        # Deserialize results and get meta data.
        all_results = {}
        for idx, boxes in data["results"].items():
            box_list = []
            for box in boxes:
                box_list.append(
                    {
                        "timestamp": box["timestamp"],
                        "translation": box["translation"],
                        "ego_dist": np.sqrt(np.sum(np.array(box["translation"][:2]) ** 2)),
                        "location": box["translation"][:3],
                        "size": box["size"],
                        "rotation": box["rotation"],
                        "num_pts": -1 if "num_pts" not in box else int(box["num_pts"]),
                        "detection_name": box["detection_name"],
                        "detection_score": -1.0
                        if "detection_score" not in box
                        else float(box["detection_score"]),
                    }
                )
            all_results[idx] = box_list

        meta = data["meta"]
        if verbose:
            print(
                "Loaded results from {}. Found detections for {} samples.".format(
                    result_path, len(all_results)
                )
            )

        # Check that each sample has no more than x predicted boxes.
        for result in all_results:
            assert len(all_results[result]) <= max_boxes_per_sample, (
                "Error: Only <= %d boxes per sample allowed!" % max_boxes_per_sample
            )

        return all_results, meta

    def load_gt(self, verbose: bool = False):
        """
        Loads ground truth boxes from database.
        :param nusc: A NuScenes instance.
        :param eval_split: The evaluation split for which we load GT boxes.
        :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
        :param verbose: Whether to print messages to stdout.
        :return: The GT boxes.
        """

        assert len(self.data_infos) > 0, "Error: Pickle has no samples!"

        all_annotations = {}

        for i, info in enumerate(self.data_infos):
            json1_file = open(info["lidar_anno_path"])
            json1_str = json1_file.read()
            lidar_annotation = json.loads(json1_str)

            lidar_anno_frame = {}

            for j in lidar_annotation["openlabel"]["frames"]:
                lidar_anno_frame = lidar_annotation["openlabel"]["frames"][j]

            timestamp = str(lidar_anno_frame["frame_properties"]["timestamp"])

            sample_boxes = []

            for id in lidar_anno_frame["objects"]:
                object_data = lidar_anno_frame["objects"][id]["object_data"]

                loc = np.asarray(object_data["cuboid"][0]["val"][:3], dtype=np.float32)
                dim = np.asarray(object_data["cuboid"][0]["val"][7:], dtype=np.float32)
                rot = np.asarray(
                    object_data["cuboid"][0]["val"][3:7], dtype=np.float32
                )  # Quaternion in x,y,z,w

                rot_temp = Rotation.from_quat(rot)
                rot_temp = rot_temp.as_euler("xyz", degrees=False)

                yaw = np.asarray(rot_temp[2], dtype=np.float32)

                num_lidar_pts = 0
                difficulty_level = None
                occlusion_level = None
                distance_level = None
                num_points_level = None

                for n in object_data["cuboid"][0]["attributes"]["num"]:
                    if n["name"] == "num_points":
                        num_lidar_pts = n["val"]
                for n in object_data["cuboid"][0]["attributes"]["text"]:
                    if n["name"] == "occlusion":
                        occlusion_level = n["val"]
                    if n["name"] == "distance_level":
                        distance_level = n["val"]
                    if n["name"] == "num_points_level":
                        num_points_level = n["val"]

                sample_boxes.append(
                    {
                        "timestamp": timestamp,
                        "translation": loc,
                        "ego_dist": np.sqrt(np.sum(np.array(loc[:2]) ** 2)),
                        "location": loc,
                        "size": dim,
                        "rotation": yaw,
                        "num_pts": num_lidar_pts,
                        "occlusion": occlusion_level,
                        "distance_level": distance_level,
                        "num_points_level": num_points_level,
                        "difficulty": difficulty_level,
                        "detection_name": object_data["cuboid"][0]["name"],
                        "detection_score": -1.0,  # GT samples do not have a score.
                    }
                )

            all_annotations[timestamp] = sample_boxes

        if verbose:
            print("Loaded ground truth annotations for {} samples.".format(len(all_annotations)))

        return all_annotations

    def center_distance(self, gt_box, pred_box) -> float:
        """
        L2 distance between the box centers (xy only).
        :param gt_box: GT annotation sample.
        :param pred_box: Predicted sample.
        :return: L2 distance.
        """
        return np.linalg.norm(
            np.array(pred_box["translation"][:2]) - np.array(gt_box["translation"][:2])
        )

    def yaw_diff(self, gt_box, eval_box, period: float = 2 * np.pi) -> float:
        """
        Returns the yaw angle difference between the orientation of two boxes.
        :param gt_box: Ground truth box.
        :param eval_box: Predicted box.
        :param period: Periodicity in radians for assessing angle difference.
        :return: Yaw angle difference in radians in [0, pi].
        """
        yaw_gt = gt_box["rotation"]
        yaw_est = eval_box["rotation"]

        return abs(self.angle_diff(yaw_gt, yaw_est, period))

    def angle_diff(self, x: float, y: float, period: float) -> float:
        """
        Get the smallest angle difference between 2 angles: the angle from y to x.
        :param x: To angle.
        :param y: From angle.
        :param period: Periodicity in radians for assessing angle difference.
        :return: <float>. Signed smallest between-angle difference in range (-pi, pi).
        """

        # calculate angle difference, modulo to [0, 2*pi]
        diff = (x - y + period / 2) % period - period / 2
        if diff > np.pi:
            diff = diff - (2 * np.pi)  # shift (pi, 2*pi] to (-pi, 0]

        return diff

    def scale_iou(self, sample_annotation, sample_result) -> float:
        """
        This method compares predictions to the ground truth in terms of scale.
        It is equivalent to intersection over union (IOU) between the two boxes in 3D,
        if we assume that the boxes are aligned, i.e. translation and rotation are considered identical.
        :param sample_annotation: GT annotation sample.
        :param sample_result: Predicted sample.
        :return: Scale IOU.
        """
        # Validate inputs.
        sa_size = np.array(sample_annotation["size"])
        sr_size = np.array(sample_result["size"])
        assert all(sa_size > 0), "Error: sample_annotation sizes must be >0."
        assert all(sr_size > 0), "Error: sample_result sizes must be >0."

        # Compute IOU.
        min_wlh = np.minimum(sa_size, sr_size)
        volume_annotation = np.prod(sa_size)
        volume_result = np.prod(sr_size)
        intersection = np.prod(min_wlh)  # type: float
        union = volume_annotation + volume_result - intersection  # type: float
        iou = intersection / union

        return iou

    def cummean(self, x: np.array) -> np.array:
        """
        Computes the cumulative mean up to each position in a NaN sensitive way
        - If all values are NaN return an array of ones.
        - If some values are NaN, accumulate arrays discording those entries.
        """
        if sum(np.isnan(x)) == len(x):
            # Is all numbers in array are NaN's.
            return np.ones(
                len(x)
            )  # If all errors are NaN set to error to 1 for all operating points.
        else:
            # Accumulate in a nan-aware manner.
            sum_vals = np.nancumsum(x.astype(float))  # Cumulative sum ignoring nans.
            count_vals = np.cumsum(~np.isnan(x))  # Number of non-nans up to each position.
            return np.divide(
                sum_vals, count_vals, out=np.zeros_like(sum_vals), where=count_vals != 0
            )

    def accumulate(
        self,
        gt_boxes: list,
        pred_boxes: list,
        class_name: str,
        dist_th: float,
        verbose: bool = False,
    ):
        """
        Average Precision over predefined different recall thresholds for a single distance threshold.
        The recall/conf thresholds and other raw metrics will be used in secondary metrics.
        :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
        :param pred_boxes: Maps every sample_token to a list of its sample_results.
        :param class_name: Class to compute AP on.
        :param dist_fcn: Distance function used to match detections and ground truths.
        :param dist_th: Distance threshold for a match.
        :param verbose: If true, print debug messages.
        :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.
        """
        # ---------------------------------------------
        # Organize input and initialize accumulators.
        # ---------------------------------------------

        # Count the positives.
        gt_boxes_all = []
        for key in gt_boxes:
            gt_boxes_all.extend(gt_boxes[key])
        npos = len([1 for box in gt_boxes_all if box["detection_name"] == class_name])
        if verbose:
            print(
                "Found {} GT of class {} out of {} total across {} samples.".format(
                    npos, class_name, len(gt_boxes_all), len(gt_boxes.keys())
                )
            )

        # For missing classes in the GT, return a data structure corresponding to no predictions.
        if npos == 0:
            # Return dict with values of nuScenes DetectionMetricData.no_predictions()
            return {
                "recall": np.linspace(0, 1, 101),  # 101 is from nuScene's nelem value
                "precision": np.zeros(101),
                "confidence": np.zeros(101),
                "trans_err": np.ones(101),
                "scale_err": np.ones(101),
                "orient_err": np.ones(101),
            }

        # Organize the predictions in a single list.
        pred_boxes_all = []
        for key in pred_boxes:
            pred_boxes_all.extend(pred_boxes[key])
        pred_boxes_list = [box for box in pred_boxes_all if box["detection_name"] == class_name]
        pred_confs = [box["detection_score"] for box in pred_boxes_list]

        if verbose:
            print(
                "Found {} PRED of class {} out of {} total across {} samples.".format(
                    len(pred_confs), class_name, len(pred_boxes_all), len(pred_boxes.keys())
                )
            )

        # Sort by confidence.
        sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

        # Do the actual matching.
        tp = []  # Accumulator of true positives.
        fp = []  # Accumulator of false positives.
        conf = []  # Accumulator of confidences.

        # match_data holds the extra metrics we calculate for each match.
        match_data = {"trans_err": [], "scale_err": [], "orient_err": [], "conf": []}

        # ---------------------------------------------
        # Match and accumulate match data.
        # ---------------------------------------------

        taken = set()  # Initially no gt bounding box is matched.
        for ind in sortind:
            pred_box = pred_boxes_list[ind]
            min_dist = np.inf
            match_gt_idx = None

            for gt_idx, gt_box in enumerate(gt_boxes[pred_box["timestamp"]]):
                # Find closest match among ground truth boxes
                if (
                    gt_box["detection_name"] == class_name
                    and not (pred_box["timestamp"], gt_idx) in taken
                ):
                    this_distance = self.center_distance(gt_box, pred_box)
                    if this_distance < min_dist:
                        min_dist = this_distance
                        match_gt_idx = gt_idx

            # If the closest match is close enough according to threshold we have a match!
            is_match = min_dist < dist_th

            if is_match:
                taken.add((pred_box["timestamp"], match_gt_idx))

                #  Update tp, fp and confs.
                tp.append(1)
                fp.append(0)
                conf.append(pred_box["detection_score"])

                # Since it is a match, update match data also.
                gt_box_match = gt_boxes[pred_box["timestamp"]][match_gt_idx]

                match_data["trans_err"].append(self.center_distance(gt_box_match, pred_box))
                match_data["scale_err"].append(1 - self.scale_iou(gt_box_match, pred_box))

                # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
                period = np.pi if class_name == "barrier" else 2 * np.pi
                match_data["orient_err"].append(
                    self.yaw_diff(gt_box_match, pred_box, period=period)
                )

                match_data["conf"].append(pred_box["detection_score"])

            else:
                # No match. Mark this as a false positive.
                tp.append(0)
                fp.append(1)
                conf.append(pred_box["detection_score"])

        # Check if we have any matches. If not, just return a "no predictions" array.
        if len(match_data["trans_err"]) == 0:
            # Return dict with values of nuScenes DetectionMetricData.no_predictions()
            return {
                "recall": np.linspace(0, 1, 101),  # 101 is from nuScene's nelem value
                "precision": np.zeros(101),
                "confidence": np.zeros(101),
                "trans_err": np.ones(101),
                "scale_err": np.ones(101),
                "orient_err": np.ones(101),
            }

        # ---------------------------------------------
        # Calculate and interpolate precision and recall
        # ---------------------------------------------

        # Accumulate.
        tp = np.cumsum(tp).astype(float)
        fp = np.cumsum(fp).astype(float)
        conf = np.array(conf)

        # Calculate precision and recall.
        prec = tp / (fp + tp)
        rec = tp / float(npos)

        rec_interp = np.linspace(0, 1, 101)  # 101 steps, from 0% to 100% recall.
        prec = np.interp(rec_interp, rec, prec, right=0)
        conf = np.interp(rec_interp, rec, conf, right=0)
        rec = rec_interp

        # ---------------------------------------------
        # Re-sample the match-data to match, prec, recall and conf.
        # ---------------------------------------------

        for key in match_data.keys():
            if key == "conf":
                continue  # Confidence is used as reference to align with fp and tp. So skip in this step.

            else:
                # For each match_data, we first calculate the accumulated mean.
                tmp = self.cummean(np.array(match_data[key]))

                # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
                match_data[key] = np.interp(conf[::-1], match_data["conf"][::-1], tmp[::-1])[::-1]

        # ---------------------------------------------
        # Done. Instantiate MetricData and return
        # ---------------------------------------------
        return {
            "recall": rec,
            "precision": prec,
            "confidence": conf,
            "trans_err": match_data["trans_err"],
            "scale_err": match_data["scale_err"],
            "orient_err": match_data["orient_err"],
        }

    def filter_eval_boxes(
        self,
        eval_boxes,
        max_dist: Dict[str, float],
        point_range: Optional[List[float]] = None,
        verbose: bool = False,
    ):
        """
        Applies filtering to boxes. Distance, bike-racks and points per box.
        :param nusc: An instance of the NuScenes class.
        :param eval_boxes: An instance of the EvalBoxes class.
        :param max_dist: Maps the detection name to the eval distance threshold for that class.
        :param point_range: Only evaluate boxes within this range from the ego vehicle.
        :param verbose: Whether to print to stdout.
        """
        # Accumulators for number of filtered boxes.
        total, after_filter, point_filter = 0, 0, 0
        for timestamp in eval_boxes:
            total += len(eval_boxes[timestamp])
            # filter based on point_range
            if point_range is not None:
                eval_boxes[timestamp] = [
                    box
                    for box in eval_boxes[timestamp]
                    if (
                        box["location"][0] > point_range[0]
                        and box["location"][0] < point_range[3]
                        and box["location"][1] > point_range[1]
                        and box["location"][1] < point_range[4]
                        and box["location"][2] > point_range[2]
                        and box["location"][2] < point_range[5]
                    )
                ]
            # Filter on distance.
            if max_dist is not None:
                eval_boxes[timestamp] = [
                    box
                    for box in eval_boxes[timestamp]
                    if box["ego_dist"] < max_dist[box["detection_name"]]
                ]

            after_filter += len(eval_boxes[timestamp])

            # Then remove boxes with zero points in them. Eval boxes have -1 points by default.
            eval_boxes[timestamp] = [
                box for box in eval_boxes[timestamp] if not box["num_pts"] == 0
            ]
            point_filter += len(eval_boxes[timestamp])

        if verbose:
            print("=> Original number of boxes: %d" % total)
            print("=> After distance based filtering: %d" % after_filter)
            print("=> After LIDAR and RADAR points based filtering: %d" % point_filter)

        return eval_boxes

    def calc_ap(self, md, min_recall: float, min_precision: float) -> float:
        """Calculated average precision."""

        assert 0 <= min_precision < 1
        assert 0 <= min_recall <= 1

        prec = np.copy(md["precision"])
        prec = prec[
            round(100 * min_recall) + 1 :
        ]  # Clip low recalls. +1 to exclude the min recall bin.
        prec -= min_precision  # Clip low precision
        prec[prec < 0] = 0
        return float(np.mean(prec)) / (1.0 - min_precision)

    def calc_tp(self, md, min_recall: float, metric_name: str) -> float:
        """Calculates true positive errors."""

        first_ind = round(100 * min_recall) + 1  # +1 to exclude the error at min recall.

        # Last instance of confidence > 0 is index of max achieved recall.
        non_zero = np.nonzero(md["confidence"])[0]
        if len(non_zero) == 0:  # If there are no matches, all the confidence values will be zero.
            max_recall_ind = 0
        else:
            max_recall_ind = non_zero[-1]

        last_ind = (
            max_recall_ind  # First instance of confidence = 0 is index of max achieved recall.
        )

        if last_ind < first_ind:
            return 1.0  # Assign 1 here. If this happens for all classes, the score for that TP metric will be 0.
        else:
            return float(
                np.mean(md[metric_name][first_ind : last_ind + 1])
            )  # +1 to include error at max recall.

    def serializeMetricDara(self, value):
        return {
            "recall": value["recall"].tolist(),
            "precision": value["precision"].tolist(),
            "confidence": value["confidence"].tolist(),
            "trans_err": value["trans_err"].tolist(),
            "scale_err": value["scale_err"].tolist(),
            "orient_err": value["orient_err"].tolist(),
        }

    def filter_boxes_per_eval_type(self, eval_boxes, eval_name: str, verbose: bool = False) -> Dict:
        evals = copy.deepcopy(eval_boxes)

        filtered_evals = defaultdict(list)
        occlusion_map = {
            "occlusion_0-25 %": "0-25 %",
            "occlusion_25-50 %": "25-50 %",
            "occlusion_50-75 %": "50-75 %",
            "occlusion_75-99 %": "75-99 %",
            "occlusion_100 %": "100 %",
        }
        num_points_map = {
            "num_points_0-199": "0-199",
            "num_points_200-499": "200-499",
            "num_points_500-999": "500-999",
            "num_points_1000-1999": "1000-1999",
            "num_points_2000-2999": "2000-2999",
            "num_points_3000-inf": "3000-inf",
        }
        distance_map = {
            "distance_0-49": "0-49",
            "distance_50-99": "50-99",
            "distance_100-149": "100-149",
            "distance_150-199": "150-199",
            "distance_200-inf": "200-inf",
        }

        if eval_name == "overall":  # no filtering applied
            return evals

        elif eval_name in list(occlusion_map.keys()):  # based on occlusion level
            for timestamp, e_boxes in evals.items():
                for e_box in e_boxes:
                    occlusion_level = occlusion_map[eval_name]
                    if e_box["occlusion"] == occlusion_level:
                        filtered_evals[timestamp].append(e_box)
        elif eval_name in list(num_points_map.keys()):  # based on number of points
            for timestamp, e_boxes in evals.items():
                for e_box in e_boxes:
                    num_points_level = num_points_map[eval_name]
                    if e_box["num_points_level"] == num_points_level:
                        filtered_evals[timestamp].append(e_box)
        elif eval_name in list(distance_map.keys()):  # based on distance
            for timestamp, e_boxes in evals.items():
                for e_box in e_boxes:
                    distance_level = distance_map[eval_name]
                    if e_box["distance_level"] == distance_level:
                        filtered_evals[timestamp].append(e_box)
        else:
            raise Exception(f"invalid eval_name while filtering: {eval_name}")

        return filtered_evals

    def get_gt_boxes_classes(self, gt_boxes, verbose: bool = False) -> Tuple[List[str], dict]:
        cls_list: set = set()
        cls_total = {}

        for _, e_boxes in gt_boxes.items():
            for e_box in e_boxes:
                cls_list.add(e_box["detection_name"])
                if e_box["detection_name"] not in cls_total:
                    cls_total[e_box["detection_name"]] = 1
                else:
                    cls_total[e_box["detection_name"]] += 1

        return list(cls_list), cls_total

    def _custom_evaluate(
        self,
        config: dict,
        result_path: str,
        output_dir: str = None,
        verbose: bool = True,
        extensive_report: bool = False,
        save_summary_path: Optional[str] = None,
    ):
        assert osp.exists(result_path), "Error: The result file does not exist!"

        if verbose:
            print("Initializing TUMTraf-I detection evaluation")
        self.pred_boxes, self.meta = self.load_prediction(
            result_path, self.max_boxes_per_sample, verbose=verbose
        )
        self.gt_boxes = self.load_gt(verbose=verbose)

        assert set(self.pred_boxes.keys()) == set(
            self.gt_boxes.keys()
        ), "Samples in split doesn't match samples in predictions."

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print("Filtering predictions")
        self.pred_boxes = self.filter_eval_boxes(
            self.pred_boxes, None, config["point_cloud_range"], verbose=verbose
        )
        if verbose:
            print("Filtering ground truth annotations")
        self.gt_boxes = self.filter_eval_boxes(
            self.gt_boxes, None, config["point_cloud_range"], verbose=verbose
        )

        self.keys = self.gt_boxes.keys()

        all_metrics_summary = {}
        all_metric_data_list = {}

        iter_list = self.eval_list if extensive_report else ["overall"]

        for eval_name in iter_list:
            if verbose:
                print(f"Starting metric evaluation for {eval_name}")

            curr_gt_boxes = self.filter_boxes_per_eval_type(
                self.gt_boxes, eval_name, verbose=verbose
            )
            curr_gt_classes, curr_gt_class_counts = self.get_gt_boxes_classes(
                curr_gt_boxes, verbose=verbose
            )
            curr_gt_classes = [i for i in self.CLASSES if i in curr_gt_classes]

            start_time = time.time()

            # -----------------------------------
            # Step 1: Accumulate metric data for all classes and distance thresholds.
            # -----------------------------------
            if verbose:
                print("Accumulating metric data...")

            metric_data_list = {}
            for class_name in curr_gt_classes:
                for dist_th in self.dist_ths:
                    md = self.accumulate(curr_gt_boxes, self.pred_boxes, class_name, dist_th)
                    metric_data_list[(class_name, dist_th)] = md

            # -----------------------------------
            # Step 2: Calculate metrics from the data.
            # -----------------------------------
            if verbose:
                print("Calculating metrics...")
            metrics = {
                "label_aps": defaultdict(lambda: defaultdict(float)),
                "label_tp_errors": defaultdict(lambda: defaultdict(float)),
            }
            for class_name in curr_gt_classes:
                # Compute APs.
                for dist_th in self.dist_ths:
                    metric_data = metric_data_list[(class_name, dist_th)]
                    ap = self.calc_ap(metric_data, self.min_recall, self.min_precision)
                    metrics["label_aps"][class_name][dist_th] = ap

                # Compute TP metrics.
                TP_METRICS = ["trans_err", "scale_err", "orient_err"]
                for metric_name in TP_METRICS:
                    metric_data = metric_data_list[(class_name, self.dist_th_tp)]
                    tp = self.calc_tp(metric_data, self.min_recall, metric_name)
                    metrics["label_tp_errors"][class_name][metric_name] = tp

            # Compute evaluation time.
            metrics["eval_time"] = time.time() - start_time

            # Compute other values for metrics summary
            mean_dist_aps = {
                class_name: np.mean(list(d.values()))
                for class_name, d in metrics["label_aps"].items()
            }
            mean_ap = float(np.mean(list(mean_dist_aps.values())))

            tp_errors = {}
            for metric_name in TP_METRICS:
                class_errors = []
                for detection_name in curr_gt_classes:
                    class_errors.append(metrics["label_tp_errors"][detection_name][metric_name])

                tp_errors[metric_name] = float(np.nanmean(class_errors))

            tp_scores = {}
            for metric_name in TP_METRICS:
                # We convert the true positive errors to "scores" by 1-error.
                score = 1.0 - tp_errors[metric_name]

                # Some of the true positive errors are unbounded, so we bound the scores to min 0.
                score = max(0.0, score)

                tp_scores[metric_name] = score

            # Summarize.
            nd_score = float(self.mean_ap_weight * mean_ap + np.sum(list(tp_scores.values())))
            # Normalize.
            nd_score = nd_score / float(self.mean_ap_weight + len(tp_scores.keys()))

            metrics_summary = {
                "label_aps": metrics["label_aps"],
                "mean_dist_aps": mean_dist_aps,
                "mean_ap": mean_ap,
                "label_tp_errors": metrics["label_tp_errors"],
                "tp_errors": tp_errors,
                "tp_scores": tp_scores,
                "nd_score": nd_score,
                "eval_time": metrics["eval_time"],
                "cfg": self.eval_detection_configs,
            }
            metrics_summary["meta"] = self.meta.copy()

            # Print high-level metrics.
            print(f"\n{eval_name} - mAP: {metrics_summary['mean_ap']:.4f}")
            err_name_mapping = {
                "trans_err": "mATE",
                "scale_err": "mASE",
                "orient_err": "mAOE",
            }
            for tp_name, tp_val in metrics_summary["tp_errors"].items():
                print(f"{eval_name} - {err_name_mapping[tp_name]}: {tp_val:.4f}")
            print(f"{eval_name} - NDS: {metrics_summary['nd_score']:.4f}")
            print("Eval time: %.1fs" % metrics_summary["eval_time"])
            total_gt_bboxes = sum([len(x) for x in curr_gt_boxes.values()])
            print(f"Total number of gt bboxes: {total_gt_bboxes}")
            print(f"GT class counts: {curr_gt_class_counts}")

            # Print per-class metrics.
            print(f"\n{eval_name} - Per-class results:")
            print("%-20s\t%-6s\t%-6s\t%-6s\t%-6s" % ("Object Class", "AP", "ATE", "ASE", "AOE"))
            class_aps = metrics_summary["mean_dist_aps"]
            class_tps = metrics_summary["label_tp_errors"]
            for class_name in class_aps.keys():
                print(
                    "%-20s\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f"
                    % (
                        class_name[15:],
                        class_aps[class_name],
                        class_tps[class_name]["trans_err"],
                        class_tps[class_name]["scale_err"],
                        class_tps[class_name]["orient_err"],
                    )
                )

            metrics_summary["total_gt_bboxes"] = total_gt_bboxes
            metrics_summary["class_counts"] = {}
            for x in self.CLASSES:
                if x in curr_gt_class_counts:
                    metrics_summary["class_counts"][x] = curr_gt_class_counts[x]
                else:
                    metrics_summary["class_counts"][x] = 0

            all_metrics_summary[eval_name] = metrics_summary
            all_metric_data_list[eval_name] = metric_data_list

        # Dump the metric data, meta and metrics to disk.
        if verbose:
            print("Saving metrics to: %s" % output_dir)

        with open(os.path.join(output_dir, "metrics_summary.json"), "w") as f:
            json.dump(all_metrics_summary["overall"], f, indent=2)

        if save_summary_path is not None:
            os.makedirs(os.path.dirname(save_summary_path), exist_ok=True)
            with open(save_summary_path, "w") as f:
                json.dump(all_metrics_summary, f, indent=2)

        mdl_dump = {
            key[0] + ":" + str(key[1]): self.serializeMetricDara(value)
            for key, value in all_metric_data_list["overall"].items()
        }

        with open(os.path.join(output_dir, "metrics_details.json"), "w") as f:
            json.dump(mdl_dump, f, indent=2)

        return all_metrics_summary

    def _evaluate_single(
        self,
        result_path,
        logger=None,
        metric="bbox",
        result_name="pts_bbox",
        extensive_report: bool = False,
        save_summary_path: Optional[str] = None,
    ):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.
            extensive_report (bool): Whether to print extensive report.
                Default: False.
            save_summary_path (str | None): Path to save the summary of

        Returns:
            dict: Dictionary of evaluation details.
        """
        output_dir = osp.join(*osp.split(result_path)[:-1])

        self._custom_evaluate(
            config=self.eval_detection_configs,
            result_path=result_path,
            output_dir=output_dir,
            verbose=False,
            extensive_report=extensive_report,
            save_summary_path=save_summary_path,
        )

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
        detail = dict()
        for name in self.CLASSES:
            for k, v in metrics["label_aps"][name].items():
                val = float("{:.4f}".format(v))
                detail["object/{}_ap_dist_{}".format(name, k)] = val
            for k, v in metrics["label_tp_errors"][name].items():
                val = float("{:.4f}".format(v))
                detail["object/{}_{}".format(name, k)] = val
            for k, v in metrics["tp_errors"].items():
                val = float("{:.4f}".format(v))
                detail["object/{}".format(self.ErrNameMapping[k])] = val

        detail["object/nds"] = metrics["nd_score"]
        detail["object/map"] = metrics["mean_ap"]
        return detail

    def evaluate(
        self,
        results,
        metric="bbox",
        jsonfile_prefix=None,
        result_names=["pts_bbox"],
        **kwargs,
    ):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        metrics = {}
        extensive_report = kwargs["extensive_report"] if "extensive_report" in kwargs else False
        save_summary_path = kwargs["save_summary_path"] if "save_summary_path" in kwargs else None

        if "boxes_3d" in results[0]:
            result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

            if isinstance(result_files, dict):
                for name in result_names:
                    print("Evaluating bboxes of {}".format(name))
                    ret_dict = self._evaluate_single(
                        result_files[name],
                        extensive_report=extensive_report,
                        save_summary_path=save_summary_path,
                    )
                metrics.update(ret_dict)
            elif isinstance(result_files, str):
                metrics.update(
                    self._evaluate_single(
                        result_files,
                        extensive_report=extensive_report,
                        save_summary_path=save_summary_path,
                    )
                )

            if tmp_dir is not None:
                tmp_dir.cleanup()

        return metrics


def output_to_box_dict(detection):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`dict`]: List of standard box dicts.
    """
    box3d = detection["boxes_3d"]
    scores = detection["scores_3d"].numpy()
    labels = detection["labels_3d"].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head

    box_list = []
    for i in range(len(box3d)):
        box = {
            "center": np.array(box_gravity_center[i]),
            "wlh": np.array(box_dims[i]),
            "orientation": box_yaw[i],
            "label": int(labels[i]) if not np.isnan(labels[i]) else labels[i],
            "score": float(scores[i]) if not np.isnan(scores[i]) else scores[i],
            "name": None,
        }
        box_list.append(box)
    return box_list


def filter_box_in_lidar_cs(boxes, classes, eval_configs):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`dict`]): List of predicted box dicts.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs : Evaluation configuration object.
        eval_version (str): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard box dicts in the global
            coordinate.
    """
    box_list = []
    point_range = eval_configs["point_cloud_range"]
    for box in boxes:
        box_x = box["center"][0]
        box_y = box["center"][1]
        if (
            box_x > point_range[0]
            and box_x < point_range[3]
            and box_y > point_range[1]
            and box_y < point_range[4]
        ):
            box_list.append(box)
        # filter det in ego.
        # cls_range_map = eval_configs["class_range"]
        # radius = np.linalg.norm(box["center"][:2], 2)
        # det_range = cls_range_map[classes[box["label"]]]
        # if radius > det_range:
        #     continue
        # box_list.append(box)
    return box_list
