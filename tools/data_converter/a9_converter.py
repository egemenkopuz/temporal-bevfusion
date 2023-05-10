import json
import logging
import os
import shutil
from glob import glob
from typing import Any, Dict, List

import mmcv
import numpy as np
from pypcd import pypcd
from scipy.spatial.transform import Rotation

lidar2ego = np.asarray(
    [
        [0.99011437, -0.13753536, -0.02752358, 2.3728100375737995],
        [0.13828977, 0.99000475, 0.02768645, -16.19297517556697],
        [0.02344061, -0.03121898, 0.99923766, -8.620000000000005],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
lidar2ego = lidar2ego[:-1, :]

lidar2s1image = np.asarray(
    [
        [7.04216073e02, -1.37317442e03, -4.32235765e02, -2.03369364e04],
        [-9.28351327e01, -1.77543929e01, -1.45629177e03, 9.80290034e02],
        [8.71736000e-01, -9.03453000e-02, -4.81574000e-01, -2.58546000e00],
    ],
    dtype=np.float32,
)

lidar2s2image = np.asarray(
    [
        [1546.63215008, -436.92407115, -295.58362676, 1319.79271737],
        [93.20805656, 47.90351592, -1482.13403199, 687.84781276],
        [0.73326062, 0.59708904, -0.32528854, -1.30114325],
    ],
    dtype=np.float32,
)

south1intrinsics = np.asarray(
    [
        [1400.3096617691212, 0.0, 967.7899705163408],
        [0.0, 1403.041082755918, 581.7195041357244],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

south12ego = np.asarray(
    [
        [-0.05009677, -0.987161, 0.15166403, -11.269694],
        [-0.46497604, -0.11134061, -0.87829417, -16.913832],
        [0.8839045, -0.11452018, -0.45342848, -11.169352],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
south12ego = south12ego[:-1, :]

south12lidar = np.linalg.inv(
    np.asarray(
        [
            [-0.09318435, -0.48103285, 0.87173647, 1.2347497162409127],
            [-0.9954856, 0.02911651, -0.09034505, -13.839581018313766],
            [0.01807675, -0.8762188, -0.48157436, 0.28000000000002956],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
)
south12lidar = south12lidar[:-1, :]

south2intrinsics = np.asarray(
    [
        [1029.2795655594014, 0.0, 982.0311857478633],
        [0.0, 1122.2781391971948, 1129.1480997238505],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

south22ego = np.asarray(
    [
        [0.650906, -0.7435749, 0.15303044, 4.6059465],
        [-0.14764456, -0.32172203, -0.935252, -15.00049],
        [0.74466264, 0.5861663, -0.3191956, -9.351643],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
south22ego = south22ego[:-1, :]

south22lidar = np.linalg.inv(
    np.asarray(
        [
            [0.64150846, -0.25893894, 0.72209245, -0.7326694886432961],
            [-0.76697475, -0.23453838, 0.5972786, 2.4730171719565988],
            [0.01469944, -0.9369856, -0.34905794, 0.5400000000000205],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
)
south22lidar = south22lidar[:-1, :]


class A92KITTI:
    """A9 dataset to KITTI converter.

    This class serves as the converter to change the A9 data to KITTI format.
    """

    def __init__(self, splits: List[str], load_dir: str, save_dir: str, name_format: str = "name"):
        """
        Args:
            splits list[(str)]: Contains the different splits
            version (str): Specify the modality
            load_dir (str): Directory to load waymo raw data.
            save_dir (str): Directory to save data in KITTI format.
            name_format (str): Specify the output name of the converted file mmdetection3d expects names to but numbers
        """

        self.splits = splits
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.name_format = name_format
        self.label_save_dir = "label_2"
        self.point_cloud_save_dir = "point_clouds"

        self.imagesets: Dict[str, list] = {"training": [], "validation": [], "testing": []}

        self.map_set_to_dir_idx = {"training": 0, "validation": 1, "testing": 2}
        self.map_version_to_dir = {"training": "train", "validation": "val", "testing": "test"}

        self.occlusion_map = {"NOT_OCCLUDED": 0, "PARTIALLY_OCCLUDED": 1, "MOSTLY_OCCLUDED": 2}

    def convert(self, info_prefix: str) -> None:
        """
        Start to the process of conversion.

        Args:
            info_prefix (str): The prefix of info filenames.
        """
        logging.info("A9 Conversion - start")
        for split in self.splits:
            logging.info(f"A9 Conversion - split: {split}")

            split_source_path = os.path.join(self.load_dir, self.map_version_to_dir[split])
            self._create_folder(split)

            test = True if split == "testing" else False

            pcd_list = sorted(
                glob(
                    os.path.join(
                        split_source_path,
                        "point_clouds",
                        "s110_lidar_ouster_south",
                        "*",
                    )
                )
            )

            for idx, pcd_path in enumerate(pcd_list):
                out_filename = pcd_path.split("/")[-1][:-4]

                self.convert_pcd_to_bin(
                    pcd_path, os.path.join(self.point_cloud_save_dir, out_filename)
                )
                pcd_list[idx] = os.path.join(self.point_cloud_save_dir, out_filename) + ".bin"

            # fmt: off
            img_south1_list = sorted(glob(os.path.join(split_source_path, 'images', 's110_camera_basler_south1_8mm', '*')))
            img_south2_list = sorted(glob(os.path.join(split_source_path, 'images', 's110_camera_basler_south2_8mm', '*')))
            pcd_labels_list = sorted(glob(os.path.join(split_source_path, 'labels_point_clouds', 's110_lidar_ouster_south', '*')))
            img_south1_labels_list = sorted(glob(os.path.join(split_source_path, 'labels_images', 's110_camera_basler_south1_8mm', '*')))
            img_south2_labels_list = sorted(glob(os.path.join(split_source_path, 'labels_images', 's110_camera_basler_south2_8mm', '*')))
            # fmt: on

            infos_list = self._fill_infos(
                pcd_list,
                img_south1_list,
                img_south2_list,
                pcd_labels_list,
                img_south1_labels_list,
                img_south2_labels_list,
                test,
            )

            metadata = dict(version="r1")

            if test:
                logging.info(f"No. test samples: {len(infos_list)}")
                data = dict(infos=infos_list, metadata=metadata)
                info_path = os.path.join(self.save_dir, f"{info_prefix}_infos_test.pkl")
                mmcv.dump(data, info_path)
            else:
                if split == "training":
                    logging.info(f"No. train samples: {len(infos_list)}")
                    data = dict(infos=infos_list, metadata=metadata)
                    info_path = os.path.join(self.save_dir, f"{info_prefix}_infos_train.pkl")
                    mmcv.dump(data, info_path)
                elif split == "validation":
                    logging.info(f"No. val samples: {len(infos_list)}")
                    data = dict(infos=infos_list, metadata=metadata)
                    info_path = os.path.join(self.save_dir, f"{info_prefix}_infos_val.pkl")
                    mmcv.dump(data, info_path)

        logging.info("A9 Conversion - end")

    def _fill_infos(
        self,
        pcd_list,
        img_south1_list,
        img_south2_list,
        pcd_labels_list,
        img_south1_labels_list,
        img_south2_labels_list,
        test=False,
    ) -> List[Dict[str, Any]]:
        """
        TODO write
        """
        infos_list = []

        for i, pcd_path in enumerate(pcd_list):
            json1_file = open(pcd_labels_list[i])
            json1_str = json1_file.read()
            lidar_annotation = json.loads(json1_str)

            lidar_anno_frame = {}

            for frame_idx in lidar_annotation["openlabel"]["frames"]:
                lidar_anno_frame = lidar_annotation["openlabel"]["frames"][frame_idx]

            info = {
                "lidar_path": pcd_path,
                "lidar_anno_path": pcd_labels_list[i],
                "sweeps": [],
                "cams": dict(),
                "lidar2ego": lidar2ego,
                "timestamp": lidar_anno_frame["frame_properties"]["timestamp"],
                "location": lidar_anno_frame["frame_properties"]["point_cloud_file_names"][0].split(
                    "_"
                )[2],
            }

            json2_file = open(img_south1_labels_list[i])
            json2_str = json2_file.read()
            south1_annotation = json.loads(json2_str)

            south1_anno_frame = {}

            for k in south1_annotation["openlabel"]["frames"]:
                south1_anno_frame = south1_annotation["openlabel"]["frames"][k]

            img_south1_info = {
                "data_path": img_south1_list[i],
                "type": "s110_camera_basler_south1_8mm",
                "lidar2image": lidar2s1image,
                "sensor2ego": south12ego,
                "sensor2lidar": south12lidar,
                "camera_intrinsics": south1intrinsics,
                "timestamp": south1_anno_frame["frame_properties"]["timestamp"],
            }

            info["cams"].update({"s110_camera_basler_south1_8mm": img_south1_info})

            json3_file = open(img_south2_labels_list[i])
            json3_str = json3_file.read()
            south2_annotation = json.loads(json3_str)

            south2_anno_frame = {}

            for frame_idx in south2_annotation["openlabel"]["frames"]:
                south2_anno_frame = south2_annotation["openlabel"]["frames"][frame_idx]

            img_south2_info = {
                "data_path": img_south2_list[i],
                "type": "s110_camera_basler_south2_8mm",
                "lidar2image": lidar2s2image,
                "sensor2ego": south22ego,
                "sensor2lidar": south22lidar,
                "camera_intrinsics": south2intrinsics,
                "timestamp": south2_anno_frame["frame_properties"]["timestamp"],
            }

            info["cams"].update({"s110_camera_basler_south2_8mm": img_south2_info})

            # obtain annotation

            if not test:
                gt_boxes = []
                gt_names = []
                velocity = []
                valid_flag = []
                num_lidar_pts = []
                num_radar_pts = []

                for id in lidar_anno_frame["objects"]:
                    object_data = lidar_anno_frame["objects"][id]["object_data"]

                    loc = np.asarray(object_data["cuboid"]["val"][:3], dtype=np.float32)
                    dim = np.asarray(object_data["cuboid"]["val"][7:], dtype=np.float32)
                    rot = np.asarray(
                        object_data["cuboid"]["val"][3:7], dtype=np.float32
                    )  # Quaternion in x,y,z,w

                    rot_temp = Rotation.from_quat(rot)
                    rot_temp = rot_temp.as_euler("xyz", degrees=False)

                    yaw = np.asarray(rot_temp[2], dtype=np.float32)

                    gt_box = np.concatenate([loc, dim, -yaw], axis=None)

                    gt_boxes.append(gt_box)
                    gt_names.append(object_data["type"])
                    velocity.append([0, 0])
                    valid_flag.append(True)

                    for n in object_data["cuboid"]["attributes"]["num"]:
                        if n["name"] == "num_points":
                            num_lidar_pts.append(n["val"])

                    num_radar_pts.append(0)

                gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
                info["gt_boxes"] = gt_boxes
                info["gt_names"] = np.array(gt_names)
                info["gt_velocity"] = np.array(velocity).reshape(-1, 2)
                info["num_lidar_pts"] = np.array(num_lidar_pts)
                info["num_radar_pts"] = np.array(num_radar_pts)
                info["valid_flag"] = np.array(valid_flag, dtype=bool)

            infos_list.append(info)

        return infos_list

    @staticmethod
    def convert_pcd_to_bin(file: str, out_file: str) -> None:
        """
        Convert file from .pcd to .bin

        Args:
            file: Filepath to .pcd
            out_file: Filepath of .bin
        """
        point_cloud = pypcd.PointCloud.from_path(file)
        np_x = np.array(point_cloud.pc_data["x"], dtype=np.float32)
        np_y = np.array(point_cloud.pc_data["y"], dtype=np.float32)
        np_z = np.array(point_cloud.pc_data["z"], dtype=np.float32)
        np_i = np.array(point_cloud.pc_data["intensity"], dtype=np.float32) / 256
        np_ts = np.zeros((np_x.shape[0],), dtype=np.float32)

        bin_format = np.column_stack((np_x, np_y, np_z, np_i, np_ts)).flatten()
        bin_format.tofile(os.path.join(f"{out_file}.bin"))

    @staticmethod
    def cp_img(file: str, out_file: str) -> None:
        """
        Copy image to new location

        Args:
            file: Path to image
            out_file: Path to new location
        """
        img_path = f"{out_file}.jpg"
        shutil.copyfile(file, img_path)

    def _create_folder(self, split: str) -> None:
        """
        Create folder for data preprocessing.
        """
        split_path = self.map_version_to_dir[split]
        logging.info(f"Creating folder - split_path: {split_path}")
        dir_list1 = ["point_clouds/s110_lidar_ouster_south"]
        for d in dir_list1:
            self.point_cloud_save_dir = os.path.join(self.save_dir, split_path, d)
            logging.info(f"Creating folder : {self.point_cloud_save_dir}")
            os.makedirs(self.point_cloud_save_dir, exist_ok=True, mode=0o777)