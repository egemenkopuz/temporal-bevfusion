import json
import logging
import multiprocessing.pool as mpp
import os
import shutil
from collections import defaultdict
from glob import glob
from typing import Any, Dict, List, Optional

import mmcv
import numpy as np
from PIL import Image
from pypcd import pypcd
from scipy.spatial.transform import Rotation
from tqdm import tqdm

lidar2ego = np.asarray(
    [
        [0.99011437, -0.13753536, -0.02752358, 2.3728100375737995],
        [0.13828977, 0.99000475, 0.02768645, -16.19297517556697],
        [0.02344061, -0.03121898, 0.99923766, -8.620000000000005],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)[:-1, :]

lidar2south1 = np.asarray(
    [
        [7.04216073e02, -1.37317442e03, -4.32235765e02, -2.03369364e04],
        [-9.28351327e01, -1.77543929e01, -1.45629177e03, 9.80290034e02],
        [8.71736000e-01, -9.03453000e-02, -4.81574000e-01, -2.58546000e00],
    ],
    dtype=np.float32,
)

lidar2south2 = np.asarray(
    [
        [1546.63215008, -436.92407115, -295.58362676, 1319.79271737],
        [93.20805656, 47.90351592, -1482.13403199, 687.84781276],
        [0.73326062, 0.59708904, -0.32528854, -1.30114325],
    ],
    dtype=np.float32,
)

south1_intrinsic = np.asarray(
    [
        [1400.3096617691212, 0.0, 967.7899705163408],
        [0.0, 1403.041082755918, 581.7195041357244],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

south12ego = np.asarray(
    [
        [-0.06377762, -0.91003007, 0.15246652, -10.409943],
        [-0.41296193, -0.10492031, -0.8399004, -16.2729],
        [0.8820865, -0.11257353, -0.45447016, -11.557314],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)[:-1, :]

south12lidar = np.asarray(
    [
        [-0.10087585, -0.51122875, 0.88484734, 1.90816304],
        [-1.0776537, 0.03094424, -0.10792235, -14.05913251],
        [0.01956882, -0.93122171, -0.45454375, 0.72290242],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)[:-1, :]

south2_intrinsic = np.asarray(
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
)[:-1, :]

south22lidar = np.asarray(
    [
        [0.49709212, -0.19863714, 0.64202357, -0.03734614],
        [-0.60406415, -0.17852863, 0.50214409, 2.52095055],
        [0.01173726, -0.77546627, -0.70523436, 0.54322305],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)[:-1, :]


class TUMTrafIntersectionConverter:
    """TUMTraf-I dataset converter.

    This class serves as the converter to change the TUMTraf-I data to custom format.
    """

    def __init__(
        self,
        splits: List[str],
        load_dir: str,
        save_dir: str,
        labels_path: Optional[str] = None,
        images_path: Optional[str] = None,
        num_workers: int = 4,
    ):
        """
        Args:
            splits list[(str)]: Contains the different splits
            version (str): Specify the modality
            load_dir (str): Directory to load TUMTraf-I raw data.
            save_dir (str): Directory to save data.
            labels_path (str): Path of labels.
            num_workers (int): Number of workers to use. Default: 4
        """

        self.splits = splits
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.labels_path = (
            labels_path if labels_path else os.path.join(self.load_dir, "point_clouds")
        )
        self.images_path = images_path if images_path else os.path.join(self.load_dir, "images")
        self.point_cloud_foldername = "point_clouds"
        self.images_foldername = "images"
        self.num_workers = num_workers

        self.imagesets: Dict[str, list] = {x: [] for x in splits}

        self.map_set_to_dir_idx = {"training": 0, "validation": 1, "testing": 2}
        self.map_version_to_dir = {"training": "train", "validation": "val", "testing": "test"}

        self.difficulty_map = {"easy": 0, "moderate": 1, "hard": 2}

    def convert(self, info_prefix: str) -> None:
        """
        Start to the process of conversion.

        Args:
            info_prefix (str): The prefix of info filenames.
        """
        logging.info("TUMTraf-I Conversion - start")
        for split in self.splits:
            logging.info(f"TUMTraf-I Conversion - split: {split}")

            split_source_path = os.path.join(self.load_dir, self.map_version_to_dir[split])
            splt_target_path = os.path.join(self.save_dir, self.map_version_to_dir[split])
            os.makedirs(splt_target_path, exist_ok=True, mode=0o777)

            test = True if split == "testing" else False

            pcd_list = sorted(
                glob(
                    os.path.join(
                        split_source_path,
                        self.point_cloud_foldername,
                        "s110_lidar_ouster_south",
                        "*",
                    )
                )
            )

            os.makedirs(
                os.path.join(
                    splt_target_path, self.point_cloud_foldername, "s110_lidar_ouster_south"
                ),
                exist_ok=True,
                mode=0o777,
            )

            img_types = ["s110_camera_basler_south1_8mm", "s110_camera_basler_south2_8mm"]

            img_lists = defaultdict(list)
            for img_type in img_types:
                img_lists[img_type] = sorted(
                    glob(os.path.join(split_source_path, self.images_foldername, img_type, "*"))
                )

                os.makedirs(
                    os.path.join(splt_target_path, self.images_foldername, img_type),
                    exist_ok=True,
                    mode=0o777,
                )

            os.makedirs(
                os.path.join(splt_target_path, "labels_point_clouds", "s110_lidar_ouster_south"),
                exist_ok=True,
                mode=0o777,
            )

            # convert pcd to bin and transfer them
            logging.info("Converting pcd files to bin files")
            with mpp.Pool(processes=self.num_workers) as pool:
                pool.starmap_async(
                    self.convert_pcd_to_bin,
                    [
                        (
                            x,
                            os.path.join(
                                splt_target_path,
                                self.point_cloud_foldername,
                                "s110_lidar_ouster_south",
                                x.split("/")[-1][:-4],
                            ),
                        )
                        for x in pcd_list
                    ],
                )
                pool.close()
                pool.join()

            # convert images to jpg and transfer them
            for img_type, img_details in img_lists.items():
                logging.info(f"Converting {img_type} files to jpg files")
                for img in tqdm(img_details, total=len(img_details), desc=f"{img_type} files"):
                    self.convert_png_to_jpg(
                        img,
                        os.path.join(
                            splt_target_path,
                            self.images_foldername,
                            img_type,
                            img.split("/")[-1][:-4],
                        ),
                    )

            for img_type, img_details in img_lists.items():
                img_lists[img_type] = [
                    os.path.join(
                        splt_target_path, self.images_foldername, img_type, x.split("/")[-1][:-4]
                    )
                    + ".jpg"
                    for x in img_details
                ]

            # transfer pcd labels
            logging.info("Transferring pcd labels")
            pcd_labels_list = sorted(
                glob(
                    os.path.join(
                        split_source_path, "labels_point_clouds", "s110_lidar_ouster_south", "*"
                    )
                )
            )
            for pcd_label in tqdm(pcd_labels_list, total=len(pcd_labels_list), desc="pcd labels"):
                os.system(
                    f"cp {pcd_label} {os.path.join(splt_target_path,'labels_point_clouds', 's110_lidar_ouster_south')}"
                )

            pcd_list = [
                os.path.join(
                    splt_target_path,
                    self.point_cloud_foldername,
                    "s110_lidar_ouster_south",
                    x.split("/")[-1][:-4],
                )
                + ".bin"
                for x in pcd_list
            ]

            pcd_labels_list = sorted(
                glob(
                    os.path.join(
                        splt_target_path, "labels_point_clouds", "s110_lidar_ouster_south", "*"
                    )
                )
            )

            infos_list = self._fill_infos(pcd_list, img_lists, pcd_labels_list, False)
            metadata = dict(version="r2")

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

        logging.info("TUMTraf-I Conversion - end")

    def _fill_infos(
        self,
        pcd_list,
        img_lists,
        pcd_labels_list,
        test=False,
    ) -> List[Dict[str, Any]]:
        """
        TODO write
        """
        infos_list = []

        for i, pcd_path in tqdm(enumerate(pcd_list), total=len(pcd_list), desc="fill infos"):
            with open(pcd_labels_list[i], "rb") as f:
                lidar_annotation = json.load(f)

            lidar_anno_frame = {}

            frame_idx = list(lidar_annotation["openlabel"]["frames"].keys())[0]
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
                # temporal related stuff
                "token": lidar_anno_frame["frame_properties"]["token"],
                "scene_token": lidar_anno_frame["frame_properties"]["scene_token"],
                "frame_idx": lidar_anno_frame["frame_properties"]["frame_idx"],
                "prev": lidar_anno_frame["frame_properties"]["prev"],
                "next": lidar_anno_frame["frame_properties"]["next"],
            }

            for img_type, img_details in img_lists.items():
                if img_type == "s110_camera_basler_south1_8mm":
                    img_type_short = "south1"
                elif img_type == "s110_camera_basler_south2_8mm":
                    img_type_short = "south2"
                else:
                    raise ValueError(f"Unknown image type: {img_type}")

                img_info = {
                    "data_path": img_details[i],
                    "type": img_type,
                    "lidar2image": eval(f"lidar2{img_type_short}"),
                    "sensor2ego": eval(f"{img_type_short}2ego"),
                    "sensor2lidar": eval(f"{img_type_short}2lidar"),
                    "camera_intrinsics": eval(f"{img_type_short}_intrinsic"),
                    "timestamp": None,
                }
                info["cams"].update({img_type: img_info})

            if not test:
                gt_boxes = []
                gt_names = []
                valid_flag = []
                num_lidar_pts = []
                num_radar_pts = []
                difficulties = []
                distances = []

                for id in lidar_anno_frame["objects"]:
                    object_data = lidar_anno_frame["objects"][id]["object_data"]

                    loc = np.asarray(object_data["cuboid"]["val"][:3], dtype=np.float32)
                    dim = np.asarray(object_data["cuboid"]["val"][7:], dtype=np.float32)
                    rot = np.asarray(
                        object_data["cuboid"]["val"][3:7], dtype=np.float32
                    )  # quaternion in x,y,z,w

                    distance = np.sqrt(np.sum(np.array(loc[:2]) ** 2))
                    distances.append(distance)

                    rot_temp = Rotation.from_quat(rot)
                    rot_temp = rot_temp.as_euler("xyz", degrees=False)

                    yaw = np.asarray(rot_temp[2], dtype=np.float32)

                    gt_box = np.concatenate([loc, dim, -yaw], axis=None)

                    gt_boxes.append(gt_box)
                    gt_names.append(object_data["type"])
                    valid_flag.append(True)

                    for n in object_data["cuboid"]["attributes"]["num"]:
                        if n["name"] == "num_points":
                            num_lidar_pts.append(n["val"])
                    for n in object_data["cuboid"]["attributes"]["text"]:
                        if n["name"] == "difficulty":
                            difficulties.append(self.difficulty_map[n["val"]])

                    num_radar_pts.append(0)

                gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
                info["gt_boxes"] = gt_boxes
                info["gt_names"] = np.array(gt_names)
                info["num_lidar_pts"] = np.array(num_lidar_pts)
                info["num_radar_pts"] = np.array(num_radar_pts)
                info["difficulties"] = np.array(difficulties)
                info["distances"] = np.array(distances)
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
    def convert_png_to_jpg(file: str, out_file: str) -> None:
        """
        Convert file from .png to .jpg

        Args:
            file: Filepath to .png
            out_file: Filepath of .jpg
        """
        img = Image.open(file).convert("RGB")
        with open(out_file + ".jpg", "w") as f:
            img.save(f, "JPEG")
