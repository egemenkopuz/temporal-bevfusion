import json
import logging
import multiprocessing.pool as mpp
import os
import re
from collections import defaultdict
from glob import glob
from typing import Any, Dict, List, Optional

import mmcv
import numpy as np
from PIL import Image
from pypcd import pypcd
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# projecton matrices
lidar2rgb_center = np.asarray(
    [
        [1.28155029e03, -4.58555989e03, -1.02967448e01, -6.88197425e01],
        [8.46420682e02, 2.22640290e01, -4.58648857e03, 9.36547118e03],
        [9.99968067e-01, 5.41023377e-03, 5.88174706e-03, -7.90217233e-02],
    ],
    dtype=np.float32,
)
lidar2rgb_highres_center = np.asarray(
    [
        [2.11443983e03, -7.18523532e03, -1.89987995e02, -1.89435698e03],
        [4.14791337e02, -3.75379653e01, -7.29679085e03, 2.55646787e04],
        [9.93614086e-01, 9.29010435e-03, -1.12448838e-01, 3.18121585e-01],
    ],
    dtype=np.float32,
)
lidar2rgb_left = np.asarray(
    [
        [2.73611511e03, -3.90615474e03, 2.24407451e01, 6.00999552e02],
        [8.82958038e02, 3.26109605e02, -4.58687996e03, 9.34507256e03],
        [9.39295629e-01, 3.42440329e-01, 2.14089134e-02, -1.36006017e-01],
    ],
    dtype=np.float32,
)
lidar2rgb_highres_left = np.asarray(
    [
        [4.60709278e03, -5.91552604e03, -2.53021286e02, -1.99736624e02],
        [2.98383823e02, 1.12462909e02, -7.29581083e03, 2.56435774e04],
        [9.21559517e-01, 3.68083432e-01, -1.23461101e-01, 4.28171490e-01],
    ],
    dtype=np.float32,
)
lidar2rgb_right = np.asarray(
    [
        [-5.15649958e02, -4.73024432e03, -6.19729395e01, -7.77740065e02],
        [8.44808580e02, -2.35699427e02, -4.57878001e03, 9.33346125e03],
        [9.31990264e-01, -3.61951168e-01, 1.96341615e-02, -1.59558444e-01],
    ],
    dtype=np.float32,
)
lidar2rgb_highres_right = np.asarray(
    [
        [-6.60454783e02, -7.46284041e03, -4.18444286e02, -2.41693304e03],
        [2.89901993e02, 3.94107215e01, -7.30206100e03, 2.56819982e04],
        [9.25346812e-01, -3.54911295e-01, -1.33308850e-01, 2.34974919e-01],
    ],
    dtype=np.float32,
)

# intrinsic matrices
rgb_highres_center_intrinsic = np.array(
    [
        [7267.95450880415, 0.0, 2056.049238502414],
        [0.0, 7267.95450880415, 1232.862908875167],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
rgb_highres_left_intrinsic = np.array(
    [
        [7265.09513308939, 0.0, 2099.699693520321],
        [0.0, 7265.09513308939, 1217.709330768128],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
rgb_highres_right_intrinsic = np.array(
    [
        [7265.854580654392, 0.0, 2093.506452810741],
        [0.0, 7265.854580654392, 1228.255759518024],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
rgb_center_intrinsic = np.array(
    [
        [4609.471892628096, 0.0, 1257.158605934],
        [0.0, 4609.471892628096, 820.0498076210201],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
rgb_left_intrinsic = np.array(
    [
        [4622.041473915607, 0.0, 1233.380196060109],
        [0.0, 4622.041473915607, 843.3909933480334],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
rgb_right_intrinsic = np.array(
    [
        [4613.465257442901, 0.0, 1230.818284106724],
        [0.0, 4613.465257442901, 783.2495217495479],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

# lidar to ego
lidar2ego = np.identity(4)[:-1, :]

# cameras to ego
rgb_highres_center2ego = np.array(
    [
        [0.00999566, -0.11238832, 0.99361409, 0.0801578],
        [-0.99993373, -0.00679984, 0.0092901, -0.332862],
        [0.00571232, -0.9936411, -0.11244884, 3.50982],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)[:-1, :]
rgb_highres_left2ego = np.array(
    [
        [3.71020361e-01, -1.14332619e-01, 9.21559517e-01, 6.09096000e-02],
        [-9.28624333e-01, -4.65986540e-02, 3.68083432e-01, -1.36682000e-01],
        [8.59490503e-04, -9.92349040e-01, -1.23461101e-01, 3.51522000e00],
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
    ],
    dtype=np.float32,
)[:-1, :]
rgb_highres_right2ego = np.array(
    [
        [-0.36046879, -0.11745438, 0.92534681, 0.0510683],
        [-0.93257067, 0.06595395, -0.3549113, -0.525541],
        [-0.01934439, -0.99088574, -0.13330885, 3.51628],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)[:-1, :]
rgb_center2ego = np.array(
    [
        [5.43294099e-03, 5.86077897e-03, 9.99968067e-01, 6.69458000e-02],
        [-9.99977824e-01, 3.88335856e-03, 5.41023377e-03, -9.11152000e-04],
        [-3.85152637e-03, -9.99975285e-01, 5.88174706e-03, 2.05432000e00],
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
    ],
    dtype=np.float32,
)[:-1, :]
rgb_left2ego = np.array(
    [
        [3.42535973e-01, 1.98198820e-02, 9.39295629e-01, 2.98925000e-02],
        [-9.39504322e-01, 8.13943312e-03, 3.42440329e-01, 1.86612000e-01],
        [-8.58207043e-04, -9.99770435e-01, 2.14089134e-02, 2.05637000e00],
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
    ],
    dtype=np.float32,
)[:-1, :]
rgb_right2ego = np.array(
    [
        [-0.36161439, 0.02508347, 0.93199026, 0.051346],
        [-0.93213946, 0.01036222, -0.36195117, -0.196979],
        [-0.01873648, -0.99963165, 0.01963416, 2.05803],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)[:-1, :]

# cameras to lidar
rgb_highres_center2lidar = rgb_highres_center2ego.copy()
rgb_highres_left2lidar = rgb_highres_left2ego.copy()
rgb_highres_right2lidar = rgb_highres_right2ego.copy()
rgb_center2lidar = rgb_center2ego.copy()
rgb_left2lidar = rgb_left2ego.copy()
rgb_right2lidar = rgb_right2ego.copy()


class OSDAR23Converter:
    """OSDAR23 dataset converter.

    This class serves as the converter to change the OSDAR23 data to custom format.
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
            load_dir (str): Directory to load OSDAR23 raw data.
            save_dir (str): Directory to save data.
            labels_path (str): Path of labels.
            images_path (str): Path of images.
            num_workers (int): Number of workers to process data.
        """

        self.splits = splits
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.labels_path = labels_path if labels_path else os.path.join(self.load_dir, "lidar")
        self.images_path = images_path if images_path else os.path.join(self.load_dir, "images")
        self.point_cloud_foldername = "lidar"
        self.images_foldername = "images"
        self.num_workers = num_workers

        self.imagesets: Dict[str, list] = {x: [] for x in splits}

        self.map_set_to_dir_idx = {"training": 0, "validation": 1, "testing": 2}
        self.map_version_to_dir = {"training": "train", "validation": "val", "testing": "test"}

        self.distance_map = {
            "0-49": 0,
            "50-99": 1,
            "100-149": 2,
            "150-199": 3,
            "200-inf": 4,
        }

        self.num_points_map = {
            "0-199": 0,
            "200-499": 1,
            "500-999": 2,
            "1000-1999": 3,
            "2000-2999": 4,
            "3000-inf": 5,
        }

        self.occlusion_map = {"0-25 %": 0, "25-50 %": 1, "50-75 %": 2, "75-99 %": 3, "100 %": 4}

    def convert(self, info_prefix: str, use_highres: bool = False) -> None:
        """
        Start to the process of conversion.

        Args:
            info_prefix (str): The prefix of info filenames.
            use_highres (bool): Use highres images.
        """
        logging.info("OSDAR23 Conversion - start")
        for split in self.splits:
            logging.info(f"OSDAR23 Conversion - split: {split}")

            split_source_path = os.path.join(self.load_dir, self.map_version_to_dir[split])
            splt_target_path = os.path.join(self.save_dir, self.map_version_to_dir[split])
            os.makedirs(splt_target_path, exist_ok=True, mode=0o777)

            test = True if split == "testing" else False

            pcd_list = sorted(
                glob(os.path.join(split_source_path, self.point_cloud_foldername, "*")),
                key=natural_key,
            )

            os.makedirs(
                os.path.join(splt_target_path, self.point_cloud_foldername),
                exist_ok=True,
                mode=0o777,
            )

            if use_highres:
                img_types = ["rgb_highres_center", "rgb_highres_left", "rgb_highres_right"]
            else:
                img_types = ["rgb_center", "rgb_left", "rgb_right"]

            img_lists = defaultdict(list)
            for img_type in img_types:
                img_lists[img_type].extend(
                    sorted(
                        glob(
                            os.path.join(split_source_path, self.images_foldername, img_type, "*")
                        ),
                        key=natural_key,
                    )
                )
                os.makedirs(
                    os.path.join(splt_target_path, self.images_foldername, img_type),
                    exist_ok=True,
                    mode=0o777,
                )

            os.makedirs(
                os.path.join(splt_target_path, "labels_point_clouds"),
                exist_ok=True,
                mode=0o777,
            )

            # convert pcd to bin and transfer them
            # logging.info("Converting pcd files to bin files")
            # with mpp.Pool(processes=self.num_workers) as pool:
            #     pool.starmap_async(
            #         self.convert_pcd_to_bin,
            #         [
            #             (
            #                 x,
            #                 os.path.join(
            #                     splt_target_path, self.point_cloud_foldername, x.split("/")[-1][:-4]
            #                 ),
            #             )
            #             for x in pcd_list
            #         ],
            #     )
            #     pool.close()
            #     pool.join()

            # # convert images to jpg and transfer them
            # for img_type, img_details in img_lists.items():
            #     logging.info(f"Converting {img_type} files to jpg files")
            #     for img in tqdm(img_details, total=len(img_details), desc=f"{img_type} files"):
            #         self.convert_png_to_jpg(
            #             img,
            #             os.path.join(
            #                 splt_target_path,
            #                 self.images_foldername,
            #                 img_type,
            #                 img.split("/")[-1][:-4],
            #             ),
            #         )

            # for img_type, img_details in img_lists.items():
            #     img_lists[img_type] = [
            #         os.path.join(
            #             splt_target_path, self.images_foldername, img_type, x.split("/")[-1][:-4]
            #         )
            #         + ".jpg"
            #         for x in img_details
            #     ]

            # transfer pcd labels
            logging.info("Transferring pcd labels")
            pcd_labels_list = sorted(
                glob(os.path.join(split_source_path, "labels_point_clouds", "*")),
                key=natural_key,
            )
            for pcd_label in tqdm(pcd_labels_list, total=len(pcd_labels_list), desc="pcd labels"):
                os.system(f"cp {pcd_label} {os.path.join(splt_target_path, 'labels_point_clouds')}")

            pcd_list = [
                os.path.join(splt_target_path, self.point_cloud_foldername, x.split("/")[-1][:-4])
                + ".bin"
                for x in pcd_list
            ]

            pcd_labels_list = sorted(
                glob(os.path.join(splt_target_path, "labels_point_clouds", "*")),
                key=natural_key,
            )

            infos_list = self._fill_infos(pcd_list, img_lists, pcd_labels_list, False)
            metadata = dict(version="1.0.0")

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

        logging.info("OSDAR23 Conversion - end")

    def _fill_infos(
        self,
        pcd_list,
        img_lists,
        pcd_labels_list,
        test=False,
    ) -> List[Dict[str, Any]]:
        infos_list = []

        for i, pcd_path in tqdm(enumerate(pcd_list), total=len(pcd_list), desc="fill infos"):
            with open(pcd_labels_list[i], "rb") as f:
                lidar_annotation = json.load(f)

            lidar_anno_frame = {}

            frame_idx = list(lidar_annotation["openlabel"]["frames"].keys())[0]
            lidar_anno_frame = lidar_annotation["openlabel"]["frames"][frame_idx]
            lidar_stream = lidar_anno_frame["frame_properties"]["streams"]["lidar"]

            info = {
                "lidar_path": pcd_path,
                "lidar_anno_path": pcd_labels_list[i],
                "sweeps": [],
                "cams": dict(),
                "lidar2ego": lidar2ego,
                "timestamp": lidar_stream["stream_properties"]["sync"]["timestamp"],
                # temporal related stuff
                "token": lidar_anno_frame["frame_properties"]["token"],
                "scene_token": lidar_anno_frame["frame_properties"]["scene_token"],
                "frame_idx": lidar_anno_frame["frame_properties"]["frame_idx"],
                "prev": lidar_anno_frame["frame_properties"]["prev"],
                "next": lidar_anno_frame["frame_properties"]["next"],
            }

            for img_type, img_details in img_lists.items():
                img_info = {
                    "data_path": img_details[i],
                    "type": img_type,
                    "lidar2image": eval(f"lidar2{img_type}"),
                    "sensor2ego": eval(f"{img_type}2ego"),
                    "sensor2lidar": eval(f"{img_type}2lidar"),
                    "camera_intrinsics": eval(f"{img_type}_intrinsic"),
                    "timestamp": None,
                }
                info["cams"].update({img_type: img_info})

            if not test:
                gt_boxes = []
                gt_names = []
                valid_flag = []
                num_lidar_pts = []
                num_radar_pts = []
                distances = []
                occlusions = []
                distance_levels = []
                num_points_levels = []

                for id in lidar_anno_frame["objects"]:
                    object_data = lidar_anno_frame["objects"][id]["object_data"]

                    loc = np.asarray(object_data["cuboid"][0]["val"][:3], dtype=np.float32)
                    dim = np.asarray(object_data["cuboid"][0]["val"][7:], dtype=np.float32)
                    rot = np.asarray(
                        object_data["cuboid"][0]["val"][3:7], dtype=np.float32
                    )  # quaternion in x,y,z,w
                    rot_temp = Rotation.from_quat(rot)
                    rot_temp = rot_temp.as_euler("xyz", degrees=False)
                    yaw = np.asarray(rot_temp[2], dtype=np.float32)
                    gt_box = np.concatenate([loc, dim, -yaw], axis=None)

                    gt_boxes.append(gt_box)
                    gt_names.append(object_data["cuboid"][0]["name"])
                    valid_flag.append(True)

                    for n in object_data["cuboid"][0]["attributes"]["num"]:
                        if n["name"] == "num_points":
                            num_lidar_pts.append(n["val"])
                        elif n["name"] == "distance":
                            distances.append(n["val"])
                    for n in object_data["cuboid"][0]["attributes"]["text"]:
                        if n["name"] == "occlusion":
                            occlusions.append(self.occlusion_map[n["val"]])
                        elif n["name"] == "distance_level":
                            distance_levels.append(self.distance_map[n["val"]])
                        elif n["name"] == "num_points_level":
                            num_points_levels.append(self.num_points_map[n["val"]])

                gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
                info["gt_boxes"] = gt_boxes
                info["gt_names"] = np.array(gt_names)
                info["num_lidar_pts"] = np.array(num_lidar_pts)
                info["num_radar_pts"] = np.array(num_radar_pts)
                info["distances"] = np.array(distances)
                info["occlusions"] = np.array(occlusions)
                info["distance_levels"] = np.array(distance_levels)
                info["num_points_levels"] = np.array(num_points_levels)
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
        np_ts = np.array(point_cloud.pc_data["timestamp"], dtype=np.float32)
        np_sensor_index = np.array(point_cloud.pc_data["sensor_index"], dtype=np.float32)

        bin_format = np.column_stack((np_x, np_y, np_z, np_i, np_ts, np_sensor_index)).flatten()
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


def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_) if s]
