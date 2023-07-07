import json
import os
from argparse import ArgumentParser, Namespace
from glob import glob

import matplotlib as mpl
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

from .utils.geometry import add_open3d_axis, visualize_bounding_box


def get_args() -> Namespace:
    """
    Parse given arguments for a9_plot_3d_boxes function.

    Returns:
        Namespace: parsed arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-p", "--point_clouds_folder_path", type=str, required=True)
    parser.add_argument("-d", "--detections_folder_path", type=str, required=True)
    parser.add_argument("-i", "--index", type=int, required=False, default=0)
    parser.add_argument("--point_size", type=int, required=False, default=2)
    parser.add_argument("--enable_coord_frame", type=bool, required=False, default=False)
    parser.add_argument("--enable_color_distance", type=bool, required=False, default=False)
    parser.add_argument("--enable_color_detection", type=bool, required=False, default=False)

    return parser.parse_args()


def a9_plot_3d_boxes(
    input_folder_path_point_clouds: str,
    input_folder_path_detections: str,
    index: int = 0,
    point_size: int = 2,
    show_coordinate_frame: bool = False,
    color_distance: bool = False,
    color_detection: bool = False,
):
    file_paths_point_clouds = sorted(glob(os.path.join(input_folder_path_point_clouds, "*")))
    file_paths_detections = sorted(glob(os.path.join(input_folder_path_detections, "*")))

    assert index < len(file_paths_point_clouds) and index < len(file_paths_detections)

    file_path_point_cloud = file_paths_point_clouds[index]
    file_path_detections = file_paths_detections[index]

    pcd = o3d.io.read_point_cloud(file_path_point_cloud)
    points = np.array(pcd.points)

    print("Raw: ", points.shape)

    # remove rows having all zeroes
    points_filtered = points[~np.all(points == 0, axis=1)]

    # remove rows having z>-1.5 (for ouster lidars)
    if "ouster" in file_path_point_cloud:
        points_filtered = points_filtered[points_filtered[:, 2] < -1.5]

    # remove rows having z<-10
    points_filtered = points_filtered[points_filtered[:, 2] > -10.0]

    # remove points with distance>120
    distances = np.array(
        [np.sqrt(row[0] * row[0] + row[1] * row[1] + row[2] * row[2]) for row in points_filtered]
    )
    distances_bev = np.array(
        [np.sqrt(row[1] * row[1] + row[2] * row[2]) for row in points_filtered]
    )

    points_filtered = points_filtered[distances < 120.0]
    distances_bev = distances_bev[distances < 120.0]
    distances = distances[distances < 120.0]

    # remove points with distance<3
    points_filtered = points_filtered[distances > 3.0]
    distances_bev = distances_bev[distances > 3.0]
    distances = distances[distances > 3.0]

    # corner_point_min = np.array([-150, -150, -10])
    # corner_point_max = np.array([150, 150, 5])

    print("Filtered: ", points_filtered.shape)

    # points = np.vstack((points_filtered, corner_point_min, corner_point_max))
    points = points_filtered
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points[:, :3]))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Visualizer", width=1024, height=720)
    vis.get_render_option().background_color = [0.1, 0.1, 0.1]
    vis.get_render_option().point_size = point_size

    if show_coordinate_frame:
        add_open3d_axis(vis)

    try:
        detection_data = json.load(open(file_path_detections))

        if color_distance:
            cmap_norm = mpl.colors.Normalize(vmin=distances_bev.min(), vmax=distances_bev.max())
            colors = plt.get_cmap("jet")(cmap_norm(distances_bev))[:, 0:3]
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            pcd.paint_uniform_color([0.4, 0.4, 0.4])

        process_detections(detection_data, pcd, vis, color_detection)

        vis.add_geometry(pcd)

        vis.get_view_control().set_zoom(0.12)
        vis.get_view_control().set_front([0.22, 0.141, 0.965])
        vis.get_view_control().set_lookat([22.964, -2.772, -7.230])
        vis.get_view_control().set_up([0.969, -0.148, -0.200])

        vis.run()
    except Exception as e:
        print(e)
        vis.destroy_window()


def process_detections(label_data, pcd, vis, color_detection: bool = False):
    for frame_id, frame_obj in label_data["openlabel"]["frames"].items():
        for object_id, label in frame_obj["objects"].items():
            l = float(label["object_data"]["cuboid"]["val"][7])
            w = float(label["object_data"]["cuboid"]["val"][8])
            h = float(label["object_data"]["cuboid"]["val"][9])
            quat_x = float(label["object_data"]["cuboid"]["val"][3])
            quat_y = float(label["object_data"]["cuboid"]["val"][4])
            quat_z = float(label["object_data"]["cuboid"]["val"][5])
            quat_w = float(label["object_data"]["cuboid"]["val"][6])
            rotation_yaw = R.from_quat([quat_x, quat_y, quat_z, quat_w]).as_euler(
                "zyx", degrees=False
            )[0]
            position_3d = [
                float(label["object_data"]["cuboid"]["val"][0]),
                float(label["object_data"]["cuboid"]["val"][1]),
                float(label["object_data"]["cuboid"]["val"][2]),
            ]
            category = label["object_data"]["type"].upper()

            obb = o3d.geometry.OrientedBoundingBox(
                position_3d,
                np.array(
                    [
                        [np.cos(rotation_yaw), -np.sin(rotation_yaw), 0],
                        [np.sin(rotation_yaw), np.cos(rotation_yaw), 0],
                        [0, 0, 1],
                    ]
                ),
                np.array([l, w, h]),
            )
            if color_detection:
                colors = np.array(pcd.colors)
                indices = obb.get_point_indices_within_bounding_box(pcd.points)
                base_color = (250, 27, 27)
                colors[indices, :] = np.array(
                    [base_color[0] / 255, base_color[1] / 255, base_color[2] / 255]
                )
                pcd.colors = o3d.utility.Vector3dVector(colors)

            visualize_bounding_box(
                l, w, h, rotation_yaw, position_3d, category, vis, None, None, "a9"
            )


if __name__ == "__main__":
    args = get_args()
    a9_plot_3d_boxes(
        input_folder_path_point_clouds=args.input_folder_path_point_clouds,
        input_folder_path_detections=args.input_folder_path_detections,
        index=args.index,
        point_size=args.point_size,
        show_coordinate_frame=args.show_coordinate_frame,
        color_distance=args.color_distance,
        color_detection=args.color_detection,
    )
