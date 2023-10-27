from argparse import ArgumentParser, Namespace
from typing import List

import torch


def get_args() -> Namespace:
    """
    Parse given arguments for create_swint_checkpoint function.

    Returns:
        Namespace: parsed arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-l", type=str, required=True, help="path to lidar model")
    parser.add_argument("-c", type=str, required=True, help="path to camera model")
    parser.add_argument("-t", type=str, required=True, help="path to save target model")

    return parser.parse_args()


def convert_to_combined_pth(
    lidar_model_path: str,
    camera_model_path: str,
    target_save_path: str,
    camera_prefixes: List[str] = [
        "encoders.camera.backbone",
        "encoders.camera.vtransform",
        "encoders.camera.neck",
    ],
    blacklist_prefixes: List[str] = [
        "temporal_fuser",
    ],
) -> None:
    lidar_model = torch.load(lidar_model_path, map_location=torch.device("cpu"))
    camera_model = torch.load(camera_model_path, map_location=torch.device("cpu"))

    print("total keys in lidar model", len(lidar_model["state_dict"].keys()))
    print("total keys in camera model", len(camera_model["state_dict"].keys()))

    camera_keys = []
    for x in camera_model["state_dict"].keys():
        for prefix in camera_prefixes:
            if x.startswith(prefix):
                camera_keys.append(x)
                break

    # create a new state dict
    new_state_dict = {}
    for key, value in lidar_model["state_dict"].items():
        skip = False
        for x in blacklist_prefixes:
            if key.startswith(x):
                skip = True
                break
        if not skip:
            new_state_dict[key] = value

    for key, value in camera_model["state_dict"].items():
        if key in camera_keys and key not in blacklist_prefixes:
            new_state_dict[key] = value

    print("total keys in new state dict", len(new_state_dict.keys()))
    for x in new_state_dict:
        print(x)

    # save the new state dict
    lidar_model["state_dict"] = new_state_dict
    torch.save(lidar_model, target_save_path)


if __name__ == "__main__":
    args = get_args()
    convert_to_combined_pth(
        lidar_model_path=args.l,
        camera_model_path=args.c,
        target_save_path=args.t,
    )
