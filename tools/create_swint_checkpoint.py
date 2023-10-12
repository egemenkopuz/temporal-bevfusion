from argparse import ArgumentParser, Namespace

import torch


def get_args() -> Namespace:
    """
    Parse given arguments for create_swint_checkpoint function.

    Returns:
        Namespace: parsed arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-m", type=str, required=True, help="path to pretrained swint")
    parser.add_argument("-s", type=str, required=True, help="path to source model")
    parser.add_argument("-t", type=str, required=True, help="path to save target model")

    return parser.parse_args()


def convert_to_swint_pth(
    pretrained_swint_path: str,
    source_model_path: str,
    target_save_path: str,
    prefix: str = "encoders.camera.backbone",
) -> None:
    pretrained_swint = torch.load(pretrained_swint_path, map_location=torch.device("cpu"))
    source_model = torch.load(source_model_path, map_location=torch.device("cpu"))

    print("total keys in pretrained swint", len(pretrained_swint["state_dict"].keys()))
    print("total keys in source model", len(source_model["state_dict"].keys()))

    common_keys = []
    other_keys = []

    for x in source_model["state_dict"].keys():
        if x.startswith(prefix):
            common_keys.append(x)
        else:
            other_keys.append(x)

    print("total common keys", len(common_keys))
    print("total other keys", len(other_keys))

    # create a new state dict
    new_state_dict = {}
    for key in common_keys:
        new_key_name = key[len(prefix) + 1 :]
        new_state_dict[new_key_name] = source_model["state_dict"][key]

    # assert that keys in pretrained_swint and new_state_dict are the same
    for key in new_state_dict.keys():
        assert key in pretrained_swint["state_dict"].keys(), "key not found in pretrained swint"

    print("total keys in new state dict", len(new_state_dict.keys()))

    # save the new state dict
    pretrained_swint["state_dict"] = new_state_dict
    torch.save(pretrained_swint, target_save_path)


if __name__ == "__main__":
    args = get_args()
    convert_to_swint_pth(
        pretrained_swint_path=args.m,
        source_model_path=args.s,
        target_save_path=args.t,
    )
