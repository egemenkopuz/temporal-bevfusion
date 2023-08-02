import os
from argparse import ArgumentParser, Namespace

import cv2


def get_args() -> Namespace:
    """
    Parse given arguments for create_video function.

    Returns:
        Namespace: parsed arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-s", "--source_folder_dir", type=str, required=True)
    parser.add_argument("-t", "--target_path", type=str, required=True)

    return parser.parse_args()


def create_video(source_folder_dir: str, target_path: str) -> None:
    images = sorted(
        [
            img
            for img in os.listdir(source_folder_dir)
            if img.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    if len(images) == 0:
        return

    os.makedirs(os.path.dirname(target_path), exist_ok=True, mode=0o777)

    frame = cv2.imread(os.path.join(source_folder_dir, images[0]))
    height, width, _ = frame.shape

    video_name = os.path.join(target_path)
    video = cv2.VideoWriter(video_name, 0x7634706D, 10, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(source_folder_dir, image)))

    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    args = get_args()
    create_video(args.source_folder_dir, args.target_path)
