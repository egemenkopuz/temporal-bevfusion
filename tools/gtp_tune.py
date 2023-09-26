import argparse
import json
import os

import joblib
import numpy as np
import optuna
from mmcv import Config
from torchpack.utils.config import configs

from mmdet3d.utils import recursive_eval

CLASSES = (
    "CAR",
    "TRAILER",
    "TRUCK",
    "VAN",
    "PEDESTRIAN",
    "BUS",
    "MOTORCYCLE",
    "BICYCLE",
    "EMERGENCY_VEHICLE",
)

AP_DISTS = [
    "ap_dist_0.5",
    "ap_dist_1.0",
    "ap_dist_2.0",
    "ap_dist_4.0",
]

# CLS_SEARCH_SPACE = {
#     "CAR": [8, 15],
#     "TRAILER": [0, 2],
#     "TRUCK": [0, 4],
#     "VAN": [0, 5],
#     "PEDESTRIAN": [0, 8],
#     "BUS": [0, 2],
#     "MOTORCYCLE": [0, 4],
#     "BICYCLE": [0, 4],
#     "EMERGENCY_VEHICLE": [0, 2],
# }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="config file")
    parser.add_argument("--run-dir", required=True, type=str, help="run directory")
    parser.add_argument("--n-epochs", required=True, type=int, help="max epochs")
    parser.add_argument("--n-gpus", required=True, type=int, help="max gpus")
    parser.add_argument("--n-trials", required=True, type=int, help="max trials")
    parser.add_argument("--CAR", nargs="+", type=int, required=True)
    parser.add_argument("--TRAILER", nargs="+", type=int, required=True)
    parser.add_argument("--TRUCK", nargs="+", type=int, required=True)
    parser.add_argument("--VAN", nargs="+", type=int, required=True)
    parser.add_argument("--PEDESTRIAN", nargs="+", type=int)
    parser.add_argument("--BUS", nargs="+", type=int, required=True)
    parser.add_argument("--MOTORCYCLE", nargs="+", type=int, required=True)
    parser.add_argument("--BICYCLE", nargs="+", type=int, required=True)
    parser.add_argument("--EMERGENCY_VEHICLE", nargs="+", type=int, required=True)
    parser.add_argument("--verbose", action="store_true")
    args, opts = parser.parse_known_args()
    return args, opts


def tune(args, opts) -> None:
    search_space = {}
    for cls in CLASSES:
        assert cls in args.__dict__ and len(args.__dict__[cls]) == 2
        search_space[cls] = args.__dict__[cls]
    assert args.n_gpus > 0
    assert args.n_epochs > 0
    assert args.n_trials > 0
    assert os.path.exists(args.config)

    os.makedirs(args.run_dir, exist_ok=True)
    run_id = os.path.basename(args.run_dir)

    study = optuna.create_study(
        study_name=run_id,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
    )

    study.optimize(
        lambda trial: objective(
            trial,
            args.config,
            args.run_dir,
            args.n_epochs,
            args.n_gpus,
            CLASSES,
            AP_DISTS,
            search_space,
            args.verbose,
        ),
        n_trials=args.n_trials,
    )

    best_trial = study.best_trial

    for key, value in best_trial.params.items():
        print(f"{key}: {value}")

    joblib.dump(study, os.path.join(args.run_dir, "study.pkl"))

    df = study.trials_dataframe()
    save_results(df, study, args.run_dir)


def read_map(checkpoint_folder_path: str, classes: list, ap_dists: list) -> float:
    try:
        json_file = [f for f in os.listdir(checkpoint_folder_path) if f.endswith(".json")][0]
        json_file = os.path.join(checkpoint_folder_path, json_file)

        data = []
        with open(json_file, "r") as f:
            for line in f:
                data.append(json.loads(line))
        data = data[-1]

        class_ap = {x: 0 for x in classes}
        for cls in classes:
            all_ap_dists = []
            for ap_dist in ap_dists:
                all_ap_dists.append(data[f"object/{cls}_{ap_dist}"])
            class_ap[cls] = np.mean(all_ap_dists)

        map = data["object/map"]
        epoch = data["epoch"]
    except Exception as e:
        print(e)
        map = 0
        class_ap = {x: 0 for x in classes}
        epoch = 0
    return map, class_ap, epoch


def find_gtp_in_pipeline(pipeline: list) -> int:
    for i, p in enumerate(pipeline):
        if p["type"] == "ObjectPaste":
            return i
    return -1


def objective(
    trial,
    source_config_path: str,
    tune_target_folder_path: str,
    max_epochs: int,
    n_gpus: int,
    target_classes: list,
    ap_dists: list,
    cls_search_space: dict,
    verbose: bool = False,
) -> float:
    params = {c: trial.suggest_int(c, v[0], v[1]) for c, v in cls_search_space.items()}
    print(f"Trial {trial.number} - Params: {params}")

    run_dir = os.path.join(tune_target_folder_path, f"trial_{trial.number}")

    configs.load(source_config_path, recursive=True)

    cfg = Config(recursive_eval(configs), filename=source_config_path)

    gtp_idx = find_gtp_in_pipeline(cfg.data.train.dataset.pipeline)
    cfg.data.train.dataset.pipeline[gtp_idx].db_sampler.sample_groups = params

    tmp_config_path = os.path.join(tune_target_folder_path, "tmp_config.yaml")
    if os.path.exists(tmp_config_path):
        os.remove(tmp_config_path)

    cfg.run_dir = run_dir
    cfg.checkpoint_config.max_keep_ckpts = 0
    cfg.runner.max_epochs = max_epochs
    cfg.optimizer.lr = 2.0e-4
    cfg.dump(tmp_config_path)

    command = f"torchpack dist-run -np {n_gpus} python tools/train.py {tmp_config_path} --run-dir {run_dir}"
    if not verbose:
        command += " > /dev/null 2>&1"
    os.system(command)

    map, _, epoch = read_map(run_dir, target_classes, ap_dists)

    trial.report(map, epoch)

    return map


def save_results(df, study, save_path: str):
    df.to_csv(os.path.join(save_path, "results.csv"))
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(os.path.join(save_path, "param_importances.png"))
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(os.path.join(save_path, "optimization_history.png"))
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_image(os.path.join(save_path, "parallel_coordinate.png"))


if __name__ == "__main__":
    args, opts = parse_args()
    tune(args, opts)