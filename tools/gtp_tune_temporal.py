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

CLASS_TRANS_TYPES = {
    "CAR": "uniform",
    "TRAILER": "uniform",
    "TRUCK": "uniform",
    "VAN": "uniform",
    "PEDESTRIAN": "uniform",
    "BUS": "uniform",
    "MOTORCYCLE": "uniform",
    "BICYCLE": "uniform",
    "EMERGENCY_VEHICLE": "uniform",
}

CLASS_ROT_TYPES = {
    "CAR": "normal",
    "TRAILER": "normal",
    "TRUCK": "normal",
    "VAN": "normal",
    "PEDESTRIAN": "normal",
    "BUS": "normal",
    "MOTORCYCLE": "normal",
    "BICYCLE": "normal",
    "EMERGENCY_VEHICLE": "normal",
}


AP_DISTS = [
    "ap_dist_0.5",
    "ap_dist_1.0",
    "ap_dist_2.0",
    "ap_dist_4.0",
]


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
    parser.add_argument("--enqueue", nargs="+", type=int, required=False)
    parser.add_argument("--timeout", type=int, required=False, help="timeout in hours")
    parser.add_argument("--verbose", action="store_true")
    args, opts = parser.parse_known_args()
    return args, opts


def tune(args, opts) -> None:
    search_space = {}
    for cls in CLASSES:
        assert (
            cls in args.__dict__ and len(args.__dict__[cls]) == 4
        )  # TRANS_MIN, TRANS_MAX, ROT_MIN, ROT_MAX
        search_space[cls] = args.__dict__[cls]

    assert args.n_gpus > 0
    assert args.n_epochs > 0
    assert args.n_trials > 0
    assert os.path.exists(args.config)

    os.makedirs(args.run_dir, exist_ok=True)
    run_id = os.path.basename(args.run_dir)

    db_path = "sqlite:///" + args.run_dir + "/tune.db"
    study = optuna.create_study(
        study_name=run_id,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        storage=db_path,
        load_if_exists=True,
    )

    if args.timeout is not None:
        print(f"Timeout: {args.timeout}h")

    if args.enqueue is not None:
        enqueue_dict = {x: args.enqueue[i] for i, x in enumerate(CLASSES)}
        print(f"Enqueue: {enqueue_dict}")
        study.enqueue_trial(enqueue_dict)

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
            args.timeout,
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
        last_val = None
        for x in data[::-1]:
            if x["mode"] == "val":
                last_val = x
                break

        if last_val is None:
            raise Exception("No val mode found in json file")

        class_ap = {x: 0 for x in classes}
        for cls in classes:
            all_ap_dists = []
            for ap_dist in ap_dists:
                all_ap_dists.append(last_val[f"object/{cls}_{ap_dist}"])
            class_ap[cls] = np.mean(all_ap_dists)

        map = last_val["object/map"]
        epoch = last_val["epoch"]
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
    timeout: int = None,
    verbose: bool = False,
) -> float:
    trans_params = {
        cls: trial.suggest_float(f"{cls}_trans_lim", v[0], v[1])
        for cls, v in cls_search_space.items()
    }
    rot_params = {
        cls: trial.suggest_float(f"{cls}_rot_lim", v[2], v[3])
        for cls, v in cls_search_space.items()
    }
    print(f"Trial {trial.number} - Translation Params: {trans_params}")
    print(f"Trial {trial.number} - Rotation Params: {rot_params}")

    run_dir = os.path.join(tune_target_folder_path, f"trial_{trial.number}")
    configs.load(source_config_path, recursive=True)

    cfg = Config(recursive_eval(configs), filename=source_config_path)

    gtp_idx = find_gtp_in_pipeline(cfg.data.train.dataset.pipeline)
    cfg.data.train.dataset.pipeline[gtp_idx].db_sampler.cls_trans_lim = {
        cls: [CLASS_TRANS_TYPES[cls.split("_")[0]], v[0], v[1]] for cls, v in trans_params.items()
    }
    cfg.data.train.dataset.pipeline[gtp_idx].db_sampler.cls_rot_lim = {
        cls: [CLASS_ROT_TYPES[cls.split("_")[0]], v[0], v[1]] for cls, v in rot_params.items()
    }

    tmp_config_path = os.path.join(tune_target_folder_path, "tmp_config.yaml")
    if os.path.exists(tmp_config_path):
        os.remove(tmp_config_path)

    cfg.run_dir = run_dir
    cfg.checkpoint_config.max_keep_ckpts = 1
    cfg.runner.max_epochs = max_epochs
    cfg.deterministic = True
    cfg.dump(tmp_config_path)

    if timeout is not None:
        command = f"timeout {timeout}h "
    else:
        command = ""

    command += f"torchpack dist-run -np {n_gpus} python tools/train.py {tmp_config_path} --run-dir {run_dir}"
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
