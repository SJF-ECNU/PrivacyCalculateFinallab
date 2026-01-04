import argparse
import copy
import random

import numpy as np
import secretflow as sf
import torch

from fl.config import SUPPORTED_MODELS, load_config
from fl.data import SUPPORTED_DATASETS, build_fed_dataset
from fl.flmodel_runner import run_fedavg_or_fedprox
from fl.personalized import run_fedbn, run_fedper


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--models",
        default="fedavg,fedbn,fedper",
        help="Comma-separated list, e.g. fedavg,fedbn,fedper",
    )
    parser.add_argument("--dataset", choices=sorted(SUPPORTED_DATASETS))
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--local-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def _apply_overrides(cfg, args):
    if args.dataset:
        cfg["data"]["dataset"] = args.dataset
        if "name" in cfg["data"]:
            cfg["data"]["name"] = args.dataset
    if args.rounds is not None:
        cfg["train"]["rounds"] = args.rounds
    if args.local_epochs is not None:
        cfg["train"]["local_epochs"] = args.local_epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.seed is not None:
        cfg["data"]["seed"] = args.seed
    if args.device is not None:
        cfg["train"]["device"] = args.device
    cfg["train"]["optimizer"] = args.optimizer
    cfg["train"]["momentum"] = args.momentum


def _set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _run_one(cfg, model_name):
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")
    cfg = copy.deepcopy(cfg)
    cfg["model"]["name"] = model_name

    seed = cfg["data"]["seed"]
    _set_seeds(seed)

    sf.shutdown()
    runtime_cfg = cfg["runtime"]
    num_gpus = runtime_cfg.get("num_gpus", 0)
    sf.init(runtime_cfg["parties"], address=runtime_cfg["address"], num_gpus=num_gpus)
    gpu_per_party = runtime_cfg.get(
        "gpu_per_party", num_gpus / max(len(runtime_cfg["clients"]), 1) if num_gpus else 0
    )

    device_map = {name: sf.PYU(name) for name in runtime_cfg["parties"]}
    device_list = [device_map[name] for name in runtime_cfg["clients"]]
    server = device_map[runtime_cfg["server"]]

    (train_data, train_label), (test_data, test_label), in_channels, num_classes = (
        build_fed_dataset(cfg, device_map)
    )

    print(f"\n=== Running {model_name} ===")
    if model_name in {"fedavg", "fedprox"}:
        run_fedavg_or_fedprox(
            cfg,
            device_list,
            server,
            train_data,
            train_label,
            test_data,
            test_label,
            in_channels,
            num_classes,
            gpu_per_party,
        )
    elif model_name == "fedbn":
        run_fedbn(
            cfg,
            device_list,
            train_data,
            train_label,
            test_data,
            test_label,
            in_channels,
            num_classes,
        )
    elif model_name == "fedper":
        run_fedper(
            cfg,
            device_list,
            train_data,
            train_label,
            test_data,
            test_label,
            in_channels,
            num_classes,
        )

    sf.shutdown()


def main():
    args = _parse_args()
    cfg = load_config(args.config)
    _apply_overrides(cfg, args)

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    for model_name in model_names:
        _run_one(cfg, model_name)


if __name__ == "__main__":
    main()
