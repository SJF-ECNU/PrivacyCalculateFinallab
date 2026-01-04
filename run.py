import random

import numpy as np
import secretflow as sf

import torch
from fl.config import SUPPORTED_MODELS, apply_overrides, load_config, parse_args
from fl.data import SUPPORTED_DATASETS, build_fed_dataset
from fl.flmodel_runner import run_fedavg_or_fedprox
from fl.personalized import run_fedbn, run_fedper


def main():
    args = parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, args)

    model_name = cfg["model"]["name"]
    data_name = cfg["data"].get("dataset") or cfg["data"].get("name")
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")
    if not data_name:
        raise ValueError("Missing data.dataset in config.")
    dataset_cfg = cfg["data"].get("datasets", {})
    if dataset_cfg and data_name not in dataset_cfg:
        raise ValueError(f"Dataset {data_name} not found in config.")
    if data_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {data_name}")

    seed = cfg["data"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"The version of SecretFlow: {sf.__version__}")
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

    (train_data, train_label), (test_data, test_label), in_channels, num_classes = build_fed_dataset(
        cfg, device_map
    )

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
    else:
        raise ValueError(f"Unsupported model: {model_name}")


if __name__ == "__main__":
    main()
