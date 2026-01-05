import inspect
import os
import random

import numpy as np
import secretflow as sf

import torch
from fl.config import SUPPORTED_MODELS, apply_overrides, load_config, parse_args
from fl.data import SUPPORTED_DATASETS, build_fed_dataset
from fl.flmodel_runner import run_fedavg_or_fedprox
from fl.personalized import run_fedbn, run_fedper
from fl.wandb_logger import WandbLogger


def _build_run_name(cfg):
    model = cfg["model"]["name"]
    arch = cfg["model"].get("arch", "convnet")
    dataset = cfg["data"].get("dataset") or cfg["data"].get("name")
    train_cfg = cfg["train"]
    lr = train_cfg.get("lr")
    bs = train_cfg.get("batch_size")
    local_epochs = train_cfg.get("local_epochs")
    agg = train_cfg.get("aggregate_freq", local_epochs)
    opt = train_cfg.get("optimizer", "sgd")
    seed = cfg["data"].get("seed")
    parts = [
        model,
        arch,
        dataset,
        f"lr{lr:.2e}",
        f"bs{bs}",
        f"e{local_epochs}",
        f"ag{agg}",
        opt,
        f"s{seed}",
    ]
    if model == "fedprox":
        mu = cfg.get("fedprox", {}).get("mu")
        if mu is not None:
            parts.append(f"mu{mu:.2e}")
    return "-".join(str(p) for p in parts if p is not None)


def run_experiment(cfg, run_name=None):
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

    if run_name is None:
        run_name = _build_run_name(cfg)
    logger = WandbLogger(
        cfg, run_name=run_name, extra_config={"model": model_name, "dataset": data_name}
    )

    print(f"The version of SecretFlow: {sf.__version__}")
    def make_pyu(name, gpus):
        try:
            sig = inspect.signature(sf.PYU)
        except (TypeError, ValueError):
            return sf.PYU(name)
        if gpus and "num_gpus" in sig.parameters:
            return sf.PYU(name, num_gpus=gpus)
        return sf.PYU(name)

    try:
        sf.shutdown()
        runtime_cfg = cfg["runtime"]
        train_device = cfg["train"].get("device")
        if train_device == "cuda":
            cuda_visible = runtime_cfg.get("cuda_visible_devices")
            if cuda_visible is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible)
            else:
                os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        num_gpus = runtime_cfg.get("num_gpus", 0)
        sf.init(runtime_cfg["parties"], address=runtime_cfg["address"], num_gpus=num_gpus)
        gpu_per_party = runtime_cfg.get(
            "gpu_per_party", num_gpus / max(len(runtime_cfg["clients"]), 1) if num_gpus else 0
        )

        device_map = {
            name: make_pyu(name, gpu_per_party if name in runtime_cfg["clients"] else 0)
            for name in runtime_cfg["parties"]
        }
        device_list = [device_map[name] for name in runtime_cfg["clients"]]
        server = device_map[runtime_cfg["server"]]

        (
            (train_data, train_label),
            (val_data, val_label),
            (test_data, test_label),
            in_channels,
            num_classes,
        ) = build_fed_dataset(cfg, device_map)

        if model_name in {"fedavg", "fedprox"}:
            run_fedavg_or_fedprox(
                cfg,
                device_list,
                server,
                train_data,
                train_label,
                val_data,
                val_label,
                test_data,
                test_label,
                in_channels,
                num_classes,
                gpu_per_party,
                logger=logger,
            )
        elif model_name == "fedbn":
            run_fedbn(
                cfg,
                device_list,
                train_data,
                train_label,
                val_data,
                val_label,
                test_data,
                test_label,
                in_channels,
                num_classes,
                logger=logger,
            )
        elif model_name == "fedper":
            run_fedper(
                cfg,
                device_list,
                train_data,
                train_label,
                val_data,
                val_label,
                test_data,
                test_label,
                in_channels,
                num_classes,
                logger=logger,
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    finally:
        if logger:
            logger.finish()
        sf.shutdown()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, args)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
