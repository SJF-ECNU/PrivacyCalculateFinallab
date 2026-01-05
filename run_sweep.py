import argparse
import copy
import json
import math
import random
from pathlib import Path

from run import run_experiment
from fl.config import load_config


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--sweep", default="sweeps/random.json")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _sample_param(spec, rng):
    if "values" in spec:
        return rng.choice(spec["values"])
    if "value" in spec:
        return spec["value"]
    if "min" in spec and "max" in spec:
        min_val = spec["min"]
        max_val = spec["max"]
        dist = spec.get("distribution", "uniform")
        if dist == "log_uniform":
            log_min = math.log10(min_val)
            log_max = math.log10(max_val)
            return 10 ** rng.uniform(log_min, log_max)
        if dist == "int_uniform":
            return rng.randint(int(min_val), int(max_val))
        return rng.uniform(min_val, max_val)
    raise ValueError(f"Unsupported sweep spec: {spec}")


def _apply_params(cfg, params):
    if "lr" in params:
        cfg["train"]["lr"] = params["lr"]
    if "batch_size" in params:
        cfg["train"]["batch_size"] = int(params["batch_size"])
    if "local_epochs" in params:
        cfg["train"]["local_epochs"] = int(params["local_epochs"])
    if "optimizer" in params:
        cfg["train"]["optimizer"] = params["optimizer"]
    if "momentum" in params:
        cfg["train"]["momentum"] = float(params["momentum"])
    if "aggregate_freq" in params:
        cfg["train"]["aggregate_freq"] = int(params["aggregate_freq"])
    if "mu" in params:
        cfg.setdefault("fedprox", {})["mu"] = float(params["mu"])
    return cfg


def _build_run_name(cfg, params, index):
    model = cfg["model"]["name"]
    arch = cfg["model"].get("arch", "convnet")
    dataset = cfg["data"].get("dataset") or cfg["data"].get("name")
    lr = params.get("lr", cfg["train"]["lr"])
    bs = params.get("batch_size", cfg["train"]["batch_size"])
    le = params.get("local_epochs", cfg["train"]["local_epochs"])
    opt = params.get("optimizer", cfg["train"].get("optimizer", "sgd"))
    return f"{model}-{arch}-{dataset}-lr{lr:.2e}-bs{bs}-e{le}-{opt}-run{index}"


def main():
    args = _parse_args()
    base_cfg = load_config(args.config)
    sweep_cfg = json.loads(Path(args.sweep).read_text())
    method = sweep_cfg.get("method", "random")
    if method != "random":
        raise ValueError(f"Only random sweep is supported, got: {method}")

    count = args.count if args.count is not None else int(sweep_cfg.get("count", 1))
    params_spec = sweep_cfg.get("parameters", {})
    rng = random.Random(args.seed)

    for i in range(count):
        params = {k: _sample_param(v, rng) for k, v in params_spec.items()}
        cfg = copy.deepcopy(base_cfg)
        cfg = _apply_params(cfg, params)
        cfg["data"]["seed"] = int(cfg["data"].get("seed", 0)) + i + 1

        wandb_cfg = cfg.setdefault("wandb", {})
        group = wandb_cfg.get("group") or "sweep"
        wandb_cfg["group"] = group
        tags = list(wandb_cfg.get("tags") or [])
        if "sweep" not in tags:
            tags.append("sweep")
        wandb_cfg["tags"] = tags

        run_name = _build_run_name(cfg, params, i + 1)
        print(f"\n=== Sweep run {i + 1}/{count}: {run_name} ===")
        run_experiment(cfg, run_name=run_name)


if __name__ == "__main__":
    main()
