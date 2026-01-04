import argparse
import json
from pathlib import Path

from .data import SUPPORTED_DATASETS


SUPPORTED_MODELS = {"fedavg", "fedprox", "feddyn"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--model", choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--dataset", choices=sorted(SUPPORTED_DATASETS))
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--local-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--aggregate-freq", type=int)
    parser.add_argument("--mu", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def load_config(path):
    return json.loads(Path(path).read_text())


def apply_overrides(cfg, args):
    if args.model:
        cfg["model"]["name"] = args.model
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
    if args.aggregate_freq is not None:
        cfg["train"]["aggregate_freq"] = args.aggregate_freq
    if args.mu is not None:
        cfg["fedprox"]["mu"] = args.mu
    if args.alpha is not None:
        cfg["feddyn"]["alpha"] = args.alpha
    if args.seed is not None:
        cfg["data"]["seed"] = args.seed
