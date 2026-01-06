import torch
import secretflow as sf

from secretflow.ml.nn import FLModel
from secretflow.security.aggregation import SecureAggregator

from .models import build_torch_model_def
from .utils import get_partition


def _metric_value(metric):
    total = metric.total
    count = metric.count
    if hasattr(total, "item"):
        total = total.item()
    if hasattr(count, "item"):
        count = count.item()
    count = int(count) if count is not None else 0
    if count <= 0:
        return 0.0
    return float(total) / count


def _metric_value_and_count(metric):
    total = metric.total
    count = metric.count
    if hasattr(total, "item"):
        total = total.item()
    if hasattr(count, "item"):
        count = count.item()
    count = int(count) if count is not None else 0
    if count <= 0:
        return 0.0, 0
    return float(total) / count, count


def _partition_size(x):
    if hasattr(x, "shape"):
        return int(x.shape[0])
    return len(x)


def _compute_partition_counts(fed_data, device_list):
    counts = {}
    for device in device_list:
        data_obj = get_partition(fed_data, device)
        count_obj = device(_partition_size)(data_obj)
        counts[device] = int(sf.reveal(count_obj))
    return counts


def run_fedavg_or_fedprox(
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
    num_gpus,
    logger=None,
):
    train_cfg = cfg["train"]
    rounds = train_cfg["rounds"]
    local_epochs = train_cfg["local_epochs"]
    batch_size = train_cfg["batch_size"]
    lr = train_cfg["lr"]
    optimizer_name = train_cfg.get("optimizer", "adam")
    momentum = train_cfg.get("momentum", 0.0)
    device = train_cfg.get("device", "cpu")
    aggregate_freq = train_cfg.get("aggregate_freq", local_epochs)
    eval_at_end = train_cfg.get("eval_at_end", True)
    eval_interval = max(int(train_cfg.get("eval_interval", 1)), 1)
    wandb_cfg = cfg.get("wandb", {})
    log_per_client = bool(wandb_cfg.get("log_per_client", False))
    log_train_per_client = bool(
        wandb_cfg.get("log_train_per_client", log_per_client)
    )
    early_cfg = cfg.get("early_stop", {})
    early_enabled = bool(early_cfg.get("enable", False)) and val_data is not None
    early_metric = early_cfg.get("metric", "val_acc")
    early_mode = early_cfg.get("mode", "max")
    early_patience = int(early_cfg.get("patience", 5))
    early_min_delta = float(early_cfg.get("min_delta", 0.0))

    model_arch = cfg.get("model", {}).get("arch", "convnet")
    model_def = build_torch_model_def(
        in_channels,
        num_classes,
        lr,
        enable_metrics=eval_at_end,
        optimizer_name=optimizer_name,
        momentum=momentum,
        device=device,
        arch=model_arch,
    )
    use_cuda = device == "cuda" and torch.cuda.is_available()
    print(
        f"[fedavg_init] device={device}, cuda_available={torch.cuda.is_available()}, "
        f"use_cuda={use_cuda}"
    )
    aggregator = SecureAggregator(server, device_list)

    strategy_name = cfg["model"]["name"]
    if strategy_name == "fedavg":
        strategy = "fed_avg_w"
        strategy_params = None
    else:
        strategy = "fed_prox"
        strategy_params = {"mu": cfg["fedprox"]["mu"]}
        if device == "cuda":
            try:
                from secretflow.ml.nn.fl.backend.torch.strategy import fed_prox

                def _w_norm(self, w1, w2):
                    l1 = len(w1)
                    assert l1 == len(w2), "weights should be same in the shape"
                    proximal_term = 0
                    for i in range(l1):
                        w1_t = torch.as_tensor(
                            w1[i], device=w2[i].device, dtype=w2[i].dtype
                        )
                        proximal_term += (w1_t - w2[i]).norm(2) ** 2
                    return proximal_term

                fed_prox.PYUFedProx.w_norm = _w_norm
            except Exception as exc:
                print(f"[fedprox_init] Failed to patch w_norm for CUDA: {exc}")

    fl_model = FLModel(
        server=server,
        device_list=device_list,
        model=model_def,
        aggregator=aggregator,
        strategy=strategy,
        backend="torch",
        strategy_params=strategy_params,
        num_gpus=num_gpus,
    )

    def _value_from_dict(metric_dict, keys):
        keys = [k.lower() for k in keys]
        for name, value in metric_dict.items():
            if any(k in name.lower() for k in keys):
                return value
        return None

    def _evaluate_split(split_data, split_label):
        if split_data is None or split_label is None:
            return None, None
        global_metrics, local_metrics = fl_model.evaluate(
            split_data,
            split_label,
            batch_size=batch_size,
        )
        global_out = {m.name: _metric_value(m) for m in global_metrics}
        per_client = {}
        if isinstance(local_metrics, dict):
            count_map = _compute_partition_counts(split_data, device_list)
            count_by_party = {
                getattr(device, "party", str(device)): count
                for device, count in count_map.items()
            }
            for party, metrics in local_metrics.items():
                party_name = party if isinstance(party, str) else getattr(party, "party", str(party))
                acc_metric = None
                loss_metric = None
                for metric in metrics:
                    metric_name = metric.name.lower()
                    if acc_metric is None and "accuracy" in metric_name:
                        acc_metric = metric
                    if loss_metric is None and "loss" in metric_name:
                        loss_metric = metric
                acc_value, _ = _metric_value_and_count(acc_metric) if acc_metric else (None, 0)
                loss_value = _metric_value(loss_metric) if loss_metric else None
                count = int(count_by_party.get(party_name, 0))
                per_client[str(party_name)] = {
                    "accuracy": acc_value,
                    "loss": loss_value,
                    "n": count,
                }
        return global_out, per_client

    def _history_last_metrics(history_obj):
        if not isinstance(history_obj, dict):
            return {}
        last_metrics = {}
        for key, values in history_obj.items():
            if isinstance(values, list) and values:
                value = values[-1]
            elif isinstance(values, (int, float)):
                value = values
            else:
                continue
            if hasattr(value, "item"):
                value = value.item()
            last_metrics[key] = value
        return last_metrics

    best_metric = None
    bad_rounds = 0
    stopped_early = False

    for r in range(rounds):
        history = fl_model.fit(
            train_data,
            train_label,
            epochs=local_epochs,
            batch_size=batch_size,
            aggregate_freq=aggregate_freq,
        )
        train_metrics = _history_last_metrics(history)
        train_loss = _value_from_dict(train_metrics, ["loss"])

        should_eval = (r + 1) % eval_interval == 0 or r == rounds - 1
        if should_eval and val_data is not None and val_label is not None:
            global_out, per_client = _evaluate_split(val_data, val_label)
            if global_out is None:
                global_out = {}
            val_acc = _value_from_dict(global_out, ["accuracy", "acc"])
            val_loss = _value_from_dict(global_out, ["loss"])
            metrics = {"round": r + 1}
            if val_acc is not None:
                metrics["val_acc"] = val_acc
            if val_loss is not None:
                metrics["val_loss"] = val_loss
            if train_loss is not None:
                metrics["train_loss"] = train_loss
            metrics.update(train_metrics)

            if logger:
                if log_per_client and per_client:
                    for name, stats in per_client.items():
                        if stats.get("accuracy") is not None:
                            metrics[f"val_acc_{name}"] = stats["accuracy"]
                        if stats.get("loss") is not None:
                            metrics[f"val_loss_{name}"] = stats["loss"]
                if log_train_per_client:
                    _, train_per_client = _evaluate_split(train_data, train_label)
                    if train_per_client:
                        for name, stats in train_per_client.items():
                            if stats.get("loss") is not None:
                                metrics[f"train_loss_{name}"] = stats["loss"]
                logger.log(metrics, step=r + 1)

            if early_enabled:
                current_metric = None
                if early_metric == "val_loss" and val_loss is not None:
                    current_metric = val_loss
                elif val_acc is not None:
                    current_metric = val_acc
                if current_metric is not None:
                    if best_metric is None:
                        improved = True
                    elif early_mode == "min":
                        improved = current_metric < (best_metric - early_min_delta)
                    else:
                        improved = current_metric > (best_metric + early_min_delta)

                    if improved:
                        best_metric = current_metric
                        bad_rounds = 0
                    else:
                        bad_rounds += 1
                    if logger:
                        logger.log(
                            {"best_metric": best_metric, "bad_rounds": bad_rounds},
                            step=r + 1,
                        )
                    if bad_rounds >= early_patience:
                        print(
                            "[early_stop] "
                            f"stop at round={r + 1} best={best_metric:.6f}"
                        )
                        stopped_early = True
                        break
        elif logger and train_metrics:
            train_metrics = dict(train_metrics)
            train_metrics["round"] = r + 1
            if train_loss is not None:
                train_metrics["train_loss"] = train_loss
            logger.log(train_metrics, step=r + 1)

    if stopped_early:
        print("[early_stop] training stopped early.")

    if eval_at_end:
        global_out, per_client = _evaluate_split(test_data, test_label)
        print("\nFinal evaluation on test set:")
        print(global_out)
        if per_client:
            print("Per-client evaluation on test set:")
            print(per_client)
            acc_pairs = {}
            total_weighted = 0.0
            total_count = 0
            for party, stats in per_client.items():
                acc_value = stats.get("accuracy")
                count = int(stats.get("n", 0))
                if acc_value is None:
                    continue
                acc_pairs[str(party)] = (acc_value, count)
                total_weighted += acc_value * count
                total_count += count
            if acc_pairs:
                summary = {"acc_global_weighted": total_weighted / max(total_count, 1)}
                for party, (acc_value, acc_count) in acc_pairs.items():
                    summary[f"acc_{party}"] = acc_value
                    summary[f"n_{party}"] = acc_count
                print("Weighted accuracy summary:")
                print(summary)
        if logger and global_out:
            metrics = {
                "test_acc": _value_from_dict(global_out, ["accuracy", "acc"]),
                "test_loss": _value_from_dict(global_out, ["loss"]),
            }
            if log_per_client and per_client:
                for name, stats in per_client.items():
                    if stats.get("accuracy") is not None:
                        metrics[f"test_acc_{name}"] = stats["accuracy"]
                    if stats.get("loss") is not None:
                        metrics[f"test_loss_{name}"] = stats["loss"]
            logger.log(metrics)
