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
    test_data,
    test_label,
    in_channels,
    num_classes,
    num_gpus,
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

    model_def = build_torch_model_def(
        in_channels,
        num_classes,
        lr,
        enable_metrics=eval_at_end,
        optimizer_name=optimizer_name,
        momentum=momentum,
        device=device,
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

    history = fl_model.fit(
        train_data,
        train_label,
        epochs=rounds * local_epochs,
        batch_size=batch_size,
        aggregate_freq=aggregate_freq,
    )
    print("Training history keys:", list(history.keys()) if isinstance(history, dict) else type(history))

    if eval_at_end:
        eval_res = fl_model.evaluate(
            test_data,
            test_label,
            batch_size=batch_size,
        )
        global_metrics, local_metrics = eval_res
        global_out = {m.name: _metric_value(m) for m in global_metrics}
        print("\nFinal evaluation on test set:")
        print(global_out)
        if isinstance(local_metrics, dict):
            local_out = {
                party: {m.name: _metric_value(m) for m in metrics}
                for party, metrics in local_metrics.items()
            }
            print("Per-client evaluation on test set:")
            print(local_out)
            acc_pairs = {}
            total_weighted = 0.0
            total_count = 0
            count_map = _compute_partition_counts(test_data, device_list)
            count_by_party = {
                getattr(device, "party", str(device)): count
                for device, count in count_map.items()
            }
            for party, metrics in local_metrics.items():
                party_name = party if isinstance(party, str) else getattr(party, "party", str(party))
                acc_metric = None
                for metric in metrics:
                    if "accuracy" in metric.name.lower():
                        acc_metric = metric
                        break
                if acc_metric is None:
                    continue
                acc_value, _ = _metric_value_and_count(acc_metric)
                count = int(count_by_party.get(party_name, 0))
                acc_pairs[str(party_name)] = (acc_value, count)
                total_weighted += acc_value * count
                total_count += count
            if acc_pairs:
                summary = {"acc_global_weighted": total_weighted / max(total_count, 1)}
                for party, (acc_value, acc_count) in acc_pairs.items():
                    summary[f"acc_{party}"] = acc_value
                    summary[f"n_{party}"] = acc_count
                print("Weighted accuracy summary:")
                print(summary)
