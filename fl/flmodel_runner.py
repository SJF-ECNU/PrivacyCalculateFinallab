import torch

from secretflow.ml.nn import FLModel
from secretflow.security.aggregation import SecureAggregator

from .models import build_torch_model_def


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
