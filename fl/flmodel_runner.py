import numpy as np
import secretflow as sf
from secretflow.ml.nn import FLModel
from secretflow.security.aggregation import SecureAggregator

from .models import build_torch_model_def
from .utils import get_partition


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _local_accuracy(preds, labels):
    preds = _to_numpy(preds)
    labels = _to_numpy(labels)
    if preds.ndim > 1:
        preds = np.argmax(preds, axis=-1)
    if labels.ndim > 1:
        labels = np.argmax(labels, axis=-1)
    correct = int(np.sum(preds == labels))
    total = int(labels.shape[0])
    return {"correct": correct, "total": total}


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
    aggregate_freq = train_cfg.get("aggregate_freq", local_epochs)
    eval_at_end = train_cfg.get("eval_at_end", True)

    model_def = build_torch_model_def(in_channels, num_classes, lr)
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
        pred_data = fl_model.predict(test_data, batch_size=batch_size)
        eval_stats = []
        for device in device_list:
            pred_part = get_partition(pred_data, device)
            label_part = get_partition(test_label, device)
            stats_obj = device(_local_accuracy)(pred_part, label_part)
            eval_stats.append(sf.reveal(stats_obj))
        total_correct = sum(s["correct"] for s in eval_stats)
        total = sum(s["total"] for s in eval_stats)
        accuracy = total_correct / max(total, 1)
        print("\nFinal evaluation on test set:")
        print({"accuracy": accuracy})
