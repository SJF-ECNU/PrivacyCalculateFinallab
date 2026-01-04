from secretflow.ml.nn import FLModel
from secretflow.security.aggregation import SecureAggregator

from .models import build_torch_model_def


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
        validation_data=(test_data, test_label),
        epochs=rounds * local_epochs,
        batch_size=batch_size,
        aggregate_freq=aggregate_freq,
    )
    print("Training history keys:", list(history.keys()) if isinstance(history, dict) else type(history))

    eval_res = fl_model.evaluate(
        test_data,
        test_label,
        batch_size=batch_size,
    )
    print("\nFinal evaluation on test set:")
    print(eval_res)
