import hashlib
import os
import random

import secretflow as sf

import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader, TensorDataset

from .models import build_model
from .utils import get_partition, prepare_tensor


def _get_named_params(model, names):
    param_map = dict(model.named_parameters())
    return [param_map[name].detach().cpu() for name in names]


def _set_named_params(model, names, params):
    param_map = dict(model.named_parameters())
    with torch.no_grad():
        for name, src in zip(names, params):
            param = param_map[name]
            param.copy_(src.to(param.device))


def _shared_param_names_fedbn(model):
    bn_param_names = set()
    for module_name, module in model.named_modules():
        if isinstance(module, _BatchNorm):
            for name, _ in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{name}" if module_name else name
                bn_param_names.add(full_name)
    return [name for name, _ in model.named_parameters() if name not in bn_param_names]


def _shared_param_names_fedper(model, personalization_prefixes):
    if not personalization_prefixes:
        return [name for name, _ in model.named_parameters()]

    def is_personalized(name):
        return any(name.startswith(prefix) for prefix in personalization_prefixes)

    return [name for name, _ in model.named_parameters() if not is_personalized(name)]


def _partition_size(x):
    if hasattr(x, "shape"):
        return int(x.shape[0])
    return len(x)


def _compute_client_weights(train_data, device_list):
    counts = []
    for device in device_list:
        data_obj = get_partition(train_data, device)
        count_obj = device(_partition_size)(data_obj)
        counts.append(sf.reveal(count_obj))

    total = float(sum(counts))
    if total <= 0:
        uniform = 1.0 / max(len(device_list), 1)
        return {device: uniform for device in device_list}
    return {device: count / total for device, count in zip(device_list, counts)}


def _build_optimizer(params, optimizer_name, lr, momentum):
    if optimizer_name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum)
    if optimizer_name == "adam":
        return optim.Adam(params, lr=lr)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _tensor_digest(tensor):
    tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = tensor.float().contiguous()
    data = tensor.numpy().tobytes()
    return {
        "l2": float(torch.norm(tensor).item()),
        "sum": float(torch.sum(tensor).item()),
        "abs_sum": float(torch.sum(torch.abs(tensor)).item()),
        "sha256": hashlib.sha256(data).hexdigest()[:12],
    }


def _client_param_digest(state, param_names):
    model = state["model"]
    param_map = dict(model.named_parameters())
    digest = {}
    with torch.no_grad():
        for name in param_names:
            param = param_map.get(name)
            if param is None:
                continue
            digest[name] = _tensor_digest(param)
    return digest


def _client_init(
    lr, seed, in_channels, num_classes, optimizer_name, momentum, device, arch
):
    torch.manual_seed(seed)
    model = build_model(in_channels, num_classes, arch=arch)
    if device == "cuda" and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        print(
            f"[client_init] device={device}, cuda_available={torch.cuda.is_available()}, "
            f"CUDA_VISIBLE_DEVICES={cuda_visible}"
        )
    model = model.to(device)
    optimizer = _build_optimizer(model.parameters(), optimizer_name, lr, momentum)
    return {"model": model, "optimizer": optimizer, "device": device}


def _client_train_one_round(
    state,
    x,
    y,
    shared_param_names,
    global_params,
    epochs,
    batch_size,
    lr,
    optimizer_name,
    momentum,
    debug_param_names=None,
    return_stats=False,
):
    model = state["model"]
    device = state.get("device", "cpu")
    debug_info = None
    if debug_param_names:
        debug_info = {"before_set": _client_param_digest(state, debug_param_names)}
    _set_named_params(model, shared_param_names, global_params)
    if debug_param_names:
        debug_info["after_set"] = _client_param_digest(state, debug_param_names)
    optimizer = _build_optimizer(model.parameters(), optimizer_name, lr, momentum)

    x, y = prepare_tensor(x, y)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    loss_sum = 0.0
    count = 0
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            batch_size_actual = yb.shape[0]
            loss_sum += loss.item() * batch_size_actual
            count += batch_size_actual

    state["optimizer"] = optimizer
    if debug_param_names:
        debug_info["after_train"] = _client_param_digest(state, debug_param_names)
        if return_stats:
            return (
                state,
                _get_named_params(model, shared_param_names),
                debug_info,
                {"loss_sum": loss_sum, "count": count},
            )
        return state, _get_named_params(model, shared_param_names), debug_info
    if return_stats:
        return (
            state,
            _get_named_params(model, shared_param_names),
            {"loss_sum": loss_sum, "count": count},
        )
    return state, _get_named_params(model, shared_param_names)


def _client_evaluate(state, x, y, shared_param_names, global_params, batch_size):
    model = state["model"]
    device = state.get("device", "cpu")
    _set_named_params(model, shared_param_names, global_params)
    model.eval()

    x, y = prepare_tensor(x, y)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, reduction="sum")
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == yb).sum().item()
            total += yb.shape[0]

    return {"loss": total_loss, "correct": total_correct, "total": total}


def _average_params(client_params):
    avg_params = []
    for param_group in zip(*client_params):
        stacked = torch.stack(param_group, dim=0)
        avg_params.append(torch.mean(stacked, dim=0))
    return avg_params


def _average_params_weighted(client_params, weights):
    avg_params = []
    for param_group in zip(*client_params):
        stacked = torch.stack(param_group, dim=0)
        weight_tensor = torch.tensor(weights, dtype=stacked.dtype)
        view_shape = [len(weights)] + [1] * (stacked.dim() - 1)
        avg_params.append(torch.sum(stacked * weight_tensor.view(view_shape), dim=0))
    return avg_params


def _run_personalized(
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
    strategy,
    logger=None,
):
    train_cfg = cfg["train"]
    rounds = train_cfg["rounds"]
    local_epochs = train_cfg["local_epochs"]
    batch_size = train_cfg["batch_size"]
    lr = train_cfg["lr"]
    optimizer_name = train_cfg.get("optimizer", "sgd")
    momentum = train_cfg.get("momentum", 0.0)
    seed = cfg["data"]["seed"]
    device = train_cfg.get("device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    runtime_cfg = cfg.get("runtime", {})
    gpu_per_client = runtime_cfg.get("gpu_per_party", 0) if device == "cuda" else 0
    eval_at_end = train_cfg.get("eval_at_end", True)
    eval_interval = max(int(train_cfg.get("eval_interval", 1)), 1)
    wandb_cfg = cfg.get("wandb", {})
    log_per_client = bool(wandb_cfg.get("log_per_client", False))
    early_cfg = cfg.get("early_stop", {})
    early_enabled = bool(early_cfg.get("enable", False)) and val_data is not None
    early_metric = early_cfg.get("metric", "val_acc")
    early_mode = early_cfg.get("mode", "max")
    early_patience = int(early_cfg.get("patience", 5))
    early_min_delta = float(early_cfg.get("min_delta", 0.0))
    model_arch = cfg.get("model", {}).get("arch", "convnet")
    want_train_stats = logger is not None

    torch.manual_seed(seed)
    template_model = build_model(in_channels, num_classes, arch=model_arch)
    personalized_param_names = ()
    debug_param_names = ()
    debug_head = False
    if strategy == "fedbn":
        shared_param_names = _shared_param_names_fedbn(template_model)
    elif strategy == "fedper":
        fedper_cfg = cfg.get("fedper", {})
        personalization_prefixes = fedper_cfg.get("personalization_prefixes", ["classifier."])
        if isinstance(personalization_prefixes, str):
            personalization_prefixes = [personalization_prefixes]
        shared_param_names = _shared_param_names_fedper(template_model, personalization_prefixes)
        personalized_param_names = [
            name for name, _ in template_model.named_parameters() if name not in shared_param_names
        ]
        bad_shared = [
            name
            for name in shared_param_names
            if any(name.startswith(prefix) for prefix in personalization_prefixes)
        ]
        if bad_shared:
            raise ValueError(
                "FedPer shared params should not include personalized keys: "
                + ", ".join(bad_shared)
            )
        debug_head = bool(fedper_cfg.get("debug_head", False))
        debug_param_names = [
            name
            for name, _ in template_model.named_parameters()
            if any(name.startswith(prefix) for prefix in personalization_prefixes)
        ]
        if not personalized_param_names:
            print(
                "[fedper_warn] No personalized params matched prefixes "
                f"{personalization_prefixes}; head may be aggregated."
            )
        print(
            "[fedper_init] personalization_prefixes="
            f"{personalization_prefixes}, shared_params={len(shared_param_names)}, "
            f"personalized_params={len(personalized_param_names)}"
        )
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    shared_param_names = tuple(shared_param_names)
    global_params = _get_named_params(template_model, shared_param_names)
    weight_map = _compute_client_weights(train_data, device_list)

    client_states = {}
    for idx, device_obj in enumerate(device_list):
        client_seed = seed if strategy != "fedper" else seed + idx + 1
        init_kwargs = {"num_returns": 1}
        if gpu_per_client:
            init_kwargs["num_gpus"] = gpu_per_client
        client_states[device_obj] = device_obj(_client_init, **init_kwargs)(
            lr,
            client_seed,
            in_channels,
            num_classes,
            optimizer_name,
            momentum,
            device,
            model_arch,
        )

    def _evaluate_split(split_data, split_label):
        if split_data is None or split_label is None:
            return None, None
        eval_stats = []
        for device_obj in device_list:
            state_obj = client_states[device_obj]
            data_obj = get_partition(split_data, device_obj)
            label_obj = get_partition(split_label, device_obj)
            eval_kwargs = {"num_returns": 1}
            if gpu_per_client:
                eval_kwargs["num_gpus"] = gpu_per_client
            stats_obj = device_obj(_client_evaluate, **eval_kwargs)(
                state_obj,
                data_obj,
                label_obj,
                shared_param_names,
                global_params,
                batch_size,
            )
            eval_stats.append(sf.reveal(stats_obj))

        total_loss = sum(s["loss"] for s in eval_stats)
        total_correct = sum(s["correct"] for s in eval_stats)
        total = sum(s["total"] for s in eval_stats)
        avg_loss = total_loss / max(total, 1)
        avg_acc = total_correct / max(total, 1)

        per_client = {}
        for device_obj, stats in zip(device_list, eval_stats):
            name = getattr(device_obj, "party", str(device_obj))
            count = stats["total"]
            loss = stats["loss"] / max(count, 1)
            acc = stats["correct"] / max(count, 1)
            per_client[name] = {"loss": loss, "accuracy": acc, "n": count}
        summary = {"loss": avg_loss, "accuracy": avg_acc}
        return summary, per_client

    best_metric = None
    bad_rounds = 0
    stopped_early = False

    for r in range(rounds):
        active_devices = list(device_list)
        random.shuffle(active_devices)

        client_params = []
        debug_infos = {}
        train_stats = {}
        for device_obj in active_devices:
            state_obj = client_states[device_obj]
            data_obj = get_partition(train_data, device_obj)
            label_obj = get_partition(train_label, device_obj)

            if debug_head and debug_param_names:
                train_kwargs = {"num_returns": 4 if want_train_stats else 3}
                if gpu_per_client:
                    train_kwargs["num_gpus"] = gpu_per_client
                train_result = device_obj(
                    _client_train_one_round, **train_kwargs
                )(
                    state_obj,
                    data_obj,
                    label_obj,
                    shared_param_names,
                    global_params,
                    local_epochs,
                    batch_size,
                    lr,
                    optimizer_name,
                    momentum,
                    debug_param_names,
                    want_train_stats,
                )
                if want_train_stats:
                    state_obj, params_obj, debug_obj, stats_obj = train_result
                    train_stats[device_obj] = sf.reveal(stats_obj)
                else:
                    state_obj, params_obj, debug_obj = train_result
                debug_infos[device_obj] = sf.reveal(debug_obj)
            else:
                train_kwargs = {"num_returns": 3 if want_train_stats else 2}
                if gpu_per_client:
                    train_kwargs["num_gpus"] = gpu_per_client
                train_result = device_obj(_client_train_one_round, **train_kwargs)(
                    state_obj,
                    data_obj,
                    label_obj,
                    shared_param_names,
                    global_params,
                    local_epochs,
                    batch_size,
                    lr,
                    optimizer_name,
                    momentum,
                    None,
                    want_train_stats,
                )
                if want_train_stats:
                    state_obj, params_obj, stats_obj = train_result
                    train_stats[device_obj] = sf.reveal(stats_obj)
                else:
                    state_obj, params_obj = train_result
            client_states[device_obj] = state_obj
            client_params.append(sf.reveal(params_obj))

        train_loss = None
        if want_train_stats:
            total_loss = 0.0
            total_count = 0
            for stats in train_stats.values():
                total_loss += stats.get("loss_sum", 0.0)
                total_count += stats.get("count", 0)
            if total_count > 0:
                train_loss = total_loss / total_count

        round_weights = [weight_map[device_obj] for device_obj in active_devices]
        weight_sum = sum(round_weights)
        if weight_sum > 0:
            round_weights = [w / weight_sum for w in round_weights]
            global_params = _average_params_weighted(client_params, round_weights)
        else:
            global_params = _average_params(client_params)
        print(f"Round {r + 1}/{rounds} finished.")
        if debug_head and debug_param_names:
            for device_obj in active_devices:
                name = getattr(device_obj, "party", str(device_obj))
                debug_info = debug_infos.get(device_obj, {})
                before_set = debug_info.get("before_set", {})
                after_set = debug_info.get("after_set", {})
                after_train = debug_info.get("after_train", {})
                for param_name in debug_param_names:
                    before = before_set.get(param_name)
                    after = after_set.get(param_name)
                    if before and after and before.get("sha256") != after.get("sha256"):
                        raise AssertionError(
                            "FedPer personalized params were overwritten by global update: "
                            f"{param_name} party={name} round={r + 1}"
                        )
                    print(
                        f"[fedper_head] round={r + 1} party={name} "
                        f"param={param_name} before_set={before} "
                        f"after_set={after} after_train={after_train.get(param_name)}"
                    )

        should_eval = (r + 1) % eval_interval == 0 or r == rounds - 1
        if should_eval and val_data is not None and val_label is not None:
            val_summary, val_per_client = _evaluate_split(val_data, val_label)
            if val_summary:
                metrics = {
                    "round": r + 1,
                    "val_loss": val_summary["loss"],
                    "val_acc": val_summary["accuracy"],
                }
                if train_loss is not None:
                    metrics["train_loss"] = train_loss
                if logger:
                    if log_per_client and val_per_client:
                        for name, stats in val_per_client.items():
                            metrics[f"val_acc_{name}"] = stats["accuracy"]
                            metrics[f"val_loss_{name}"] = stats["loss"]
                    if log_per_client and train_stats:
                        for device_obj, stats in train_stats.items():
                            name = getattr(device_obj, "party", str(device_obj))
                            count = stats.get("count", 0)
                            if count > 0:
                                metrics[f"train_loss_{name}"] = stats["loss_sum"] / count
                    logger.log(metrics, step=r + 1)

                if early_enabled:
                    if early_metric == "val_loss":
                        current_metric = val_summary["loss"]
                    else:
                        current_metric = val_summary["accuracy"]

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
                            {
                                "best_metric": best_metric,
                                "bad_rounds": bad_rounds,
                            },
                            step=r + 1,
                        )
                    if bad_rounds >= early_patience:
                        print(
                            "[early_stop] "
                            f"stop at round={r + 1} best={best_metric:.6f}"
                        )
                        stopped_early = True
                        break
        elif logger and train_loss is not None:
            metrics = {"round": r + 1, "train_loss": train_loss}
            if log_per_client and train_stats:
                for device_obj, stats in train_stats.items():
                    name = getattr(device_obj, "party", str(device_obj))
                    count = stats.get("count", 0)
                    if count > 0:
                        metrics[f"train_loss_{name}"] = stats["loss_sum"] / count
            logger.log(metrics, step=r + 1)

    if eval_at_end:
        test_summary, per_client = _evaluate_split(test_data, test_label)

        print("\nFinal evaluation on test set:")
        if test_summary:
            print({"loss": test_summary["loss"], "accuracy": test_summary["accuracy"]})
        else:
            print({"loss": None, "accuracy": None})
        print("Per-client evaluation on test set:")
        print(per_client)
        summary = {"acc_global_weighted": test_summary["accuracy"] if test_summary else 0.0}
        for name, stats in per_client.items():
            summary[f"acc_{name}"] = stats["accuracy"]
            summary[f"n_{name}"] = stats["n"]
        print("Weighted accuracy summary:")
        print(summary)
        if logger and test_summary:
            metrics = {
                "test_loss": test_summary["loss"],
                "test_acc": test_summary["accuracy"],
            }
            if log_per_client and per_client:
                for name, stats in per_client.items():
                    metrics[f"test_acc_{name}"] = stats["accuracy"]
                    metrics[f"test_loss_{name}"] = stats["loss"]
            logger.log(metrics)


def run_fedbn(
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
    logger=None,
):
    _run_personalized(
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
        strategy="fedbn",
        logger=logger,
    )


def run_fedper(
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
    logger=None,
):
    _run_personalized(
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
        strategy="fedper",
        logger=logger,
    )
