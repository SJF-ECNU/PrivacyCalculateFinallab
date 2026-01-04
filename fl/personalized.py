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


def _client_init(lr, seed, in_channels, num_classes, optimizer_name, momentum, device):
    torch.manual_seed(seed)
    model = build_model(in_channels, num_classes)
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
):
    model = state["model"]
    device = state.get("device", "cpu")
    _set_named_params(model, shared_param_names, global_params)
    optimizer = _build_optimizer(model.parameters(), optimizer_name, lr, momentum)

    x, y = prepare_tensor(x, y)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()

    state["optimizer"] = optimizer
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
    test_data,
    test_label,
    in_channels,
    num_classes,
    strategy,
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
    eval_at_end = train_cfg.get("eval_at_end", True)

    torch.manual_seed(seed)
    template_model = build_model(in_channels, num_classes)
    if strategy == "fedbn":
        shared_param_names = _shared_param_names_fedbn(template_model)
    elif strategy == "fedper":
        fedper_cfg = cfg.get("fedper", {})
        personalization_prefixes = fedper_cfg.get("personalization_prefixes", ["classifier."])
        if isinstance(personalization_prefixes, str):
            personalization_prefixes = [personalization_prefixes]
        shared_param_names = _shared_param_names_fedper(template_model, personalization_prefixes)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    shared_param_names = tuple(shared_param_names)
    global_params = _get_named_params(template_model, shared_param_names)
    weight_map = _compute_client_weights(train_data, device_list)

    client_states = {}
    for idx, device_obj in enumerate(device_list):
        client_seed = seed if strategy != "fedper" else seed + idx + 1
        client_states[device_obj] = device_obj(_client_init)(
            lr, client_seed, in_channels, num_classes, optimizer_name, momentum, device
        )

    for r in range(rounds):
        active_devices = list(device_list)
        random.shuffle(active_devices)

        client_params = []
        for device_obj in active_devices:
            state_obj = client_states[device_obj]
            data_obj = get_partition(train_data, device_obj)
            label_obj = get_partition(train_label, device_obj)

            state_obj, params_obj = device_obj(_client_train_one_round, num_returns=2)(
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
            )
            client_states[device_obj] = state_obj
            client_params.append(sf.reveal(params_obj))

        round_weights = [weight_map[device_obj] for device_obj in active_devices]
        weight_sum = sum(round_weights)
        if weight_sum > 0:
            round_weights = [w / weight_sum for w in round_weights]
            global_params = _average_params_weighted(client_params, round_weights)
        else:
            global_params = _average_params(client_params)
        print(f"Round {r + 1}/{rounds} finished.")

    if eval_at_end:
        eval_stats = []
        for device_obj in device_list:
            state_obj = client_states[device_obj]
            data_obj = get_partition(test_data, device_obj)
            label_obj = get_partition(test_label, device_obj)
            stats_obj = device_obj(_client_evaluate)(
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

        print("\nFinal evaluation on test set:")
        print({"loss": avg_loss, "accuracy": avg_acc})
        per_client = {}
        for device_obj, stats in zip(device_list, eval_stats):
            name = getattr(device_obj, "party", str(device_obj))
            count = stats["total"]
            loss = stats["loss"] / max(count, 1)
            acc = stats["correct"] / max(count, 1)
            per_client[name] = {"loss": loss, "accuracy": acc}
        print("Per-client evaluation on test set:")
        print(per_client)


def run_fedbn(
    cfg, device_list, train_data, train_label, test_data, test_label, in_channels, num_classes
):
    _run_personalized(
        cfg,
        device_list,
        train_data,
        train_label,
        test_data,
        test_label,
        in_channels,
        num_classes,
        strategy="fedbn",
    )


def run_fedper(
    cfg, device_list, train_data, train_label, test_data, test_label, in_channels, num_classes
):
    _run_personalized(
        cfg,
        device_list,
        train_data,
        train_label,
        test_data,
        test_label,
        in_channels,
        num_classes,
        strategy="fedper",
    )
