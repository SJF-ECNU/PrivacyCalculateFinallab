import random

import secretflow as sf

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .models import build_model
from .utils import get_model_params, get_partition, prepare_tensor, set_model_params


def _resolve_device(device_str):
    if not device_str or device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def client_init(lr, seed, in_channels, num_classes, device_str):
    torch.manual_seed(seed)
    device = _resolve_device(device_str)
    model = build_model(in_channels, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    h = [torch.zeros_like(p) for p in model.parameters()]
    return {"model": model, "optimizer": optimizer, "h": h, "device": str(device)}


def client_train_one_round(state, x, y, global_params, alpha, epochs, batch_size, lr):
    model = state["model"]
    h = state["h"]
    device = torch.device(state["device"])

    set_model_params(model, global_params)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    x, y = prepare_tensor(x, y, device)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    global_params = [g.to(device) for g in global_params]

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            base_loss = F.cross_entropy(logits, yb)

            reg = 0.0
            lin = 0.0
            for p, g, h_i in zip(model.parameters(), global_params, h):
                reg = reg + torch.sum((p - g) ** 2)
                lin = lin + torch.sum(p * h_i)

            loss = base_loss + 0.5 * alpha * reg - lin
            loss.backward()
            optimizer.step()

    new_h = []
    for p, g, h_i in zip(model.parameters(), global_params, h):
        new_h.append(h_i - alpha * (p.detach() - g))

    state["optimizer"] = optimizer
    state["h"] = new_h
    return state, get_model_params(model)


def client_evaluate(in_channels, num_classes, global_params, x, y, batch_size, device_str):
    device = _resolve_device(device_str)
    model = build_model(in_channels, num_classes).to(device)
    set_model_params(model, global_params)
    model.eval()

    x, y = prepare_tensor(x, y, device)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, reduction="sum")
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == yb).sum().item()
            total += yb.shape[0]

    return {"loss": total_loss, "correct": total_correct, "total": total}


def run_feddyn(
    cfg, device_list, train_data, train_label, test_data, test_label, in_channels, num_classes
):
    train_cfg = cfg["train"]
    rounds = train_cfg["rounds"]
    local_epochs = train_cfg["local_epochs"]
    batch_size = train_cfg["batch_size"]
    lr = train_cfg["lr"]
    alpha = cfg["feddyn"]["alpha"]
    seed = cfg["data"]["seed"]
    device_str = train_cfg.get("device", "auto")

    torch.manual_seed(seed)
    global_model = build_model(in_channels, num_classes)
    global_params = get_model_params(global_model)
    global_h = [torch.zeros_like(p) for p in global_params]

    client_states = {}
    for device in device_list:
        client_states[device] = device(client_init)(
            lr, seed, in_channels, num_classes, device_str
        )

    for r in range(rounds):
        active_devices = list(device_list)
        random.shuffle(active_devices)

        client_params = []
        for device in active_devices:
            state_obj = client_states[device]
            data_obj = get_partition(train_data, device)
            label_obj = get_partition(train_label, device)

            state_obj, params_obj = device(client_train_one_round, num_returns=2)(
                state_obj,
                data_obj,
                label_obj,
                global_params,
                alpha,
                local_epochs,
                batch_size,
                lr,
            )
            client_states[device] = state_obj
            client_params.append(sf.reveal(params_obj))

        avg_params = []
        deltas = []
        for idx, param_group in enumerate(zip(*client_params)):
            stacked = torch.stack(param_group, dim=0)
            avg_params.append(torch.mean(stacked, dim=0))
            deltas.append(torch.sum(stacked - global_params[idx], dim=0))

        m = len(device_list)
        global_h = [h - (alpha / m) * delta for h, delta in zip(global_h, deltas)]
        global_params = [avg - (1.0 / alpha) * h for avg, h in zip(avg_params, global_h)]

        print(f"Round {r + 1}/{rounds} finished.")

    eval_stats = []
    for device in device_list:
        data_obj = get_partition(test_data, device)
        label_obj = get_partition(test_label, device)
        stats_obj = device(client_evaluate)(
            in_channels,
            num_classes,
            global_params,
            data_obj,
            label_obj,
            batch_size,
            device_str,
        )
        eval_stats.append(sf.reveal(stats_obj))

    total_loss = sum(s["loss"] for s in eval_stats)
    total_correct = sum(s["correct"] for s in eval_stats)
    total = sum(s["total"] for s in eval_stats)
    avg_loss = total_loss / max(total, 1)
    avg_acc = total_correct / max(total, 1)

    print("\nFinal evaluation on test set:")
    print({"loss": avg_loss, "accuracy": avg_acc})
