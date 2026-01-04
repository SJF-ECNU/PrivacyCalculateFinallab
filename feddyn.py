import random

import secretflow as sf

from secretflow.utils.simulation.datasets import load_mnist

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# 0. Init SecretFlow runtime
# -----------------------------
print(f"The version of SecretFlow: {sf.__version__}")

# In case you have a running secretflow runtime already.
sf.shutdown()

sf.init(["alice", "bob", "charlie"], address="local")
alice, bob, charlie = sf.PYU("alice"), sf.PYU("bob"), sf.PYU("charlie")


# -----------------------------
# 1. Define model (Torch)
# -----------------------------
class ConvNet(nn.Module):
    """Small ConvNet for MNIST (Torch)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc_in_dim = 192
        self.fc = nn.Linear(self.fc_in_dim, 10)

    def forward(self, x):
        # x: [N, 1, 28, 28]
        x = F.relu(F.max_pool2d(self.conv1(x), 3))  # -> [N, 3, 8, 8]
        x = x.view(-1, self.fc_in_dim)              # -> [N, 192]
        x = self.fc(x)                              # -> [N, 10]
        # CrossEntropyLoss expects logits (no softmax).
        return x


def build_model():
    return ConvNet()


def get_model_params(model):
    return [p.detach().cpu() for p in model.parameters()]


def set_model_params(model, params):
    with torch.no_grad():
        for p, src in zip(model.parameters(), params):
            p.copy_(src)


def _prepare_tensor(x, y):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if not torch.is_tensor(y):
        y = torch.tensor(y)
    x = x.float()
    if y.ndim > 1:
        y = y.argmax(dim=-1)
    y = y.long()
    return x, y


def _get_partition(fed_data, device):
    if hasattr(fed_data, "partitions"):
        return fed_data.partitions[device]
    if isinstance(fed_data, dict):
        return fed_data[device]
    raise TypeError("Unsupported federated data type.")


# -----------------------------
# 2. Client-side functions
# -----------------------------
def client_init(lr, seed):
    torch.manual_seed(seed)
    model = build_model()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    h = [torch.zeros_like(p) for p in model.parameters()]
    return {"model": model, "optimizer": optimizer, "h": h}


def client_train_one_round(state, x, y, global_params, alpha, epochs, batch_size, lr):
    model = state["model"]
    optimizer = state["optimizer"]
    h = state["h"]

    set_model_params(model, global_params)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    x, y = _prepare_tensor(x, y)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

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


def client_evaluate(global_params, x, y, batch_size):
    model = build_model()
    set_model_params(model, global_params)
    model.eval()

    x, y = _prepare_tensor(x, y)
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


# -----------------------------
# 3. Load dataset (Torch format)
# -----------------------------
(train_data, train_label), (test_data, test_label) = load_mnist(
    parts={alice: 0.4, bob: 0.6},
    normalized_x=True,
    categorical_y=True,
    is_torch=True,
)


# -----------------------------
# 4. FedDyn training (manual)
# -----------------------------
device_list = [alice, bob]
server = charlie

rounds = 20
local_epochs = 1
batch_size = 32
alpha = 0.01
lr = 1e-2

seed = 1234
torch.manual_seed(seed)
global_model = build_model()
global_params = get_model_params(global_model)
global_h = [torch.zeros_like(p) for p in global_params]

client_states = {}
for device in device_list:
    client_states[device] = device(client_init)(lr, seed)

for r in range(rounds):
    active_devices = list(device_list)
    random.shuffle(active_devices)

    client_params = []
    for device in active_devices:
        state_obj = client_states[device]
        data_obj = _get_partition(train_data, device)
        label_obj = _get_partition(train_label, device)

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
    for param_group in zip(*client_params):
        stacked = torch.stack(param_group, dim=0)
        avg_params.append(torch.mean(stacked, dim=0))
        deltas.append(torch.sum(stacked - global_params[len(deltas)], dim=0))

    m = len(device_list)
    global_h = [h - (alpha / m) * delta for h, delta in zip(global_h, deltas)]
    global_params = [avg - (1.0 / alpha) * h for avg, h in zip(avg_params, global_h)]

    print(f"Round {r + 1}/{rounds} finished.")


# -----------------------------
# 5. Evaluate
# -----------------------------
eval_stats = []
for device in device_list:
    data_obj = _get_partition(test_data, device)
    label_obj = _get_partition(test_label, device)
    stats_obj = device(client_evaluate)(
        global_params,
        data_obj,
        label_obj,
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
