import torch

from secretflow.data.ndarray import FedNdarray


def get_model_params(model):
    return [p.detach().cpu() for p in model.parameters()]


def set_model_params(model, params):
    with torch.no_grad():
        for p, src in zip(model.parameters(), params):
            p.copy_(src.to(p.device))


def prepare_tensor(x, y, device=None):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if not torch.is_tensor(y):
        y = torch.tensor(y)
    x = x.float()
    if y.ndim > 1:
        y = y.argmax(dim=-1)
    y = y.long()
    if device is not None:
        x = x.to(device)
        y = y.to(device)
    return x, y


def get_partition(fed_data, device):
    if isinstance(fed_data, FedNdarray):
        return fed_data.partitions[device]
    if hasattr(fed_data, "partitions"):
        return fed_data.partitions[device]
    if isinstance(fed_data, dict):
        return fed_data[device]
    raise TypeError("Unsupported federated data type.")
