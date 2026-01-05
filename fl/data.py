import numpy as np

from secretflow.utils.simulation.data.ndarray import create_ndarray
from torchvision import datasets


SUPPORTED_DATASETS = {"mnist", "cifar10", "cifar100"}


def _normalize_parts(parts):
    total = float(sum(parts.values()))
    if total <= 0:
        uniform = 1.0 / max(len(parts), 1)
        return {k: uniform for k in parts}
    return {k: v / total for k, v in parts.items()}


def _partition_counts(total, weights):
    counts = [int(total * w) for w in weights[:-1]]
    counts.append(total - sum(counts))
    return counts


def _split_train_val(x, y, parts, val_ratio, val_max_samples, seed):
    if val_ratio <= 0:
        return x, y, None, None, parts, None

    parts = _normalize_parts(parts)
    devices = list(parts.keys())
    weights = [parts[d] for d in devices]
    total = len(x)
    if total == 0:
        return x, y, None, None, parts, None

    counts = _partition_counts(total, weights)
    rng = np.random.default_rng(seed)

    train_chunks = []
    val_chunks = []
    train_counts = []
    val_counts = []
    offset = 0
    for count in counts:
        x_part = x[offset : offset + count]
        y_part = y[offset : offset + count]
        offset += count
        if count <= 0:
            train_chunks.append((x_part, y_part))
            val_chunks.append((x_part[:0], y_part[:0]))
            train_counts.append(0)
            val_counts.append(0)
            continue

        perm = rng.permutation(count)
        val_count = int(count * val_ratio)
        if val_max_samples is not None:
            val_count = min(val_count, int(val_max_samples))
        val_count = min(val_count, count)
        if val_count >= count:
            val_count = max(count - 1, 0)

        val_idx = perm[:val_count]
        train_idx = perm[val_count:]
        val_x = x_part[val_idx]
        val_y = y_part[val_idx]
        train_x = x_part[train_idx]
        train_y = y_part[train_idx]

        train_chunks.append((train_x, train_y))
        val_chunks.append((val_x, val_y))
        train_counts.append(len(train_x))
        val_counts.append(len(val_x))

    train_x = np.concatenate([c[0] for c in train_chunks], axis=0)
    train_y = np.concatenate([c[1] for c in train_chunks], axis=0)
    train_parts = _normalize_parts(
        {device: count for device, count in zip(devices, train_counts)}
    )

    val_total = int(sum(val_counts))
    if val_total <= 0:
        return train_x, train_y, None, None, train_parts, None

    val_x = np.concatenate([c[0] for c in val_chunks], axis=0)
    val_y = np.concatenate([c[1] for c in val_chunks], axis=0)
    val_parts = _normalize_parts(
        {device: count for device, count in zip(devices, val_counts)}
    )
    return train_x, train_y, val_x, val_y, train_parts, val_parts


def load_raw_dataset(name, root, train, download):
    if name == "mnist":
        dataset = datasets.MNIST(root=root, train=train, download=download)
        x = dataset.data.numpy().astype("float32") / 255.0
        x = x[:, None, :, :]
        y = np.array(dataset.targets)
        in_channels = 1
        num_classes = 10
    elif name == "cifar10":
        dataset = datasets.CIFAR10(root=root, train=train, download=download)
        x = dataset.data.astype("float32") / 255.0
        x = np.transpose(x, (0, 3, 1, 2))
        y = np.array(dataset.targets)
        in_channels = 3
        num_classes = 10
    elif name == "cifar100":
        dataset = datasets.CIFAR100(root=root, train=train, download=download)
        x = dataset.data.astype("float32") / 255.0
        x = np.transpose(x, (0, 3, 1, 2))
        y = np.array(dataset.targets)
        in_channels = 3
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return x, y, in_channels, num_classes


def build_fed_dataset(cfg, device_map):
    data_cfg = cfg["data"]
    name = data_cfg.get("dataset") or data_cfg.get("name")
    if not name:
        raise ValueError("Missing data.dataset in config.")
    datasets_cfg = data_cfg.get("datasets", {})
    dataset_cfg = datasets_cfg.get(name, {})
    root = dataset_cfg.get("root", data_cfg.get("root", "data"))
    download = dataset_cfg.get("download", data_cfg.get("download", False))
    categorical_y = dataset_cfg.get(
        "categorical_y", data_cfg.get("categorical_y", True)
    )
    shuffle = data_cfg.get("shuffle", False)
    seed = data_cfg.get("seed", 1234)
    val_ratio = data_cfg.get("val_ratio", 0.0)
    val_max_samples = data_cfg.get("val_max_samples")
    parts_cfg = dataset_cfg.get("parts", data_cfg.get("parts"))
    if not parts_cfg:
        raise ValueError(f"Missing parts config for dataset {name}.")

    parts = {device_map[k]: v for k, v in parts_cfg.items()}

    x_train, y_train, in_channels, num_classes = load_raw_dataset(
        name, root, True, download
    )
    x_test, y_test, _, _ = load_raw_dataset(name, root, False, download)

    if categorical_y:
        y_train = np.eye(num_classes, dtype=np.float32)[y_train]
        y_test = np.eye(num_classes, dtype=np.float32)[y_test]

    if shuffle:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(x_train))
        x_train = x_train[perm]
        y_train = y_train[perm]

    x_train, y_train, x_val, y_val, train_parts, val_parts = _split_train_val(
        x_train, y_train, parts, val_ratio, val_max_samples, seed
    )

    train_data = create_ndarray(
        x_train,
        parts=train_parts,
        axis=0,
        shuffle=False,
    )
    train_label = create_ndarray(
        y_train,
        parts=train_parts,
        axis=0,
        shuffle=False,
    )
    val_data = None
    val_label = None
    if val_parts and x_val is not None and y_val is not None and len(x_val) > 0:
        val_data = create_ndarray(
            x_val,
            parts=val_parts,
            axis=0,
            shuffle=False,
        )
        val_label = create_ndarray(
            y_val,
            parts=val_parts,
            axis=0,
            shuffle=False,
        )
    test_data = create_ndarray(
        x_test,
        parts=parts,
        axis=0,
        shuffle=False,
    )
    test_label = create_ndarray(
        y_test,
        parts=parts,
        axis=0,
        shuffle=False,
    )

    return (
        (train_data, train_label),
        (val_data, val_label),
        (test_data, test_label),
        in_channels,
        num_classes,
    )
