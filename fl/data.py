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


def _split_client_train_val(x, y, val_ratio, val_max_samples, rng):
    if val_ratio <= 0 or len(x) == 0:
        return x, y, x[:0], y[:0]
    count = len(x)
    perm = rng.permutation(count)
    val_count = int(count * val_ratio)
    if val_max_samples is not None:
        val_count = min(val_count, int(val_max_samples))
    val_count = min(val_count, count)
    if val_count >= count:
        val_count = max(count - 1, 0)
    val_idx = perm[:val_count]
    train_idx = perm[val_count:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def _partition_indices_by_parts(num_samples, parts, rng):
    parts = _normalize_parts(parts)
    devices = list(parts.keys())
    weights = [parts[d] for d in devices]
    counts = _partition_counts(num_samples, weights)
    indices = rng.permutation(num_samples)
    split_points = np.cumsum(counts)[:-1]
    splits = np.split(indices, split_points)
    return {device: split for device, split in zip(devices, splits)}


def _partition_indices_dirichlet(y, num_clients, alpha, rng, min_size):
    labels = np.asarray(y)
    classes = np.unique(labels)
    class_indices = {c: np.where(labels == c)[0] for c in classes}
    while True:
        client_indices = [[] for _ in range(num_clients)]
        for c in classes:
            idx = rng.permutation(class_indices[c])
            proportions = rng.dirichlet([alpha] * num_clients)
            split_points = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
            splits = np.split(idx, split_points)
            for client_id, split in enumerate(splits):
                client_indices[client_id].extend(split.tolist())
        sizes = [len(idx) for idx in client_indices]
        if min(sizes) >= min_size:
            break
    return [np.array(idx, dtype=np.int64) for idx in client_indices]


def _partition_indices_label_skew(y, num_clients, classes_per_client, rng):
    labels = np.asarray(y)
    num_samples = len(labels)
    if num_samples == 0:
        return [np.array([], dtype=np.int64) for _ in range(num_clients)]
    num_shards = max(num_clients * classes_per_client, num_clients)
    idxs = np.argsort(labels)
    shard_size = max(num_samples // num_shards, 1)
    shards = []
    for i in range(num_shards - 1):
        shards.append(idxs[i * shard_size : (i + 1) * shard_size])
    shards.append(idxs[(num_shards - 1) * shard_size :])
    rng.shuffle(shards)
    client_indices = [[] for _ in range(num_clients)]
    for shard_id, shard in enumerate(shards):
        client_indices[shard_id % num_clients].extend(shard.tolist())
    return [np.array(idx, dtype=np.int64) for idx in client_indices]


def _maybe_one_hot(y, num_classes, categorical):
    if not categorical:
        return y
    y = np.asarray(y, dtype=np.int64)
    return np.eye(num_classes, dtype=np.float32)[y]


def _apply_feature_shift(x, params, rng, noise_std):
    scale = params["scale"]
    shift = params["shift"]
    x = x.astype("float32")
    x = x * scale + shift
    if noise_std > 0:
        x = x + rng.normal(0.0, noise_std, size=x.shape).astype("float32")
    return np.clip(x, 0.0, 1.0)


def _feature_shift_params(num_clients, num_channels, cfg, rng):
    scale_range = cfg.get("scale_range", [0.5, 1.5])
    shift_range = cfg.get("shift_range", [-0.2, 0.2])
    channel_wise = bool(cfg.get("channel_wise", True))
    params = []
    for _ in range(num_clients):
        if channel_wise:
            scale = rng.uniform(scale_range[0], scale_range[1], size=(num_channels, 1, 1))
            shift = rng.uniform(shift_range[0], shift_range[1], size=(num_channels, 1, 1))
        else:
            scale = float(rng.uniform(scale_range[0], scale_range[1]))
            shift = float(rng.uniform(shift_range[0], shift_range[1]))
        params.append({"scale": scale, "shift": shift})
    return params


def _assemble_fed_ndarray(partitions, devices):
    arrays = [partitions[device] for device in devices]
    sizes = [len(arr) for arr in arrays]
    total = int(sum(sizes))
    if total == 0:
        return None
    combined = np.concatenate(arrays, axis=0)
    parts = _normalize_parts({device: size for device, size in zip(devices, sizes)})
    return create_ndarray(combined, parts=parts, axis=0, shuffle=False)


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
    partition_cfg = data_cfg.get("partitioning", {}) or data_cfg.get("noniid", {})
    partition_type = partition_cfg.get("type", "iid")
    partition_seed = partition_cfg.get("seed", seed)
    parts_cfg = dataset_cfg.get("parts", data_cfg.get("parts"))
    if not parts_cfg:
        raise ValueError(f"Missing parts config for dataset {name}.")

    parts = {device_map[k]: v for k, v in parts_cfg.items()}
    devices = list(parts.keys())

    x_train, y_train, in_channels, num_classes = load_raw_dataset(
        name, root, True, download
    )
    x_test, y_test, _, _ = load_raw_dataset(name, root, False, download)

    if shuffle:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(x_train))
        x_train = x_train[perm]
        y_train = y_train[perm]

    rng_train = np.random.default_rng(partition_seed)
    rng_test = np.random.default_rng(partition_seed + 1)
    num_clients = len(devices)

    def build_indices(y, base_type, rng):
        if base_type in {"iid", "quantity"}:
            return _partition_indices_by_parts(len(y), parts, rng)
        if base_type == "label_skew":
            classes_per_client = int(
                partition_cfg.get("label_skew", {}).get("classes_per_client", 2)
            )
            indices_list = _partition_indices_label_skew(
                y, num_clients, classes_per_client, rng
            )
            return {device: idx for device, idx in zip(devices, indices_list)}
        if base_type == "dirichlet":
            dir_cfg = partition_cfg.get("dirichlet", {})
            alpha = float(dir_cfg.get("alpha", 0.5))
            min_size = int(dir_cfg.get("min_size", 1))
            indices_list = _partition_indices_dirichlet(
                y, num_clients, alpha, rng, min_size
            )
            return {device: idx for device, idx in zip(devices, indices_list)}
        raise ValueError(f"Unsupported partition type: {base_type}")

    if partition_type == "feature_shift":
        feature_cfg = partition_cfg.get("feature_shift", {})
        base_type = feature_cfg.get("base_type", "iid")
        train_indices = build_indices(y_train, base_type, rng_train)
        test_indices = build_indices(y_test, base_type, rng_test)
    else:
        train_indices = build_indices(y_train, partition_type, rng_train)
        test_indices = build_indices(y_test, partition_type, rng_test)

    train_x_parts = {}
    train_y_parts = {}
    val_x_parts = {}
    val_y_parts = {}
    test_x_parts = {}
    test_y_parts = {}

    if partition_type == "feature_shift":
        feature_cfg = partition_cfg.get("feature_shift", {})
        noise_std = float(feature_cfg.get("noise_std", 0.0))
        params = _feature_shift_params(num_clients, in_channels, feature_cfg, rng_train)
    else:
        params = None
        noise_std = 0.0

    for idx, device in enumerate(devices):
        train_idx = train_indices[device]
        test_idx = test_indices[device]
        train_x = x_train[train_idx]
        train_y = y_train[train_idx]
        test_x = x_test[test_idx]
        test_y = y_test[test_idx]

        if params is not None:
            train_x = _apply_feature_shift(
                train_x,
                params[idx],
                np.random.default_rng(partition_seed + 10 + idx),
                noise_std,
            )
            test_x = _apply_feature_shift(
                test_x,
                params[idx],
                np.random.default_rng(partition_seed + 100 + idx),
                noise_std,
            )

        split_rng = np.random.default_rng(partition_seed + 1000 + idx)
        train_x, train_y, val_x, val_y = _split_client_train_val(
            train_x, train_y, val_ratio, val_max_samples, split_rng
        )

        train_x_parts[device] = train_x
        train_y_parts[device] = _maybe_one_hot(train_y, num_classes, categorical_y)
        val_x_parts[device] = val_x
        val_y_parts[device] = _maybe_one_hot(val_y, num_classes, categorical_y)
        test_x_parts[device] = test_x
        test_y_parts[device] = _maybe_one_hot(test_y, num_classes, categorical_y)

    train_data = _assemble_fed_ndarray(train_x_parts, devices)
    train_label = _assemble_fed_ndarray(train_y_parts, devices)
    val_data = _assemble_fed_ndarray(val_x_parts, devices)
    val_label = _assemble_fed_ndarray(val_y_parts, devices)
    test_data = _assemble_fed_ndarray(test_x_parts, devices)
    test_label = _assemble_fed_ndarray(test_y_parts, devices)

    return (
        (train_data, train_label),
        (val_data, val_label),
        (test_data, test_label),
        in_channels,
        num_classes,
    )
