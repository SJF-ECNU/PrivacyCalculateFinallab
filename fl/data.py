import numpy as np

from secretflow.utils.simulation.data.ndarray import create_ndarray
from torchvision import datasets


SUPPORTED_DATASETS = {"mnist", "cifar10", "cifar100"}


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

    train_data = create_ndarray(
        x_train,
        parts=parts,
        axis=0,
        shuffle=False,
    )
    train_label = create_ndarray(
        y_train,
        parts=parts,
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

    return (train_data, train_label), (test_data, test_label), in_channels, num_classes
