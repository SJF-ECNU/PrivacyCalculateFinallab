from torch import nn, optim

from secretflow.ml.nn.core.torch import (
    BaseModule,
    TorchModel,
    metric_wrapper,
    optim_wrapper,
)
from torchmetrics import Accuracy, Precision


class DeviceAwareAccuracy(Accuracy):
    def update(self, preds, target):
        if preds.is_cuda:
            preds = preds.detach().cpu()
        if target.is_cuda:
            target = target.detach().cpu()
        self.to("cpu")
        return super().update(preds, target)


class DeviceAwarePrecision(Precision):
    def update(self, preds, target):
        if preds.is_cuda:
            preds = preds.detach().cpu()
        if target.is_cuda:
            target = target.detach().cpu()
        self.to("cpu")
        return super().update(preds, target)


class ConvNet(BaseModule):
    """Small ConvNet for MNIST/CIFAR (Torch)."""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def build_model(in_channels, num_classes):
    return ConvNet(in_channels, num_classes)


def build_torch_model_def(in_channels, num_classes, lr):
    loss_fn = nn.CrossEntropyLoss
    optim_fn = optim_wrapper(optim.Adam, lr=lr)
    return TorchModel(
        model_fn=lambda: build_model(in_channels, num_classes),
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[
            metric_wrapper(DeviceAwareAccuracy, task="multiclass", num_classes=num_classes, average="micro"),
            metric_wrapper(DeviceAwarePrecision, task="multiclass", num_classes=num_classes, average="micro"),
        ],
    )
