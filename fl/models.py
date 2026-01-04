from torch import nn, optim

from secretflow.ml.nn.core.torch import (
    BaseModule,
    TorchModel,
    metric_wrapper,
    optim_wrapper,
)
from torchmetrics import Accuracy, Precision


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

    def update_metrics(self, y_pred, y_true):
        if hasattr(y_pred, "is_cuda") and y_pred.is_cuda:
            y_pred = y_pred.detach().cpu()
        else:
            y_pred = y_pred.detach()
        if hasattr(y_true, "is_cuda") and y_true.is_cuda:
            y_true = y_true.detach().cpu()
        else:
            y_true = y_true.detach()
        return super().update_metrics(y_pred, y_true)


def build_model(in_channels, num_classes):
    return ConvNet(in_channels, num_classes)


def build_torch_model_def(in_channels, num_classes, lr, enable_metrics=True):
    loss_fn = nn.CrossEntropyLoss
    optim_fn = optim_wrapper(optim.Adam, lr=lr)
    metrics = []
    if enable_metrics:
        metrics = [
            metric_wrapper(Accuracy, task="multiclass", num_classes=num_classes, average="micro"),
            metric_wrapper(Precision, task="multiclass", num_classes=num_classes, average="micro"),
        ]
    return TorchModel(
        model_fn=lambda: build_model(in_channels, num_classes),
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=metrics,
    )
