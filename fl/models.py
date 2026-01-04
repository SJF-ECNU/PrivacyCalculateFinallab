import torch
from torch import nn, optim

from secretflow.ml.nn.core.torch import BaseModule, TorchModel, metric_wrapper, optim_wrapper
from torchmetrics import Metric


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


class CpuAccuracy(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds, target):
        preds = preds.detach()
        target = target.detach()
        if preds.is_cuda:
            preds = preds.cpu()
        if target.is_cuda:
            target = target.cpu()
        if preds.ndim > 1:
            preds = preds.argmax(dim=1)
        if target.ndim > 1:
            target = target.argmax(dim=1)
        self.correct += (preds == target).sum()
        self.total += torch.tensor(target.numel(), dtype=torch.long)

    def compute(self):
        if self.total == 0:
            return torch.tensor(0.0)
        return self.correct.float() / self.total.float()


def build_torch_model_def(in_channels, num_classes, lr):
    loss_fn = nn.CrossEntropyLoss
    optim_fn = optim_wrapper(optim.Adam, lr=lr)
    return TorchModel(
        model_fn=lambda: build_model(in_channels, num_classes),
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[metric_wrapper(CpuAccuracy)],
    )
