import torch
from torch import nn, optim

from secretflow.ml.nn.core.torch import (
    BaseModule,
    TorchModel,
    metric_wrapper,
    optim_wrapper,
)
from torchmetrics import Accuracy, Precision


class FloatBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, **kwargs):
        super().__init__(num_features, **kwargs)
        if self.num_batches_tracked is not None:
            self.register_buffer(
                "num_batches_tracked",
                torch.zeros(1, dtype=torch.float32),
                persistent=True,
            )


class ConvNet(BaseModule):
    """Small ConvNet for MNIST/CIFAR (Torch)."""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            FloatBatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            FloatBatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def get_weights(self):
        return [p.detach().cpu().numpy() for p in self.parameters()]

    def set_weights(self, weights):
        with torch.no_grad():
            for param, src in zip(self.parameters(), weights):
                if not torch.is_tensor(src):
                    src = torch.tensor(src)
                param.copy_(src.to(param.device))

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


def build_torch_model_def(
    in_channels,
    num_classes,
    lr,
    enable_metrics=True,
    optimizer_name="adam",
    momentum=0.0,
    device="cpu",
):
    loss_fn = nn.CrossEntropyLoss
    if optimizer_name == "sgd":
        optim_fn = optim_wrapper(optim.SGD, lr=lr, momentum=momentum)
    elif optimizer_name == "adam":
        optim_fn = optim_wrapper(optim.Adam, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    def _model_fn():
        model = build_model(in_channels, num_classes)
        use_cuda = device == "cuda" and torch.cuda.is_available()
        if use_cuda:
            model = model.to("cuda")
        return model
    metrics = []
    if enable_metrics:
        metrics = [
            metric_wrapper(Accuracy, task="multiclass", num_classes=num_classes, average="micro"),
            metric_wrapper(Precision, task="multiclass", num_classes=num_classes, average="micro"),
        ]
    return TorchModel(
        model_fn=_model_fn,
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=metrics,
    )
