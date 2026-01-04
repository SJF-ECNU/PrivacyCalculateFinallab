import secretflow as sf

from secretflow.ml.nn.core.torch import (
    BaseModule,
    TorchModel,
    metric_wrapper,
    optim_wrapper,
)
from secretflow.ml.nn import FLModel
from secretflow.security.aggregation import SecureAggregator
from secretflow.utils.simulation.datasets import load_mnist

from torch import nn, optim
from torch.nn import functional as F
from torchmetrics import Accuracy, Precision


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
class ConvNet(BaseModule):
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
        # 注意：CrossEntropyLoss 期望 logits（未 softmax）
        return x


# -----------------------------
# 2. Load dataset (Torch format)
# -----------------------------
(train_data, train_label), (test_data, test_label) = load_mnist(
    parts={alice: 0.4, bob: 0.6},
    normalized_x=True,
    categorical_y=True,
    is_torch=True,
)

# -----------------------------
# 3. Build TorchModel definition
# -----------------------------
loss_fn = nn.CrossEntropyLoss  # 传类/构造器即可（SecretFlow内部会实例化）
optim_fn = optim_wrapper(optim.Adam, lr=1e-2)

model_def = TorchModel(
    model_fn=ConvNet,
    loss_fn=loss_fn,
    optim_fn=optim_fn,
    metrics=[
        metric_wrapper(Accuracy, task="multiclass", num_classes=10, average="micro"),
        metric_wrapper(Precision, task="multiclass", num_classes=10, average="micro"),
    ],
)

# -----------------------------
# 4. Build FLModel (FedAvg)
# -----------------------------
device_list = [alice, bob]
server = charlie
aggregator = SecureAggregator(server, device_list)

fl_model = FLModel(
    server=server,
    device_list=device_list,
    model=model_def,
    aggregator=aggregator,
    strategy="fed_prox",   # <- FedProx
    backend="torch",
    strategy_params={      # <- FedProx 的关键参数
        "mu": 0.01,        # proximal 系数，可调：0.001 / 0.01 / 0.1 常见
    },
)


# -----------------------------
# 5. Train
# -----------------------------
history = fl_model.fit(
    train_data,
    train_label,
    validation_data=(test_data, test_label),
    epochs=20,
    batch_size=32,
    aggregate_freq=1,
)

print("Training history keys:", list(history.keys()) if isinstance(history, dict) else type(history))


# -----------------------------
# 6. Evaluate (after training)
# -----------------------------
# 训练结束后，使用测试集评估一次
eval_res = fl_model.evaluate(
    test_data,
    test_label,
    batch_size=32,
)

print("\nFinal evaluation on test set:")
print(eval_res)


# -----------------------------
# 7. (Optional) Predict
# -----------------------------
# 如果你还想拿预测结果做更细致的分析（如混淆矩阵），可以：
# y_pred = fl_model.predict(test_data, batch_size=32)
# print(type(y_pred))
