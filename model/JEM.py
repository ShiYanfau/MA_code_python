import torch
import torch.nn as nn
import torch.nn.functional as F
from model.LSTM_predictor import LSTM_Predictor

class JEMWithLSTMDynamics(nn.Module):
    """
    输入:
        x: (B, T, D)

    输出:
        logits: (B, num_classes)
        Z:      (B, T, latent_dim)
        Z_pred: (B, T-1, latent_dim)
    """

    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int,
        num_classes: int,
        dynamics_hidden_dim: int = 256,
        dynamics_num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = encoder

        self.classifier = nn.Linear(latent_dim, num_classes)

        self.dynamics_predictor = LSTM_Predictor(
            input_dim=latent_dim,
            hidden_dim=dynamics_hidden_dim,
            num_layers=dynamics_num_layers,
            dropout=dropout,
        )

    def forward(self, x):
        """
        x: (B, T, D)

        encoder 输出必须是:
            Z: (B, T, latent_dim)
        """

        Z = self.encoder(x)  # (B, T, latent_dim)

        # 分类：对时间维做平均池化
        h = Z.mean(dim=1)  # (B, latent_dim)

        logits = self.classifier(h)  # (B, num_classes)

        Z_pred = self.dynamics_predictor(Z)  # (B, T-1, latent_dim)

        return logits, Z, Z_pred

    def energy_score(self, x):
        """
        Energy score:
            E(x) = -logsumexp(logits)
        """
        logits, _, _ = self.forward(x)

        energy = -torch.logsumexp(logits, dim=1)

        return energy

    def dynamics_error(self, x):
        """
        每个样本自己的 dynamics prediction error.

        返回:
            error_per_sample: (B,)
        """
        _, Z, Z_pred = self.forward(x)

        Z_target = Z[:, 1:, :]  # (B, T-1, D)

        error = (Z_pred - Z_target) ** 2  # (B, T-1, D)

        error_per_sample = error.mean(dim=(1, 2))  # (B,)

        return error_per_sample




if __name__ == "__main__":
    B = 8
    T = 64
    D = 128
    num_classes = 6

    x = torch.randn(B, T, D)
    y = torch.randint(0, num_classes, (B,))

    class DummyEncoder(nn.Module):
        def __init__(self, input_dim, latent_dim):
            super().__init__()
            self.proj = nn.Linear(input_dim, latent_dim)

        def forward(self, x):
            # x: (B, T, D)
            Z = self.proj(x)  # (B, T, latent_dim)
            return Z

    encoder = DummyEncoder(
        input_dim=D,
        latent_dim=D,
    )

    model = JEMWithLSTMDynamics(
        encoder=encoder,
        latent_dim=D,
        num_classes=num_classes,
        dynamics_hidden_dim=256,
        dynamics_num_layers=2,
        dropout=0.1,
    )

    logits, Z, Z_pred = model(x)

    print("x shape:", x.shape)
    print("logits shape:", logits.shape)
    print("Z shape:", Z.shape)
    print("Z_pred shape:", Z_pred.shape)

    loss_cls = F.cross_entropy(logits, y)
    loss_dyn = F.mse_loss(Z_pred, Z[:, 1:, :])
    loss = loss_cls + 0.1 * loss_dyn

    print("loss_cls:", loss_cls.item())
    print("loss_dyn:", loss_dyn.item())
    print("loss:", loss.item())

    energy = model.energy_score(x)
    dyn_error = model.dynamics_error(x)

    print("energy shape:", energy.shape)
    print("energy:", energy)

    print("dyn_error shape:", dyn_error.shape)
    print("dyn_error:", dyn_error)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Test passed: forward, loss, backward, optimizer step all work.")



























