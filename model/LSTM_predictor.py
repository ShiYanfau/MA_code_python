import torch
import torch.nn as nn
import torch.nn.functional as F
from model.LSTM_basic import LSTM


class LSTM_Predictor(nn.Module):
    """
    LSTM-based dynamics predictor.

    输入:
        Z: (B, T, D)

    输出:
        Z_pred: (B, T-1, D)
        用 z_1 ... z_{T-1} 预测 z_2 ... z_T
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 256,
            num_layers: int = 2,
            dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = LSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.pred_head = nn.Linear(hidden_dim, input_dim)

    def forward(self, Z):
        """
        Z: (B, T, D)

        return:
            Z_pred: (B, T-1, D)
        """

        # 用前 T-1 个 latent states 预测后 T-1 个 latent states
        Z_input = Z[:, :-1, :]  # (B, T-1, D)

        lstm_out, _ = self.lstm(Z_input)  # (B, T-1, hidden_dim)

        Z_pred = self.pred_head(lstm_out)  # (B, T-1, D)

        return Z_pred


def dynamics_loss(Z_pred, Z):
    """
    Z_pred: (B, T-1, D)
    Z:      (B, T, D)

    target 是 Z[:, 1:, :]
    """

    Z_target = Z[:, 1:, :]  # (B, T-1, D)

    loss = F.mse_loss(Z_pred, Z_target)

    return loss


if __name__ == "__main__":
    B = 8
    T = 64
    D = 128
    Z = torch.randn(B, T, D)

    predictor = LSTM_Predictor(input_dim=D)

    Z_pred = predictor(Z)

    print("Z_pred shape:", Z_pred.shape)  # 应该是 (B, T-1, D)

    loss = dynamics_loss(Z_pred, Z)
    print("Dynamics loss:", loss.item())











































