import torch
import torch.nn as nn
from model.mamba_basic import SimpleMambaBlock


class MambaEncoder(nn.Module):
    """
    输入:
        x: (B, T, D)

    输出:
        hidden_states: (B, T, D)
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(d_model),
                "mamba": SimpleMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                ),
                "dropout": nn.Dropout(dropout),
            })
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """

        if x.dim() != 3:
            raise ValueError(f"Expected x shape (B, T, D), but got {x.shape}")

        for layer in self.layers:
            residual = x

            x_norm = layer["norm"](x)          # (B, T, D)
            x_mamba = layer["mamba"](x_norm)   # (B, T, D)
            x_mamba = layer["dropout"](x_mamba)

            x = residual + x_mamba             # (B, T, D)

        hidden_states = self.final_norm(x)      # (B, T, D)

        return hidden_states


if __name__ == "__main__":
    B = 8
    T = 64
    D = 128

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    x = torch.randn(B, T, D).to(device)

    model = MambaEncoder(
        d_model=D,
        num_layers=4,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
    ).to(device)

    hidden_states = model(x)

    print("input shape: ", x.shape)
    print("hidden shape:", hidden_states.shape)



































