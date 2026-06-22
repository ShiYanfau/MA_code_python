import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMambaBlock(nn.Module):
    """
    纯 PyTorch 简化版 Mamba-like block

    输入:
        x: (B, T, D)

    输出:
        y: (B, T, D)

    特点:
        - 不依赖 mamba_ssm
        - 不依赖 causal-conv1d
        - 不需要额外 CUDA 编译
        - 可以在 cuda / cpu / mps 上跑
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.d_conv = d_conv

        # 输入投影，分成 x 分支和 gate 分支
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # depthwise causal conv
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # 产生 B, C, dt
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner)

        # dt projection
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)

        # SSM 参数 A 和 D
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state) * 0.02)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        B, T, D = x.shape

        # (B, T, 2 * d_inner)
        xz = self.in_proj(x)

        # x_branch: (B, T, d_inner)
        # z_branch: (B, T, d_inner)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        # depthwise causal conv
        # (B, T, d_inner) -> (B, d_inner, T)
        x_conv = x_branch.transpose(1, 2)

        # padding 后长度会多出来，只取前 T 个，保证 causal
        x_conv = self.conv1d(x_conv)[:, :, :T]

        # (B, d_inner, T) -> (B, T, d_inner)
        x_conv = x_conv.transpose(1, 2)

        x_conv = F.silu(x_conv)

        # 投影得到 B, C, dt_input
        # params: (B, T, 2 * d_state + d_inner)
        params = self.x_proj(x_conv)

        B_param, C_param, dt_input = torch.split(
            params,
            [self.d_state, self.d_state, self.d_inner],
            dim=-1,
        )

        # dt: (B, T, d_inner)
        dt = F.softplus(self.dt_proj(dt_input))

        # A: (d_inner, d_state), 保证为负，增强稳定性
        A = -torch.exp(self.A_log)

        # selective scan
        y = self.selective_scan(
            x=x_conv,
            dt=dt,
            A=A,
            B_param=B_param,
            C_param=C_param,
            D=self.D,
        )

        # gate
        y = y * F.silu(z_branch)

        # 输出投影回 d_model
        y = self.out_proj(y)

        return y

    def selective_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B_param: torch.Tensor,
        C_param: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """
        简化版 selective scan

        x:       (B, T, d_inner)
        dt:      (B, T, d_inner)
        A:       (d_inner, d_state)
        B_param: (B, T, d_state)
        C_param: (B, T, d_state)
        D:       (d_inner,)

        return:
            y: (B, T, d_inner)
        """

        B_size, T, d_inner = x.shape
        d_state = A.shape[-1]

        # hidden state: (B, d_inner, d_state)
        h = torch.zeros(
            B_size,
            d_inner,
            d_state,
            device=x.device,
            dtype=x.dtype,
        )

        outputs = []

        for t in range(T):
            # 当前时间步
            x_t = x[:, t, :]          # (B, d_inner)
            dt_t = dt[:, t, :]        # (B, d_inner)
            B_t = B_param[:, t, :]    # (B, d_state)
            C_t = C_param[:, t, :]    # (B, d_state)

            # 离散化 A
            # dA: (B, d_inner, d_state)
            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))

            # 输入项
            # dB_x: (B, d_inner, d_state)
            dB_x = dt_t.unsqueeze(-1) * B_t.unsqueeze(1) * x_t.unsqueeze(-1)

            # 更新状态
            h = dA * h + dB_x

            # 输出
            # y_t: (B, d_inner)
            y_t = torch.sum(h * C_t.unsqueeze(1), dim=-1)

            # skip connection D * x
            y_t = y_t + D.unsqueeze(0) * x_t

            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, T, d_inner)

        return y


class MambaEncoder(nn.Module):
    """
    不依赖 mamba_ssm 的 Mamba-like Encoder

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

            x_norm = layer["norm"](x)
            x_mamba = layer["mamba"](x_norm)
            x_mamba = layer["dropout"](x_mamba)

            x = residual + x_mamba

        hidden_states = self.final_norm(x)

        return hidden_states


if __name__ == "__main__":
    B = 8
    T = 64
    D = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    print("hidden min:", hidden_states.min().item())
    print("hidden max:", hidden_states.max().item())
    print("has NaN:", torch.isnan(hidden_states).any().item())
    print("has Inf:", torch.isinf(hidden_states).any().item())


