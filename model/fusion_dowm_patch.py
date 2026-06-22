


import torch
import torch.nn as nn


class Fusion_Patch(nn.Module):
    """
    Input:
        x: (B, 3, 128, 128)
           B = batch size
           3 = RGB / 3 channels
           128 = frequency axis
           128 = time axis

    Output:
        x: (B, 64, 128)
           64 = temporal tokens
           128 = embedding dimension
    """

    def __init__(self, embed_dim=128):
        super().__init__()

        self.cnn = nn.Sequential(
            # (B, 3, 128, 128) -> (B, 16, 128, 128)
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # 只压缩频率轴: F 128 -> 64, T 不变
            # (B, 16, 128, 128) -> (B, 16, 64, 128)
            nn.AvgPool2d(kernel_size=(2, 1)),

            # (B, 16, 64, 128) -> (B, 32, 64, 128)
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 只压缩频率轴: F 64 -> 32, T 不变
            # (B, 16, 64, 128) -> (B, 32, 32, 128)
            nn.AvgPool2d(kernel_size=(2, 1)),
        )

        # 只压缩时间轴: T 128 -> 64, F 不变
        # (B, 32, 32, 128) -> (B, 32, 32, 64)
        self.time_pool = nn.AvgPool2d(kernel_size=(1, 2))

        # 每个时间 token 的特征是 C * F = 64 * 32 = 2048
        self.proj = nn.Linear(32 * 32, embed_dim)

    def forward(self, x):
        """
        x: (B, 3, 128, 128)
        """

        # CNN feature extraction
        x = self.cnn(x)
        # x: (B, 32, 32, 128)

        # Downsample time axis
        x = self.time_pool(x)
        # x: (B, 32, 32, 64)

        B, C, F, T = x.shape
        # C = 32 F = 32, T = 64

        # Move time axis to token dimension
        x = x.permute(0, 3, 1, 2)
        # x: (B, T, C, F)
        # x: (B, 64, 32, 32)

        # Flatten C and F for each time token
        x = x.flatten(2)
        # x: (B, T, C * F)
        # x: (B, 64, 1024)

        # Project each token to embed_dim
        x = self.proj(x)
        # x: (B, 64, 128)

        return x


import torch
import torch.nn as nn


class Fusion_Patch_2(nn.Module):
    """
    Input:
        x: (B, 43, 128)

    Output:
        x: (B, 64, 128)
    """

    def __init__(
        self,
        freq_bins=43,
        embed_dim=128,
        num_tokens=64,
        dropout=0.1,
    ):
        super().__init__()

        self.freq_bins = freq_bins
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        self.conv1 = nn.Sequential(
            # (B, 1, 43, 128) -> (B, 16, 43, 128)
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            # (B, 16, 43, 128) -> (B, 32, 43, 128)
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 只沿时间轴下采样: T 128 -> 64
        # F=43 不变
        self.time_pool = nn.AvgPool2d(kernel_size=(1, 2))

        # 每个时间 token: C * F = 32 * 43 = 1376
        self.proj = nn.Sequential(
            nn.Linear(32 * freq_bins, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        x: (B, 43, 128)
        """

        # (B, 43, 128) -> (B, 1, 43, 128)
        x = x.unsqueeze(1)

        # (B, 1, 43, 128) -> (B, 16, 43, 128)
        x = self.conv1(x)

        # (B, 16, 43, 128) -> (B, 32, 43, 128)
        x = self.conv2(x)

        # (B, 32, 43, 128) -> (B, 32, 43, 64)
        x = self.time_pool(x)

        B, C, F, T = x.shape
        # C=32, F=43, T=64

        # (B, 32, 43, 64) -> (B, 64, 32, 43)
        x = x.permute(0, 3, 1, 2)

        # (B, 64, 32, 43) -> (B, 64, 1376)
        x = x.flatten(2)

        # (B, 64, 1376) -> (B, 64, 128)
        x = self.proj(x)

        return x

if __name__ == "__main__":
    x = torch.randn(8, 3, 128, 128)

    model = Fusion_Patch(embed_dim=128)

    out = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", out.shape)

    x = torch.randn(8, 43, 128)

    model = Fusion_Patch_2()

    out = model(x)

    print("input:", x.shape)
    print("output:", out.shape)























