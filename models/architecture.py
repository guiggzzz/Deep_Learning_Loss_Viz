# models/cnn.py
import torch.nn as nn

# ---------- Activation factory ----------
def get_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "silu":
        return nn.SiLU()
    else:
        raise ValueError(f"Unknown activation {name}")

# ---------- Basic Conv Block ----------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=True, activation="relu"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn   = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.act  = get_activation(activation)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# ---------- Residual Block ----------
class ResidualBlock(nn.Module):
    def __init__(self, ch, use_bn=True, activation="relu"):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(ch, ch, use_bn, activation),
            ConvBlock(ch, ch, use_bn, activation),
        )

    def forward(self, x):
        return x + self.block(x)

# ---------- Experimental CNN ----------
class CustomCNN(nn.Module):
    def __init__(
        self,
        depth=3,
        width=64,
        use_skip=True,
        use_bn=True,
        activation="relu",
        dropout=0.0
    ):
        super().__init__()

        self.use_skip = use_skip
        self.dropout = nn.Dropout(dropout)

        self.stem = ConvBlock(3, width, use_bn, activation)

        blocks = []
        for _ in range(depth):
            if use_skip:
                blocks.append(ResidualBlock(width, use_bn, activation))
            else:
                blocks.append(ConvBlock(width, width, use_bn, activation))

        self.blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(width, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

