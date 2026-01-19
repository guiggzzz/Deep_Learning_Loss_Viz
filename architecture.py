import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "silu":
        return nn.SiLU()
    else:
        raise ValueError(f"Unknown activation {name}")

class BasicResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, activation="relu", dropout=0.0):
        super().__init__()
        self.act = get_activation(activation)
        
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.act(out)


class BasicResNet(nn.Module):
    def __init__(self, blocks_per_stage, activation="relu", dropout=0.0):
        super().__init__()
        self.in_ch = 64
        self.act = get_activation(activation)

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            self.act
        )

        self.stage1 = self._make_stage(64,  blocks_per_stage[0], 1, activation, dropout=dropout)
        self.stage2 = self._make_stage(128, blocks_per_stage[1], 2, activation, dropout=dropout)
        self.stage3 = self._make_stage(256, blocks_per_stage[2], 2, activation, dropout=dropout)
        self.stage4 = self._make_stage(512, blocks_per_stage[3], 2, activation, dropout=dropout)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(512, 10)  # ← FIXÉ À 10 CLASSES

    def _make_stage(self, out_ch, n_blocks, stride, activation, dropout=0.0):
        layers = [BasicResBlock(self.in_ch, out_ch, stride, activation, dropout=dropout)]
        self.in_ch = out_ch
        for _ in range(1, n_blocks):
            layers.append(BasicResBlock(out_ch, out_ch, activation=activation, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
    
    
class DenseBlock(nn.Module):
    def __init__(self, in_ch, growth_rate, activation="relu", dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            get_activation(activation),
            nn.Conv2d(in_ch, growth_rate, 3, padding=1, bias=False),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        out = self.block(x)
        return torch.cat([x, out], dim=1)
   

class Transition(nn.Module):
    def __init__(self, in_ch, out_ch, activation="relu"):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            get_activation(activation),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        return self.block(x)

class BasicDenseNet(nn.Module):
    def __init__(self, blocks, growth_rate=32, reduction=0.5, activation="relu", dropout=0.0):
        super().__init__()

        ch = 2 * growth_rate
        self.stem = nn.Conv2d(3, ch, 3, padding=1, bias=False)

        self.features = nn.ModuleList()
        for n_blocks in blocks:
            for _ in range(n_blocks):
                self.features.append(DenseBlock(ch, growth_rate, activation, dropout))
                ch += growth_rate
            out_ch = int(ch * reduction)
            self.features.append(Transition(ch, out_ch, activation))
            ch = out_ch

        self.bn = nn.BatchNorm2d(ch)
        self.act = get_activation(activation)
        self.fc = nn.Linear(ch, 10)  # ← FIXÉ À 10 CLASSES

    def forward(self, x):
        x = self.stem(x)
        for layer in self.features:
            x = layer(x)
        x = self.act(self.bn(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)
    



configs_dense_net = {
    "121": [6, 12, 24, 16],
    "169": [6, 12, 32, 32],
}
configs_resnet = {
    "18": [2, 2, 2, 2],
    "34": [3, 4, 6, 3],
}

def build_model(resnet, num_config, activation="relu", dropout=0.0):
    if resnet:
        return BasicResNet(configs_resnet[num_config], activation, dropout=dropout)
    else:
        return BasicDenseNet(configs_dense_net[num_config], activation=activation, dropout=dropout)