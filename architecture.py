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

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.act(out)


class BasicResNet(nn.Module):
    def __init__(
        self,
        blocks_per_stage,
        num_classes=10,
        activation="relu",
        dropout=0.0    # ← ajouter ici
    ):
        super().__init__()
        self.in_ch = 32
        self.act = get_activation(activation)

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            self.act
        )

        self.stage1 = self._make_stage(32,  blocks_per_stage[0], stride=1, dropout=dropout)
        self.stage2 = self._make_stage(64, blocks_per_stage[1], stride=2, dropout=dropout)
        self.stage3 = self._make_stage(128, blocks_per_stage[2], stride=2, dropout=dropout)
        self.stage4 = self._make_stage(256, blocks_per_stage[3], stride=2, dropout=dropout)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(256, num_classes)

    def _make_stage(self, out_ch, n_blocks, stride, dropout=0.0):
        layers = [
            BasicResBlock(
                self.in_ch,
                out_ch,
                stride=stride,
                dropout=dropout
            )
        ]
        self.in_ch = out_ch

        for _ in range(1, n_blocks):
            layers.append(
                BasicResBlock(
                    out_ch,
                    out_ch,
                    dropout=dropout
                )
            )

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


class BasicVGG(nn.Module):
    def __init__(self, config, activation="relu", dropout=0.0):
        super().__init__()
        
        layers = []
        in_ch = 3
        channels = [64, 128, 256, 512, 512]
        
        for i, num_blocks in enumerate(config):
            out_ch = channels[i]
            for _ in range(num_blocks):
                layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
                layers.append(nn.BatchNorm2d(out_ch))
                layers.append(get_activation(activation))
                if dropout > 0:
                    layers.append(nn.Dropout2d(dropout))
                in_ch = out_ch
            layers.append(nn.MaxPool2d(2, 2))
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            get_activation(activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)

    
configs_resnet = {
    "18":  [2, 2, 2, 2],
    "34":  [3, 4, 6, 3],
}

configs_densenet = {
    "121": [6, 12, 24, 16],
    "169": [6, 12, 32, 32],
}

configs_vgg = {
    "16": [2, 2, 3, 3, 3],
    "19": [2, 2, 4, 4, 4],
}

def build_model(nn_architecture, num_config, use_skip=True, activation="relu", dropout=0.0):
    if nn_architecture == "ResNet":
        return BasicResNet(
            blocks_per_stage=configs_resnet[num_config],
            num_classes=10,
            activation=activation,
            dropout=dropout
        )
    elif nn_architecture == "DenseNet":
        return BasicDenseNet(
            blocks=configs_densenet[num_config],
            activation=activation,
            dropout=dropout
        )
    elif nn_architecture == "VGG":
        return BasicVGG(
            config=configs_vgg[num_config],
            activation=activation,
            dropout=dropout
        )
