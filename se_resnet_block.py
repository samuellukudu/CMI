import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        se = F.adaptive_avg_pool1d(x, 1).squeeze(-1)      # -> (B, C)
        se = F.relu(self.fc1(se), inplace=True)           # -> (B, C//r)
        se = self.sigmoid(self.fc2(se)).unsqueeze(-1)     # -> (B, C, 1)
        return x * se

class ResNetSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, wd=1e-4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        # SE
        self.se = SEBlock(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          padding=0, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)              # (B, out, L)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)                       # (B, out, L)
        out = out + identity
        return self.relu(out)
