import torch
import torch.nn as nn
import torch.nn.functional as F

class ReLUConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.op(x)

class DilConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        )

    def forward(self, x):
        return self.op(x)

class SepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        )

    def forward(self, x):
        return self.op(x)

class OperationLayer(nn.Module):
    def __init__(self, channels, stride):
        super().__init__()
        self.ops = nn.ModuleList([OPS[op](channels, stride) for op in OPERATIONS])
        self.out = nn.Sequential(
            nn.Conv2d(channels * len(OPERATIONS), channels, 1, padding=0, bias=False),
            nn.ReLU()
        )

    def forward(self, x, weights):
        states = [op(x) * weight.view(-1, 1, 1, 1) for op, weight in zip(self.ops, weights.transpose(1, 0))]
        return self.out(torch.cat(states, dim=1))

class GroupOperationLayers(nn.Module):
    def __init__(self, steps, channels):
        super().__init__()
        self.preprocess = ReLUConv(channels, channels, 1, 1, 0)
        self.steps = steps
        self.ops = nn.ModuleList([OperationLayer(channels, 1) for _ in range(steps)])
        self.relu = nn.ReLU()

    def forward(self, x, weights):
        x = self.preprocess(x)
        for step in range(self.steps):
            residual = x
            x = self.ops[step](x, weights[:, step, :])
            x = self.relu(x + residual)
        return x

class AttentionLayer(nn.Module):
    def __init__(self, channels, steps, num_ops):
        super().__init__()
        self.steps = steps
        self.num_ops = num_ops
        self.output_size = steps * num_ops
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, self.output_size * 2),
            nn.ReLU(),
            nn.Linear(self.output_size * 2, self.output_size)
        )

    def forward(self, x):
        pooled = self.avg_pool(x).view(x.size(0), -1)
        attention_weights = self.fc(pooled).view(-1, self.steps, self.num_ops)
        return F.softmax(attention_weights, dim=-1)

OPERATIONS = [
    'sep_conv_1x1', 'sep_conv_3x3', 'sep_conv_5x5', 'sep_conv_7x7',
    'dil_conv_3x3', 'dil_conv_5x5', 'dil_conv_7x7', 'avg_pool_3x3'
]

OPS = {
    'avg_pool_3x3': lambda c, s: nn.AvgPool2d(3, stride=s, padding=1, count_include_pad=False),
    'sep_conv_1x1': lambda c, s: SepConv(c, c, 1, s, 0),
    'sep_conv_3x3': lambda c, s: SepConv(c, c, 3, s, 1),
    'sep_conv_5x5': lambda c, s: SepConv(c, c, 5, s, 2),
    'sep_conv_7x7': lambda c, s: SepConv(c, c, 7, s, 3),
    'dil_conv_3x3': lambda c, s: DilConv(c, c, 3, s, 2, 2),
    'dil_conv_5x5': lambda c, s: DilConv(c, c, 5, s, 4, 2),
    'dil_conv_7x7': lambda c, s: DilConv(c, c, 7, s, 6, 2)
}

class SubNet(nn.Module):
    def __init__(self, channels, num_layers=1, steps=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(AttentionLayer(channels, steps, len(OPERATIONS)))
            self.layers.append(GroupOperationLayers(steps, channels))

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, AttentionLayer):
                weights = layer(x)
            else:
                x = layer(x, weights)
        return x

class FusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fgnet = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
        self.feat_encoder = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, hsi_feats, msi_feats):
        residual = hsi_feats
        fused = self.fgnet(torch.cat([hsi_feats, msi_feats], dim=1))
        features = self.feat_encoder(torch.cat([hsi_feats, msi_feats], dim=1))
        feat1, feat2 = self.conv1(features), self.conv2(features)
        return feat1 + feat2 + fused + residual

class HyperFusion(nn.Module):
    def __init__(self, hsi_bands, msi_bands, channels=64, upscale=4, depth=3):
        super().__init__()
        self.hsi_head = nn.Conv2d(hsi_bands, channels, 3, 1, 1)
        self.msi_head = nn.Conv2d(msi_bands, channels, 3, 1, 1)
        self.tail = nn.Sequential(
            nn.Conv2d(channels * depth, channels, 3, 1, 1),
            SubNet(channels),
            nn.Conv2d(channels, hsi_bands, 3, 1, 1)
        )
        self.hsi_encoders = nn.ModuleList([RIFU(channels) for _ in range(depth)])
        self.msi_encoders = nn.ModuleList([FRDAB(channels) for _ in range(depth)])
        self.fusion_blocks = nn.ModuleList([CBM(channels) for _ in range(depth)])

    def forward(self, hsi, msi):
        hsi = F.interpolate(hsi, scale_factor=self.upscale, mode='bicubic', align_corners=False)
        hsi_feats, msi_feats, fused_feats = [], [], []

        hsi_x = self.hsi_head(hsi)
        msi_x = self.msi_head(msi)

        for hsi_encoder, msi_encoder in zip(self.hsi_encoders, self.msi_encoders):
            hsi_x = hsi_encoder(hsi_x)
            msi_x = msi_encoder(msi_x)
            fused_feats.append(self.fusion_blocks[len(hsi_feats)](hsi_x, msi_x))
            hsi_feats.append(hsi_x)
            msi_feats.append(msi_x)

        return self.tail(torch.cat(fused_feats, dim=1)) + hsi
