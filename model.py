import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from linformer import Linformer


class LeakySiLU(nn.Module):
    def __init__(self, negative_slope=0.1):
        super(LeakySiLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return x * (self.negative_slope + (1 - self.negative_slope) * torch.sigmoid(x))


class InceptionModule1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1)
        )
        self.act = nn.LeakyReLU(0.3)#LeakySiLU(0.5)
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return self.act(torch.cat([out1, out2, out3, out4], dim=1))


class LearnableWaveletTransform(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.center_frequencies = nn.Parameter(torch.rand(out_channels, device=torch.device('cuda')))
        self.scales = nn.Parameter(torch.rand(out_channels, device=torch.device('cuda')))
        self.t = torch.linspace(-1, 1, steps=kernel_size, device=torch.device('cuda'))

    def forward(self, x):
        eps = 1e-6
        kernels = []
        for i in range(self.out_channels):
            f = self.center_frequencies[i]
            sigma = self.scales[i]#.clamp(min=1e-4)
            gauss = torch.exp(-self.t ** 2 / (2 * (sigma ** 2 + eps)))
            correction = torch.exp(-((2 * torch.pi * f * sigma) ** 2) / 2)
            kernel = gauss * (torch.cos(2 * torch.pi * f * self.t) - correction)
            kernel = kernel / (kernel.norm(p=1) + eps)
            kernels.append(kernel)
        kernels = torch.stack(kernels, dim=0)
        kernels = kernels.unsqueeze(1).repeat(1, self.in_channels, 1)
        out = F.conv1d(x, kernels, padding=self.padding)
        return out

class ECGClassifier(nn.Module):
    def __init__(self, no_labels, seq_len):
        super().__init__()
        self.act = nn.LeakyReLU(0.3)#LeakySiLU(0.5)
        self.lift_0 = nn.Conv1d(1, 12, kernel_size=1)
        self.lift_linformer = Linformer(dim=12, seq_len=seq_len, depth=1, heads=12, k=12)
        self.lift_linformer_project = nn.Linear(12, 12)
        self.lift_3 = nn.Linear(12, 32)
        self.wavelet_main = LearnableWaveletTransform(32, 32, kernel_size=10, padding=1)
        self.linformer_main = Linformer(dim=32, seq_len=seq_len, depth=1, heads=16, k=16)
        self.linformer_project_main = nn.Linear(32, 32)
        self.pool_0 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.inception1 = InceptionModule1d(32, 64)
        self.res1 = nn.Conv1d(32, 64, kernel_size=1)
        self.inception2 = InceptionModule1d(64, 128)
        self.res2 = nn.Conv1d(64, 128, kernel_size=1)
        self.inception3 = InceptionModule1d(128, 256)
        self.res3 = nn.Conv1d(128, 256, kernel_size=1)
        self.inception4 = InceptionModule1d(256, 512)
        self.res4 = nn.Conv1d(256, 512, kernel_size=1)
        self.pool_1 = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.1)
        self.fc0 = nn.Linear(512, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.head = nn.Linear(64, no_labels)
        self.wavelet_aux = LearnableWaveletTransform(32, 32, kernel_size=10, padding=1)
        self.linformer_aux = Linformer(dim=32, seq_len=seq_len, depth=1, heads=16, k=16)
        self.linformer_project_aux = nn.Linear(32, 32)
        self.aux_inception = InceptionModule1d(32, 32)
        self.aux_res = nn.Conv1d(32, 32, kernel_size=1)
        self.aux_avg = nn.AdaptiveAvgPool1d(1)
        self.aux_fc0 = nn.Linear(32, 16)
        self.aux_fc1 = nn.Linear(16, 8)
        self.aux_head = nn.Linear(8, 2)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # main path
        x = self.lift_0(x)
        r0 = x.transpose(1, 2)
        x = self.lift_linformer(r0)
        x = self.act(self.lift_linformer_project(x))
        x_out = x + r0
        x = self.act(self.lift_3(x))
        x = x.transpose(1, 2)
        x_main = self.wavelet_main(x)
        x_main = x_main.transpose(1,2)
        x_main = self.linformer_main(x_main)
        x_main = self.linformer_project_main(x_main)
        x_main = x_main.transpose(1,2)
        x_main = self.act(x_main)
        x_main = self.pool_0(x_main)
        r1 = self.res1(x_main)
        x_main = self.inception1(x_main)
        x_main = x_main + r1
        r2 = self.res2(x_main)
        x_main = self.inception2(x_main)
        x_main = x_main + r2
        r3 = self.res3(x_main)
        x_main = self.inception3(x_main)
        x_main = x_main + r3
        r4 = self.res4(x_main)
        x_main = self.inception4(x_main)
        x_main = x_main + r4
        x_main = self.pool_1(x_main)
        x_main = x_main.view(x_main.size(0), -1)
        x_main = self.dropout(x_main)
        x_m = self.fc0(x_main)
        x_m = self.act(x_m)
        x_m = self.fc1(x_m)
        x_m = self.act(x_m)
        x_m = self.fc2(x_m)
        x_m = self.act(x_m)
        main_out = self.head(x_m)
        # aux path
        x_aux = self.wavelet_aux(x)
        x_aux = x_aux.transpose(1,2)
        x_aux = self.linformer_aux(x_aux)
        x_aux = self.linformer_project_aux(x_aux)
        x_aux = x_aux.transpose(1,2)
        x_aux = self.act(x_aux)
        r_aux = self.aux_res(x_aux)
        x_aux = self.aux_inception(x_aux)
        x_aux = x_aux + r_aux
        x_aux = self.aux_avg(x_aux)
        x_aux = x_aux.view(x_aux.size(0), -1)
        x_aux = self.act(self.aux_fc0(x_aux))
        x_aux = self.act(self.aux_fc1(x_aux))
        aux_out = self.aux_head(x_aux)
        return main_out, aux_out,x_out
