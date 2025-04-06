import torch
import torch.nn as nn
import torch.nn.functional as F
from linformer import Linformer

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
        self.act = nn.SiLU()
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
        self.t = torch.linspace(-self.kernel_size//2, self.kernel_size//2, steps=5, device=torch.device('cuda'))
    def forward(self, x):
        eps = 1e-6
        kernels = []
        for i in range(self.out_channels):
            f = self.center_frequencies[i]
            sigma = self.scales[i].clamp(min=1e-4)
            kernel = torch.exp(-self.t**2/(2*(sigma**2+eps))) * torch.cos(2*torch.pi*f*self.t)
            kernel = kernel/(kernel.sum()+eps)
            kernels.append(kernel)
        kernels = torch.stack(kernels, dim=0)
        kernels = kernels.unsqueeze(1).repeat(1, self.in_channels, 1)
        out = F.conv1d(x, kernels, padding=self.padding)
        return out

class ECGClassifier(nn.Module):
    def __init__(self, no_labels, seq_len):
        super().__init__()
        self.lift = nn.Conv1d(1, 32, kernel_size=1)
        self.wavelet = LearnableWaveletTransform(32, 32, kernel_size=3, padding=2)
        self.linformer = Linformer(dim=32, seq_len=seq_len, depth=1, heads=16, k=16)
        self.linformer_project = nn.Linear(32, 64)
        self.act = nn.SiLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.inception1 = InceptionModule1d(64, 128)
        self.res1 = nn.Conv1d(64, 128, kernel_size=1)
        self.inception2 = InceptionModule1d(128, 256)
        self.res2 = nn.Conv1d(128, 256, kernel_size=1)
        self.inception3 = InceptionModule1d(256, 512)
        self.res3 = nn.Conv1d(256, 512, kernel_size=1)
        self.inception4 = InceptionModule1d(512, 1024)
        self.res4 = nn.Conv1d(512, 1024, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.1)
        self.fc0 = nn.Linear(1024, 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.head = nn.Linear(128, no_labels)
        self.aux_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        if x.dim()==2:
            x = x.unsqueeze(1)
        x = self.lift(x)
        x = self.wavelet(x)
        x = x.transpose(1,2)
        x = self.linformer(x)
        x = self.linformer_project(x)
        x = x.transpose(1,2)
        x = self.act(x)
        x = self.maxpool(x)
        residual = self.res1(x)
        x = self.inception1(x)
        x = x+residual
        residual = self.res2(x)
        x = self.inception2(x)
        x = x+residual
        residual = self.res3(x)
        x = self.inception3(x)
        x = x+residual
        residual = self.res4(x)
        x = self.inception4(x)
        x = x+residual
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.dropout(x)
        x_c = self.fc0(x)
        x = self.act(x_c)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        main = self.head(x)
        aux = self.aux_head(x_c)
        return main, aux
