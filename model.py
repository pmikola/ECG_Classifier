import torch
import torch.nn as nn
import torch.nn.functional as F

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

class ECGClassifier(nn.Module):
    def __init__(self, no_labels):
        super().__init__()
        self.lift = nn.Conv1d(1, 128, kernel_size=1)
        self.act = nn.SiLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.inception1 = InceptionModule1d(128, 256)
        self.res1 = nn.Conv1d(128, 256, kernel_size=1)
        self.inception2 = InceptionModule1d(256, 512)
        self.res2 = nn.Conv1d(256, 512, kernel_size=1)
        self.inception3 = InceptionModule1d(512, 1024)
        self.res3 = nn.Conv1d(512, 1024, kernel_size=1)
        self.inception4 = InceptionModule1d(1024, 2048)
        self.res4 = nn.Conv1d(1024, 2048, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.1)
        self.fc0 = nn.Linear(2048, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.head = nn.Linear(256, no_labels)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.lift(x)
        x = self.act(x)
        x = self.maxpool(x)
        residual = self.res1(x)
        x = self.inception1(x)
        x = x + residual
        residual = self.res2(x)
        x = self.inception2(x)
        x = x + residual
        residual = self.res3(x)
        x = self.inception3(x)
        x = x + residual
        residual = self.res4(x)
        x = self.inception4(x)
        x = x + residual
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc0(x)
        x = self.act(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.head(x)
        return x
