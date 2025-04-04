import math
import torch
from torch import nn

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat))

    def forward(self, x):
        batch_size, in_channels, n = x.shape
        pos_enc = self.positional_encoding(n, x.device)
        x = x + pos_enc
        x_ft = torch.fft.fft(x, dim=-1)
        out_ft = torch.zeros(batch_size, self.out_channels, x_ft.shape[-1], dtype=torch.cfloat, device=x.device)
        out_ft[..., :self.modes] = torch.einsum("bim, ioj -> bom", x_ft[..., :self.modes], self.weights)
        x_out = torch.fft.ifft(out_ft, n, dim=-1).real
        return x_out

    @staticmethod
    def positional_encoding(n, device):
        position = torch.arange(n//2, dtype=torch.float32, device=device)
        div_term = torch.exp(torch.arange(0, n, 2, device=device) * (-math.log(10000.0) / n))
        pe = torch.zeros(1, 1, n, device=device)
        pe[..., 0::2] = torch.sin(position * div_term)
        pe[..., 1::2] = torch.cos(position * div_term)
        return pe

class EcgFNOClassifier(nn.Module):
    def __init__(self, number_of_points, modes, width, no_labels):
        super().__init__()
        self.modes = modes
        self.width = width
        self.no_labels = no_labels
        self.no_points = number_of_points
        self.lift = nn.Linear(1, width)
        self.dropout = nn.Dropout(0.2)
        self.sconv1 = SpectralConv1d(width, width, modes)
        self.w1 = nn.Conv1d(width, width, kernel_size=1)
        self.conv1 = nn.Conv1d(width, width // 2, kernel_size=3, padding=1)
        self.sconv2 = SpectralConv1d(width // 2, width // 2, modes)
        self.w2 = nn.Conv1d(width // 2, width // 2, kernel_size=1)
        self.sconv3 = SpectralConv1d(width // 2, width // 2, modes)
        self.w3 = nn.Conv1d(width // 2, width // 2, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.downlift = nn.Linear((width // 2) * 4, width // 4)
        self.head = nn.Linear(width // 4, no_labels)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.lift(x)
        x = self.act(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        r = x
        x1 = self.sconv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.act(x + r)
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.dropout(x)
        r = x
        x1 = self.sconv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.act(x + r)
        x = self.dropout(x)
        r = x
        x1 = self.sconv3(x)
        x2 = self.w3(x)
        x = self.act(x1 + x2 + r)
        x = self.dropout(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.downlift(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.head(x)
        return x

    @staticmethod
    def act(x):
        return torch.relu(x)
