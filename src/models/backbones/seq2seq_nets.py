import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.backbones.model_parts import DoubleConv, Up, Down, OutConv, PositionalEncoding


class SingleLayerConvNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=output_dim,
                              kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # x: (bs, input_dim, T)
        x = self.conv(x)  # (bs, output_dim, T)
        return x


class ConvNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, hidden_channels, activation):
        super().__init__()
        self.activation = activation

        if n_layers == 1:
            self.conv_layers = nn.ModuleList([nn.Conv1d(input_dim, output_dim, kernel_size=3, padding=1)])
            self.res_layers = nn.ModuleList([nn.Conv1d(input_dim, output_dim, kernel_size=1)])

        elif n_layers == 2:
            self.conv_layers = nn.ModuleList([nn.Conv1d(input_dim, hidden_channels, kernel_size=3, padding=1),
                                              nn.Conv1d(hidden_channels, output_dim, kernel_size=3, padding=1)])
            self.res_layers = nn.ModuleList([nn.Conv1d(input_dim, hidden_channels, kernel_size=1),
                                             nn.Conv1d(hidden_channels, output_dim, kernel_size=1)])

        else:
            self.conv_layers = nn.ModuleList([nn.Conv1d(input_dim, hidden_channels, kernel_size=3, padding=1)] \
                                             + [nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
                                                for _ in range(n_layers - 2)] \
                                             + [nn.Conv1d(hidden_channels, output_dim, kernel_size=3, padding=1)])
            self.res_layers = nn.ModuleList([nn.Conv1d(input_dim, hidden_channels, kernel_size=1)] \
                                            + [nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1) for _ in
                                               range(n_layers - 2)] \
                                            + [nn.Conv1d(hidden_channels, output_dim, kernel_size=1)])

    def forward(self, x):
        # x: (bs, input_dim, T)
        for conv_layer, res_layer in zip(self.conv_layers[:-1], self.res_layers[:-1]):
            out = conv_layer(x)
            x = res_layer(x)
            x += out
            if self.activation == 'tanh':
                x = torch.tanh(x)
            else:
                x = torch.relu(x)

        out = self.conv_layers[-1](x)
        x = self.res_layers[-1](x)
        x += out

        return x  # x: (bs, output_dim, T)


class UNetLikeConvNet(nn.Module):
    def __init__(self, input_dim, output_dim, init_channels, n_layers, act, always_decrease_n_ch):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init_channels = init_channels
        self.n_layers = n_layers
        self.act = act

        self.inc = DoubleConv(input_dim, init_channels, act)
        self.down = nn.ModuleList([])

        n_channels = self.init_channels
        self.down_ch = [n_channels]

        for _ in range(self.n_layers):
            if always_decrease_n_ch:
                self.down.append(Down(n_channels, n_channels // 2, act))
                n_channels = n_channels // 2
            else:
                self.down.append(Down(n_channels, 2 * n_channels, act))
                n_channels = 2 * n_channels
            self.down_ch.append(n_channels)

        self.up = nn.ModuleList([])
        for _, down_ch in zip(range(self.n_layers), self.down_ch[-2::-1]):
            self.up.append(Up(n_channels, n_channels // 2 + down_ch, n_channels // 2, act))
            n_channels = n_channels // 2

        self.outc = OutConv(n_channels, self.output_dim)

    def forward(self, x):
        # x: (bs, input_dim, T)
        down_out = [self.inc(x)]
        for layer in self.down:
            down_out.append(layer(down_out[-1]))

        x = down_out[-1]

        for layer, down_x in zip(self.up, down_out[-2::-1]):
            x = layer(x, down_x)

        x = self.outc(x)

        return x  # x: (bs, output_dim, T)


class Seq2SeqTransformerNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, nhead, dim_feedforward, dropout, activation):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   activation=activation)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pos_encoder = PositionalEncoding(d_model=dim_feedforward, dropout=dropout, max_len=1000)

        self.inl = nn.Linear(input_dim, dim_feedforward)
        self.outl = nn.Linear(dim_feedforward, output_dim)

    def forward(self, x):
        # x: (bs, input_dim, T)
        x = self.inl(x.permute(0, 2, 1)).permute(1, 0, 2)
        x = self.pos_encoder(x).permute(1, 0, 2)
        x = self.transformer(x)
        x = self.outl(x).permute(0, 2, 1)

        return x  # x: (bs, output_dim, T)


class ShrinkingResNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, activation='tanh'):
        super().__init__()
        self.activation = activation

        shrinking_factor = 0 if n_layers == 1 else (output_dim / input_dim) ** (1 / n_layers)
        self.conv_layers = nn.ModuleList([nn.Conv1d(int(input_dim * shrinking_factor ** i),
                                                    int(input_dim * shrinking_factor ** (i + 1)), kernel_size=3,
                                                    padding=1)
                                          for i in range(n_layers - 1)]
                                         +
                                         [nn.Conv1d(int(input_dim * shrinking_factor ** (n_layers - 1)), output_dim,
                                                    kernel_size=3, padding=1)])
        self.res_layers = nn.ModuleList([nn.Conv1d(int(input_dim * shrinking_factor ** i),
                                                   int(input_dim * shrinking_factor ** (i + 1)), kernel_size=1)
                                         for i in range(n_layers - 1)]
                                        +
                                        [nn.Conv1d(int(input_dim * shrinking_factor ** (n_layers - 1)), output_dim,
                                                   kernel_size=1)])

    def forward(self, x):
        # x: (bs, input_dim, T)
        for conv_layer, res_layer in zip(self.conv_layers[:-1], self.res_layers[:-1]):
            out = conv_layer(x)
            if self.activation == 'tanh':
                out = torch.tanh(out)
            else:
                out = torch.relu(out)

            x = res_layer(x)
            x += out

        out = self.conv_layers[-1](x)
        x = self.res_layers[-1](x)
        x += out

        return x  # x: (bs, output_dim, T)
