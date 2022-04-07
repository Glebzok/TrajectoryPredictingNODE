import torch.nn as nn
from model_parts import DoubleConv, Up, Down, OutConv, \
    PositionalEncoding
# from transformers import RoFormerConfig
# from transformers.models.roformer.modeling_roformer import RoFormerEncoder


class SingleLayerConvNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=output_dim,
                              kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, hidden_channels):
        super().__init__()
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
        for conv_layer, res_layer in zip(self.conv_layers[:-1], self.res_layers[:-1]):
            out = conv_layer(x)
            x = res_layer(x)
            x += out
            x = nn.Tanh()(x)

        out = self.conv_layers[-1](x)
        x = self.res_layers[-1](x)
        x += out

        return x


class UNetLikeConvNet(nn.Module):
    def __init__(self, input_dim, output_dim, min_channels, n_layers, act='ReLU'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.min_channels = min_channels
        self.n_layers = n_layers
        self.act = act

        self.inc = DoubleConv(input_dim, min_channels, act)

        self.down = nn.ModuleList([])
        n_channels = self.min_channels
        for _ in range(self.n_layers):
            self.down.append(Down(n_channels, 2 * n_channels, act))
            n_channels = 2 * n_channels

        self.up = nn.ModuleList([])
        for _ in range(self.n_layers):
            self.up.append(Up(n_channels, n_channels // 2, act))
            n_channels = n_channels // 2

        self.outc = OutConv(n_channels, self.output_dim)

    def forward(self, x):
        down_out = [self.inc(x)]
        for layer in self.down:
            down_out.append(layer(down_out[-1]))

        x = down_out[-1]

        for layer, down_x in zip(self.up, down_out[-2::-1]):
            x = layer(x, down_x)

        x = self.outc(x)

        return x


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
        x = self.inl(x.permute(0, 2, 1)).permute(1, 0, 2)
        x = self.pos_encoder(x).permute(1, 0, 2)
        x = self.transformer(x)
        x = self.outl(x).permute(0, 2, 1)

        return x

# class RoFormerNet(nn.Module):
#     def __init__(self, input_dim, output_dim, n_layers, nhead, dim_feedforward, dropout, activation):
#         super().__init__()
#
#         config = RoFormerConfig(vocab_size=1, embedding_size=input_dim, hidden_size=dim_feedforward,
#                                 num_hidden_layers=n_layers,
#                                 num_attention_heads=nhead, intermediate_size=dim_feedforward, hidden_act=activation,
#                                 hidden_dropout_prob=dropout,
#                                 attention_probs_dropout_prob=dropout, max_position_embeddings=200)
#
#         self.encoder = RoFormerEncoder(config)
#         self.inl = nn.Linear(input_dim, dim_feedforward)
#         self.outl = nn.Linear(dim_feedforward, output_dim)
#
#     def forward(self, x):
#         x = self.inl(x.permute(0, 2, 1))
#         x = self.encoder(x)[0]
#         x = self.outl(x).permute(0, 2, 1)
#         return x
