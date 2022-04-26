import torch
import torch.nn as nn
import torch.nn.functional as F

import copy


class InnerProduct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # bs, in_features
        x = x[:, :, None] @ x[:, None, :]
        x = x[torch.triu(torch.ones(x.shape)) == 1].view(x.shape[0], -1)
        # bs, in_features * (in_features+1) / 2
        return x


class Power(nn.Module):
    def __init__(self, in_features, n_powers):
        super().__init__()
        self.W = nn.Parameter(torch.rand((1, in_features, n_powers)) - 0.5)

    def forward(self, x):
        # bs, in_features
        x = F.relu(x) + 1e-9
        x = (x[:, :, None] ** self.W).view(x.shape[0], -1)
        # bs, in_features * n_powers
        return x


class Algebraic(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.inner_product = InnerProduct(in_features=in_features)
        inner_product_out_features = in_features * (in_features + 1) // 2

        self.linear = nn.Linear(in_features=in_features + inner_product_out_features,
                                out_features=in_features)

    def forward(self, x):
        x1 = self.inner_product(x)

        x = x + self.linear(torch.cat([x, x1], dim=1))
        return x


class AlgebraicNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, hidden_dim):
        super().__init__()
        self.l1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.alg = nn.ModuleList([Algebraic(in_features=hidden_dim) for _ in range(n_layers)])
        self.l2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):
        bs, n_points, latent_dim = x.shape
        # print(x.shape)
        x = x.reshape(-1, latent_dim)
        x = self.l1(x)

        for l in self.alg:
            x = l(x)
            x = torch.tanh(x)

        x = self.l2(x)
        x = x.view(bs, n_points, -1)
        return x


class PointwiseTransformerNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, nhead, dim_feedforward, dropout, activation):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   activation=activation)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.inl = nn.Linear(input_dim, input_dim * dim_feedforward)
        # self.outl1 = nn.Linear(latent_dim * dim_feedforward, latent_dim)
        # self.outl2 = nn.Linear(latent_dim, latent_dim)
        # self.outl3 = nn.Linear(latent_dim, signal_dim)

        self.outl = nn.Linear(input_dim * dim_feedforward, output_dim)

    def forward(self, x):
        # bs, n_vars, latent_dim
        bs, n_vars, latent_dim = x.shape
        x = x.reshape(-1, latent_dim)  # bs x n_vars, latent_dim
        x = self.inl(x)  # bs x n_vars, latent_dim * dim_ff
        x = x.view(bs * n_vars, latent_dim, -1)  # bs x n_vars, latent_dim, dim_ff
        x = x.permute(1, 0, 2)  # latent_dim, bs x n_vars, dim_ff
        x = self.transformer(x)  # latent_dim, bs x n_vars, dim_ff
        x = x.permute(1, 0, 2)  # bs x n_vars, latent_dim, dim_ff
        x = x.reshape(bs * n_vars, -1)  # bs x n_vars, latent_dim * dim_ff
        x = self.outl(x)  # bs x n_vars, signal_dim
        # x = self.outl1(x)  # bs x n_vars, latent_dim
        # x = F.relu(x) # bs x n_vars, latent_dim
        # x = self.outl2(x) # bs x n_vars, latent_dim
        # x = F.relu(x) # bs x n_vars, latent_dim
        # x = self.outl3(x) # bs x n_vars, signal_dim
        x = x.view(bs, n_vars, -1)  # bs , n_vars, signal_dim

        return x


class PointwisePermformerNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, nhead, dim_feedforward, dropout, activation):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   activation=activation)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(n_layers)])

        self.inl = nn.Linear(input_dim, input_dim * dim_feedforward)
        self.outl = nn.Linear(input_dim * dim_feedforward, output_dim)
        # self.outl1 = nn.Linear(latent_dim * dim_feedforward, latent_dim)
        # self.outl2 = nn.Linear(latent_dim, latent_dim)
        # self.outl3 = nn.Linear(latent_dim, signal_dim)

    def forward(self, x):
        # bs, n_vars, latent_dim
        bs, n_vars, latent_dim = x.shape
        bs_times_n_vars = bs * n_vars
        x = x.reshape(-1, latent_dim)  # bs x n_vars, latent_dim
        x = self.inl(x)  # bs x n_vars, latent_dim * dim_ff
        x = x.view(bs_times_n_vars, latent_dim, -1)  # bs x n_vars, latent_dim, dim_ff

        for layer in self.layers:
            x = x.reshape(bs_times_n_vars, -1)  # bs x n_vars, latent_dim x dim_ff
            x = x.view(bs_times_n_vars, latent_dim, -1,
                       latent_dim)  # bs x n_vars, latent_dim,  dim_ff / latent_dim, latent_dim
            x = x.permute(0, 3, 2, 1)  # bs x n_vars, latent_dim, dim_ff / latent_dim, latent_dim
            x = x.reshape(bs_times_n_vars, latent_dim, -1)  # bs x n_vars, latent_dim,  dim_ff

            x = x.permute(1, 0, 2)  # latent_dim, bs x n_vars, dim_ff
            x = layer(x)  # latent_dim, bs x n_vars, dim_ff
            x = x.permute(1, 0, 2)  # bs x n_vars, latent_dim, dim_ff

        x = x.reshape(bs * n_vars, -1)  # bs x n_vars, latent_dim * dim_ff
        x = self.outl(x)  # bs x n_vars, signal_dim
        # x = self.outl1(x)  # bs x n_vars, latent_dim
        # x = F.relu(x)  # bs x n_vars, latent_dim
        # x = self.outl2(x)  # bs x n_vars, latent_dim
        # x = F.relu(x)  # bs x n_vars, latent_dim
        # x = self.outl3(x)  # bs x n_vars, signal_dim
        x = x.view(bs, n_vars, -1)  # bs , n_vars, signal_dim

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
