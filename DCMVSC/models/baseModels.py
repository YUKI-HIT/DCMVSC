"""
模型的所有的基本模块，提高复用性
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module



class Encoder(nn.Module):
    """输出的隐变量"""

    def __init__(self, encoder_dim, activation='relu', batchnorm=True):
        super(Encoder, self).__init__()
        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim:
                if self._batchnorm and i < self._dim - 1:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        self._encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        latent = self._encoder(x)
        return latent


class Decoder(nn.Module):
    """将隐变量z解码重构成x-hat"""

    def __init__(self, encoder_dim, activation='relu', batchnorm=True):
        super(Decoder, self).__init__()
        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm and i < self._dim - 1:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._decoder = nn.Sequential(*decoder_layers)

    def forward(self, latent):
        x_hat = self._decoder(latent)
        return x_hat


class GNNLayer(Module):
    def __init__(self, in_features_dim, out_features_dim, activation='relu', use_bias=True):
        super(GNNLayer, self).__init__()
        self.in_features = in_features_dim
        self.out_features = out_features_dim
        self.use_bias = use_bias
        self.weight = Parameter(torch.FloatTensor(in_features_dim, out_features_dim))
        if self.use_bias:
            self.bias = Parameter(torch.FloatTensor(out_features_dim))
        self.init_parameters()

        self._bn1d = nn.BatchNorm1d(out_features_dim)
        if activation == 'sigmoid':
            self._activation = nn.Sigmoid()
        elif activation == 'leakyrelu':
            self._activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self._activation = nn.Tanh()
        elif activation == 'relu':
            self._activation = nn.ReLU()
        else:
            raise ValueError('Unknown activation type %s' % self._activation)

    def init_parameters(self):
        """初始化权重"""
        torch.nn.init.xavier_uniform_(self.weight)
        if self.use_bias:
            torch.nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, features, adj, active=True, batchnorm=True):
        support = torch.mm(features, self.weight)  # 矩阵相乘
        output = torch.spmm(adj, support)  # 稀疏矩阵相乘
        if self.use_bias:
            output += self.bias
        if batchnorm:
            output = self._bn1d(output)
        if active:
            output = self._activation(output)
        return output


class GATLayer(torch.nn.Module):
    head_dim = 1

    def __init__(self, in_features_dim, out_features_dim, num_of_heads=3, activation='relu', use_bias=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features_dim
        self.out_features = out_features_dim
        self.use_bias = use_bias
        self.num_of_heads = num_of_heads

        self.weight = Parameter(torch.FloatTensor(num_of_heads, in_features_dim, out_features_dim))
        if self.use_bias:
            self.bias = Parameter(torch.FloatTensor(out_features_dim))
        self.scoring_fn_target = nn.Parameter(torch.Tensor(num_of_heads, out_features_dim, 1))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(num_of_heads, out_features_dim, 1))

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the log-softmax along the last dimension
        self.bn1d = nn.BatchNorm1d(out_features_dim)
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError('Unknown activation type %s' % self.activation)
        self.attention_weights = None  # for later visualization purposes, I cache the weights here

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        if self.use_bias:
            torch.nn.init.zeros_(self.bias)

    def forward(self, features, adj, active=True, batchnorm=True):

        # Step 1: Linear Projection + regularization
        adj_dense = adj.to_dense()
        connectivity_mask = torch.where(adj_dense == 0, torch.tensor([-torch.inf], device=adj_dense.device), adj_dense)
        features_proj = torch.matmul(features.unsqueeze(0), self.weight)

        # Step 2: Edge attention calculation
        scores_source = torch.bmm(features_proj, self.scoring_fn_source)
        scores_target = torch.bmm(features_proj, self.scoring_fn_target)
        # shape = (NH, N, 1) + (NH, 1, N) -> (NH, N, N) with the magic of automatic broadcast
        all_scores = self.leakyReLU(scores_source + scores_target.transpose(1, 2))
        all_attention_coefficients = self.softmax(all_scores + connectivity_mask)

        # Step 3: Neighborhood aggregation
        # shape = (NH, N, N) * (NH, N, FOUT) -> (NH, N, FOUT)
        out_features = torch.bmm(all_attention_coefficients, features_proj)
        # shape = (N, NH, FOUT)
        out_features = out_features.transpose(0, 1)

        # Step 4: averaging and bias
        # shape = (N, NH, FOUT) -> (N, FOUT)
        out_features = out_features.mean(dim=self.head_dim)
        if self.use_bias:
            out_features += self.bias
        if batchnorm:
            out_features = self.bn1d(out_features)
        if active:
            out_features = self.activation(out_features)
        return out_features


class GraphEncoder(nn.Module):
    def __init__(self, encoder_dim, activation='relu', batchnorm=True, GAT=False):
        super(GraphEncoder, self).__init__()
        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm
        self._GAT = GAT

        encoder_layers = []
        for i in range(self._dim):
            if self._GAT:
                encoder_layers.append(
                    GATLayer(encoder_dim[i], encoder_dim[i + 1], num_of_heads=3, activation=self._activation))
            else:
                encoder_layers.append(GNNLayer(encoder_dim[i], encoder_dim[i + 1], activation=self._activation))
        self._encoder = nn.Sequential(*encoder_layers)

    def forward(self, x, adj, skip_connetion=False):
        z = x
        if skip_connetion:  # 加入跳跃连接
            z = self._encoder[0](z, adj)
            for layer in self._encoder[1:-1]:
                z = layer(z, adj) + z
        else:
            for layer in self._encoder[0:-1]:
                z = layer(z, adj)

        z = self._encoder[-1](z, adj, False, True)
        return z


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, activation=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.activation = activation

    def forward(self, z):
        adj = torch.mm(z, z.t())
        adj = self.activation(adj)
        return adj


class ClusterProject(nn.Module):
    def __init__(self, latent_dim, n_clusters):
        super(ClusterProject, self).__init__()
        self._latent_dim = latent_dim
        self._n_clusters = n_clusters
        self.cluster_projector = nn.Sequential(
            nn.Linear(self._latent_dim, self._latent_dim),
            nn.BatchNorm1d(self._latent_dim),
            nn.ReLU(),
            nn.Linear(self._latent_dim, self._latent_dim),
            nn.BatchNorm1d(self._latent_dim),
            nn.ReLU(),
        )
        self.cluster = nn.Sequential(
            nn.Linear(self._latent_dim, self._n_clusters),
            nn.BatchNorm1d(self._n_clusters),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        z = self.cluster_projector(x)
        y = self.cluster(z)
        return y, z

