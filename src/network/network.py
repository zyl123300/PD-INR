import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
import torch.sparse as sp
import numpy as np
import math

class DReLU(nn.Module):
    def __init__(self, gamma):
        super(DReLU, self).__init__()
        self.gamma = gamma
        if self.gamma <= 0:
            raise ValueError("The gamma value must be greater than zero.")

    def forward(self, x):
        return torch.where(x < self.gamma, self.gamma * torch.exp(x / self.gamma - 1), x)


class ModifiedSigmoid(nn.Module):
    def __init__(self, shift=0.1):
        super(ModifiedSigmoid, self).__init__()
        self.shift = shift

    def forward(self, x):
        return torch.sigmoid(x) - self.shift

class DensityNetwork(nn.Module):
    def __init__(self, encoder, sVoxel, n_samples=256, num_layers=8, hidden_dim=256, skips=[4], out_dim=1,
                 last_activation="sigmoid"):
        super().__init__()
        self.encoder = encoder

        # if type(bound) is list:
        #     self.bound = nn.Parameter(torch.Tensor(bound), requires_grad=False)
        # else:
        #     self.bound = bound
        self.bound = nn.Parameter(torch.Tensor(sVoxel / 2), requires_grad=False)

        self.density = DensityNet(encoder.output_dim, num_layers=num_layers, hidden_dim=hidden_dim,
                                  skips=skips, out_dim=out_dim, last_activation=last_activation)  # .to(torch.half)
        # self.density = GraphConvNet(encoder.output_dim, num_layers=num_layers, hidden_dim=hidden_dim,
        #                         out_dim=out_dim, last_activation=last_activation, n_samples = 256) # .to(torch.half)

    def forward(self, x):
        x = self.encoder(x, self.bound)

        return self.density(x)


class OneLayerMLP(nn.Module):
    def __init__(self, encoder, sVoxel, n_samples=256, num_layers=8, hidden_dim=256, skips=[4], out_dim=1,
                 last_activation="drelu"):
        super().__init__()
        self.bound = nn.Parameter(torch.Tensor(sVoxel / 2), requires_grad=False)
        self.encoder = encoder

        # 单层 MLP
        self.layer = nn.Linear(in_features=encoder.output_dim, out_features=out_dim, bias=False)
        with torch.no_grad():
            self.layer.weight[:, :8] = 0
            self.layer.weight[:, 8] = 1


        if last_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif last_activation == "relu":
            self.activation = nn.ReLU()
        elif last_activation == "lrelu":
            self.activation = nn.LeakyReLU()
        elif last_activation == "modified_sigmoid":
            self.activation = ModifiedSigmoid()
        elif last_activation == "drelu":
            self.activation = DReLU(0.5)
        elif last_activation == "none":
            self.activation = None
        else:
            raise NotImplementedError("Unknown last activation")
        self.density = nn.Sequential(self.layer, self.activation)

    def forward(self, x):
        x = self.encoder(x, self.bound)
        return self.density(x)

class GCNNetwork(nn.Module):
    def __init__(self, encoder, sVoxel, n_samples=256, num_layers=8, hidden_dim=256, skips=[4], out_dim=1,
                 last_activation="sigmoid"):
        super().__init__()
        self.encoder = encoder

        self.bound = nn.Parameter(torch.Tensor(sVoxel / 2), requires_grad=False)

        self.density = GraphConvNet(encoder.output_dim, num_layers=num_layers, hidden_dim=hidden_dim,
                                    out_dim=out_dim, last_activation=last_activation,
                                    n_samples=n_samples)  # .to(torch.half)

    def forward(self, x):
        x = self.encoder(x, self.bound)

        return self.density(x)


class DensityNet(nn.Module):
    def __init__(self, in_dim, num_layers=8, hidden_dim=256, skips=[4], out_dim=1, last_activation="sigmoid"):
        super().__init__()
        self.nunm_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.in_dim = in_dim

        # Linear layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.in_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) if i not in skips
                                                    else nn.Linear(hidden_dim + self.in_dim, hidden_dim) for i in
                                                    range(1, num_layers - 1, 1)])
        self.layers.append(nn.Linear(hidden_dim, out_dim))

        # Activations
        # self.activations = nn.ModuleList([nn.LeakyReLU() for i in range(0, num_layers-1, 1)])
        self.activations = nn.ModuleList([nn.ReLU() for i in range(0, num_layers - 1, 1)])
        if last_activation == "sigmoid":
            self.activations.append(nn.Sigmoid())
        elif last_activation == "relu":
            # self.activations.append(nn.LeakyReLU())
            self.activations.append(nn.ReLU())
        elif last_activation == "lrelu":
            self.activations.append(nn.LeakyReLU())
            # self.activations.append(nn.ReLU())
        elif last_activation == "modified_sigmoid":
            self.activations.append(ModifiedSigmoid())
        elif last_activation == "drelu":
            self.activations.append(DReLU(0.5))
        else:
            raise NotImplementedError("Unknown last activation")

    def forward(self, x):
        input_pts = x[..., :self.in_dim]

        for i in range(len(self.layers)):

            linear = self.layers[i]
            activation = self.activations[i]

            if i in self.skips:
                x = torch.cat([input_pts, x], -1)

            x = linear(x)
            x = activation(x)

        return x



class GraphConvolution(nn.Module):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907."""

    def __init__(self, in_features, out_features, adjmat, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adjmat = adjmat
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        stdv = 6. / math.sqrt(self.weight.size(0) + self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(self.adjmat, support)
        # output = spmm(self.adjmat, support)
        if self.bias is not None:
            output = output + self.bias
        return output

class GraphLinear(nn.Module):
    """
    Generalization of 1x1 convolutions on Graphs
    """

    def __init__(self, in_channels, out_channels):
        super(GraphLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = nn.Parameter(torch.FloatTensor(out_channels, in_channels))
        self.b = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        w_stdv = 1 / (self.in_channels * self.out_channels)
        self.W.data.uniform_(-w_stdv, w_stdv)
        self.b.data.uniform_(-w_stdv, w_stdv)

    def forward(self, x):
        return torch.matmul(self.W[None, :], x) + self.b[None, :, None]


class GraphResBlock(nn.Module):
    """
    Graph Residual Block similar to the Bottleneck Residual Block in ResNet
    """

    def __init__(self, in_channels, out_channels, A):
        super(GraphResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin1 = GraphLinear(in_channels, out_channels // 2)
        self.conv = GraphConvolution(out_channels // 2, out_channels // 2, A)
        self.lin2 = GraphLinear(out_channels // 2, out_channels)
        self.skip_conv = GraphLinear(in_channels, out_channels)
        self.pre_norm = nn.GroupNorm(in_channels // 8, in_channels)
        self.norm1 = nn.GroupNorm((out_channels // 2) // 8, (out_channels // 2))
        self.norm2 = nn.GroupNorm((out_channels // 2) // 8, (out_channels // 2))

    def forward(self, x):
        # print(x.shape)
        y = F.relu(self.pre_norm(x))
        y = self.lin1(y)

        y = F.relu(self.norm1(y))
        y = self.conv(y.transpose(1, 2)).transpose(1, 2)

        y = F.relu(self.norm2(y))
        y = self.lin2(y)
        if self.in_channels != self.out_channels:
            x = self.skip_conv(x)
        return x + y


class GraphBlock(nn.Module):
    """
    Graph Residual Block similar to the Bottleneck Residual Block in ResNet
    """

    def __init__(self, in_channels, out_channels, A):
        super(GraphBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.A = A
        # self.conv = GraphConvolution(in_channels, out_channels, A)
        # self.lin2 = GraphLinear(in_channels // 2, out_channels)
        # self.skip_conv = GraphLinear(in_channels, out_channels)
        # self.norm1 = nn.GroupNorm((in_channels // 2) // 8, in_channels)
        # self.norm2 = nn.GroupNorm((in_channels // 2) // 8, (in_channels // 2))
        self.linear = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(in_channels, out_channels)
        # self.BN = nn.BatchNorm1d(out_channels)
        # self.norm = nn.LayerNorm([out_channels])

    def forward(self, x):
        # print(x.shape)
        # y = F.relu(self.norm1(x))
        y = torch.matmul(self.A, self.linear(x))
        # y = self.norm(y)
        return F.relu(y) + self.linear2(x)
        # return F.relu(y)


class SparseMM(torch.autograd.Function):
    """Redefine sparse @ dense matrix multiplication to enable backpropagation.
    The builtin matrix multiplication operation does not support backpropagation in some cases.
    """

    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.req_grad = dense.requires_grad
        ctx.save_for_backward(sparse)
        return torch.matmul(sparse, dense)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        sparse, = ctx.saved_tensors
        if ctx.req_grad:
            grad_input = torch.matmul(sparse.t(), grad_output)
        return None, grad_input


def spmm(sparse, dense):
    return SparseMM.apply(sparse, dense)


class GraphConvNet(nn.Module):
    def __init__(self, in_dim, num_layers=8, hidden_dim=256, out_dim=1, last_activation="sigmoid",
                 n_samples=320):
        super().__init__()
        # self.num_layers = num_layers
        # self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.n_samples = n_samples
        # construct adjacency matrix for line
        W = np.diag(np.ones(n_samples - 1), -1) + np.diag(np.ones(n_samples - 1), 1)
        W = W / np.sum(W, axis=1).reshape((-1, 1))  # normalize
        W = W.astype(np.float32)
        W = nn.Parameter(torch.from_numpy(W), requires_grad=False)
        # .to_sparse()

        # num_layers_graph = 4
        # layers = [GraphBlock(in_dim, hidden_dim, W)]
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for i in range(num_layers):
            layers.append(GraphBlock(hidden_dim, hidden_dim, W))
        # layers.append(GraphBlock(hidden_dim, out_dim, W))
        layers.append(nn.Linear(hidden_dim, out_dim))
        # layers.append(nn.BatchNorm1d(out_dim))

        self.gc = nn.Sequential(*layers)

        # Activations
        if last_activation == "sigmoid":
            self.act_last = nn.Sigmoid()
        elif last_activation == "relu":
            self.act_last = nn.ReLU()
        elif last_activation == "lrelu":
            self.act_last = nn.LeakyReLU()
        else:
            raise NotImplementedError("Unknown last activation")

    def forward(self, x):
        y = x.view(-1, self.n_samples, self.in_dim)
        # y = torch.transpose(y, 1, 2)
        y = self.gc(y).view(-1, 1)
        return self.act_last(y)
