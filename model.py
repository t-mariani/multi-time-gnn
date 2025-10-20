import torch
from torch import nn
from torch.nn.functional import tanh, relu, sigmoid
from torch.nn import MSELoss


class GraphLearningLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.node_emb = nn.Embedding(config.N, config.embedding_dim)
        self.theta1 = nn.Parameter(torch.randn(config.embedding_dim, config.N))
        self.theta2 = nn.Parameter(torch.randn(config.embedding_dim, config.N))

    def forward(self):
        M1 = tanh(self.config.alpha * self.node_emb @ self.theta1)
        M2 = tanh(self.config.alpha * self.node_emb @ self.theta2)
        A = relu(tanh(self.config.alpha * (M1 @ M2.T - M1.T @ M2)))

        # Improve sparsity of A
        # for i in range(1): # A.shape[0]):
        #     torch.
        return A


class MixHopPropagationLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mlps = [nn.Linear(config.N, config.N, bias=False) for _ in range(config.k)]

    def forward(self, Hin, A):
        # Hin : BxTxN
        Dmoins1 = torch.diag(
            torch.Tensor([1 / torch.sum((A[i, :]))] for i in range(self.config.N))
        )  # will not work due to diag limitation s
        Atilde = Dmoins1 @ (A + torch.eye(self.config.N))

        Hprev = Hin
        for i in range(self.config.k):
            Hprev = self.config.beta * Hin + (1 - self.config.beta) * Atilde @ Hprev
            Hout += self.mlps[i](Hprev)
        return Hout


class GraphConvolutionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.left_mix_hop = MixHopPropagationLayer(config)
        self.right_mix_hop = MixHopPropagationLayer(config)

    def forward(self, x, A: torch.Tensor):
        return self.left_mix_hop(x, A) + self.left_mix_hop(x, A.T)


class DilatedInceptionLayer(nn.Module):
    def __init__(self, config, d):
        super().__init__()
        self.conv2 = nn.Conv1d(1, 1, kernel_size=(1, 2), stride=(1, d))
        self.conv3 = nn.Conv1d(1, 1, kernel_size=(1, 3), stride=(1, d))
        self.conv5 = nn.Conv1d(1, 1, kernel_size=(1, 5), stride=(1, d))
        self.conv7 = nn.Conv1d(1, 1, kernel_size=(1, 7), stride=(1, d))

    def forward(self):
        pass


class TimeConvolutionModule(nn.Module):
    def __init__(self, config, d):
        super().__init__()
        self.left_dilated_incep = DilatedInceptionLayer(config, d)
        self.right_dilated_incep = DilatedInceptionLayer(config, d)

    def forward(self, x):
        return tanh(self.left_dilated_incep(x)) + sigmoid(self.right_dilated_incep(x))


class NextStepModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.graph_learn = GraphLearningLayer(config)
        self.timeCM = [TimeConvolutionModule(config, i) for i in range(config.m)]
        self.graphCM = [GraphConvolutionModule(config) for _ in range(config.m)]

    def forward(self, x, y, v=None):
        graph = self.graph_learn(v)  # NxN
        x = self.first_conv(x)  # BxTxN
        for i in range(self.config.m):
            x1 = self.timeCM[i](x)  # BxTxNx4
            x2 = self.graphCM[i](x1, graph)  #
            x = x1 + x2  # ??
        next_point = self.output_module(x)
        loss = MSELoss(next_point, y)
        return next_point, loss
