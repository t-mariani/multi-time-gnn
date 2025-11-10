from einops import rearrange, repeat
import torch
from torch import nn
from torch.nn.functional import tanh, relu, sigmoid
from torch.nn import MSELoss

from multi_time_gnn.utils import get_logger

log = get_logger()


class GraphLearningLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        k = 1  # could be another hyperparameter
        # If node_emb is already known with k features, change k accordingly and change node_emb_emitter and receiver
        self.node_emb_emitter = nn.Embedding(config.N, k)
        self.node_emb_receiver = nn.Embedding(config.N, k)
        self.theta1 = nn.Parameter(torch.randn(k, config.embedding_dim))
        self.theta2 = nn.Parameter(torch.randn(k, config.embedding_dim))

    def forward(self, v=None):
        """
        v: list of nodes (index) to consider, if None all nodes are considered
        """
        if v is None:
            embed_emitter = self.node_emb_emitter(
                torch.arange(0, self.config.N, dtype=torch.int)
            )
            embed_receiver = self.node_emb_receiver(
                torch.arange(0, self.config.N, dtype=torch.int)
            )
        else:
            embed_emitter = self.node_emb_emitter[v]
            embed_receiver = self.node_emb_receiver[v]

        M1 = tanh(self.config.alpha * embed_emitter @ self.theta1)
        M2 = tanh(self.config.alpha * embed_receiver @ self.theta2)
        A = relu(tanh(self.config.alpha * (M1 @ M2.T - M1.T @ M2)))

        log.debug(f"Graph : {A.shape}")
        log.debug(f"Graph val : {A}")

        # Improve sparsity of A
        # for i in range(1): # A.shape[0]):
        #     torch.
        return A


class MixHopPropagationLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mlps = nn.ModuleList(
            [nn.Linear(config.N, config.N, bias=False) for _ in range(config.k)]
        )

    def forward(self, Hin, A):
        # Hin : BxCxNxT
        graph = torch.detach(A)
        Dmoins1 = torch.diag(
            torch.tensor(
                [1 / (1 + torch.sum((graph[i, :]))) for i in range(self.config.N)]
            )
        )
        log.debug(Dmoins1.shape)
        Atilde = Dmoins1 @ (A + torch.eye(self.config.N))

        Hprev = Hin
        Hout = 0
        for i in range(self.config.k):
            graph_times_hprev = torch.einsum("n m, bcnt -> b c m t", Atilde, Hprev)
            log.debug(f"graph_times_hp shape :{graph_times_hprev.shape}")
            Hprev = self.config.beta * Hin + (1 - self.config.beta) * graph_times_hprev
            log.debug(f"hprev shape :{Hprev.shape}")
            Hout += rearrange(
                self.mlps[i](rearrange(Hprev, "b c n t -> b c t n")),
                "b c t n -> b c n t",
            )

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
        out_channel = config.residual_channels // 4
        conv_size = [3, 3, 5, 7]
        self.convs = []
        for size in conv_size:
            self.convs.append(
                nn.Conv2d(
                    config.residual_channels,
                    out_channel,
                    kernel_size=(1, size),
                    stride=(1, d),
                    padding=(0, size // 2),
                )
            )
        self.convs = nn.ModuleList(self.convs)

    def forward(self, x):
        log.debug(f"shape x dilated : {x.shape}")
        res = []
        min_t_size = x.shape[2]
        for conv in self.convs:
            res_conv = conv(x)
            res.append(res_conv)
            min_t_size = min(min_t_size, res_conv.shape[3])  # retrieve T
            log.debug(f"min_size_t: {min_t_size}, res_conv_shape: {res_conv.shape}")
        # Truncate to match size along T
        # res = [xt[:, :, :, :min_t_size] for xt in res]
        return torch.cat(res, dim=1)  # along C


class TimeConvolutionModule(nn.Module):
    def __init__(self, config, d):
        super().__init__()
        self.left_dilated_incep = DilatedInceptionLayer(config, d)
        self.right_dilated_incep = DilatedInceptionLayer(config, d)

    def forward(self, x):
        return tanh(self.left_dilated_incep(x)) + sigmoid(self.right_dilated_incep(x))


class OutputModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.end_conv_1 = nn.Conv2d(
            in_channels=config.residual_channels, out_channels=1, kernel_size=1
        )
        self.end_conv_2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)

    def forward(self, x):
        log.debug(f"output x shape : {x.shape}")
        x = relu(x)
        x = relu(self.end_conv_1(x))
        log.debug(f"output x shape endconv1 : {x.shape}")
        x = self.end_conv_2(x)
        return x


class NextStepModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # C = config.residual_channels
        self.first_conv = nn.Conv2d(1, self.config.residual_channels, 1)
        self.first_skip = nn.Conv2d(1, self.config.residual_channels, 1)
        self.graph_learn = GraphLearningLayer(config)
        self.timeCM = nn.ModuleList(
            [TimeConvolutionModule(config, 1) for i in range(config.m)]
        )
        self.graphCM = nn.ModuleList(
            [GraphConvolutionModule(config) for _ in range(config.m)]
        )
        self.layer_norm = nn.ModuleList(
            [
                nn.LayerNorm(
                    (config.residual_channels, config.N, config.timepoints_input)
                )
                for _ in range(config.m)
            ]
        )
        self.output_module = OutputModule(config)

        self.loss = MSELoss()

    def forward(self, input, y=None, v=None):
        graph = self.graph_learn(v)  # NxN
        log.debug(f"graph shape: {graph.shape}")
        x = self.first_conv(input)  # BxCxTxN
        log.debug(f"x shape first conv: {x.shape}")
        skip = self.first_skip(input)  # BxCxTxN
        log.debug(f"x shape first skip: {x.shape}")
        residual = x
        for i in range(self.config.m):
            x1 = self.timeCM[i](x)  # BxCxT'xN  # may reduce in T due to convolution
            log.debug(f"x shape timeCM: {x1.shape}")
            skip += x1
            x2 = self.graphCM[i](x1, graph)  # BxCxTxN
            log.debug(f"x shape graphCM: {x2.shape}")
            x = x2 + residual
            x = self.layer_norm[i](x)

        next_point = self.output_module(skip + x)
        log.debug(f"Next point : {next_point}")
        if y is None:
            return next_point, None

        log.debug(f"Compared y : {y}")
        loss = self.loss(next_point, y)
        return next_point, loss
