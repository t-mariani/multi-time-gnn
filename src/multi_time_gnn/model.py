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

        log.debug(f"Embed emitter : {embed_emitter.shape}")
        log.debug(f"Embed receiver : {embed_receiver.shape}")
        M1 = tanh(self.config.alpha * embed_emitter @ self.theta1)  # N * embedding_dim
        M2 = tanh(self.config.alpha * embed_receiver @ self.theta2)  # N * embedding_dim
        log.debug(f"M1 : {M1.shape}")
        log.debug(f"M2 : {M2.shape}")
        A = relu(tanh(self.config.alpha * (M1 @ M2.T - M2 @ M1.T)))  # N * N

        log.debug(f"Graph : {A.shape}")

        # Improve sparsity of A
        if self.config.k_sparsity:
            indices_to_keep = torch.topk(A, self.config.k_sparsity, dim=1).indices
            mask = torch.zeros_like(A, dtype=torch.bool)
            mask.scatter_(1, indices_to_keep, True)
            A = A * mask
        return A


class MixHopPropagationLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mlps = nn.ModuleList(
            [nn.Linear(config.N, config.N, bias=False) for _ in range(config.k)]
        )

    def forward(self, Hin, A):
        """
        B: batch dimension
        C: b of channels  ## what is a channel, not clear 
        N: nb captors
        T: length of the time series
        """
        # Hin : BxCxNxT
        graph = A  # NxN
        # graph = torch.detach(A)  # We need to compute the gradients during the optimisation
        # I think that we need to not detach A from the graph otherwise we will never learn the graph and just have a random one


        ## In order to speed up the compute of Dmoins1, we can parallelise it with torch:
        # Dmoins1 = torch.diag(
        #     torch.tensor(
        #         [1 / (1 + torch.sum((graph[i, :]))) for i in range(self.config.N)]
        #     )
        # )
        row_sums = torch.sum(graph, dim=1)
        Dmoins1 = torch.diag(1 / (1 + row_sums))  # NxN
        log.debug(Dmoins1.shape)
        Atilde = Dmoins1 @ (A + torch.eye(self.config.N))  # NxN

        Hprev = Hin
        Hout = 0
        for i in range(self.config.k):
            graph_times_hprev = torch.einsum("n m, bcnt -> b c m t", Atilde, Hprev)  # BxCxNxT
            log.debug(f"graph_times_hp shape :{graph_times_hprev.shape}")
            Hprev = self.config.beta * Hin + (1 - self.config.beta) * graph_times_hprev  # BxCxNxT
            log.debug(f"hprev shape :{Hprev.shape}")
            Hout += rearrange(
                self.mlps[i](rearrange(Hprev, "b c n t -> b c t n")),
                "b c t n -> b c n t",
            )
        return Hout # BxCxNxT


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
        conv_size = [2, 3, 5, 7]  # change from 3 to 2, right ?
        padding = ["same", "same", "same", "same"]
        self.convs = []
        self.convs_1D = []
        for k, size in enumerate(conv_size):
            # no stride because we just want to have the same size at the end 
            # self.convs.append(
            #     nn.Conv2d(
            #         config.residual_channels,
            #         out_channel,
            #         kernel_size=(1, size),
            #         stride=(1, d),
            #         padding=(0, size // 2),
            #     )
            # )
            self.convs.append(
                nn.Conv1d(
                    config.residual_channels*config.N,
                    out_channel*config.N,
                    kernel_size=size,
                    dilation=d,
                    padding=padding[k],
                )
            )
        self.convs = nn.ModuleList(self.convs)

    def forward(self, x):
        log.debug(f"shape x dilated : {x.shape}")
        res = []
        # min_t_size = x.shape[2]
        for conv in self.convs:
            x_reshape = x.reshape(x.shape[0], -1, x.shape[-1]) # BxC*NxT 
            res_conv = conv(x_reshape) # BxC*NxT
            res_conv = res_conv.reshape(x.shape[0], x.shape[1]//4, -1, x.shape[3]) # BxCxNxT
            res.append(res_conv)

            ## Not sure what it is doing after this line, it's giving a bug because the shape has changed
            # min_t_size = min(min_t_size, res_conv.shape[3])  # retrieve T
            # log.debug(f"min_size_t: {min_t_size}, res_conv_shape: {res_conv.shape}")
            ##
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
        self.end_conv_2 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=(1, config.timepoints_input)
        )

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
            [TimeConvolutionModule(config, d=2**i) for i in range(config.m)]
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
            x2 = self.graphCM[i](x1, graph)  # BxCxTxN
            log.debug(f"x shape graphCM: {x2.shape}")
            x = x2 + residual
            x = self.layer_norm[i](x)
            residual = x  # we need to update the residual ?
            skip += x1  # are we sure we need to add them and not to concatenate them?

        next_point = self.output_module(skip + x)
        log.debug(f"Next point : {next_point.shape}")
        if y is None:
            return next_point, None

        log.debug(f"Compared y : {y.shape}")
        loss = self.loss(next_point, y)
        return next_point, loss
