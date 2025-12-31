from einops import rearrange, repeat
import numpy as np
import torch
from torch import nn
from torch.nn.functional import tanh, relu, sigmoid
from torch.nn import MSELoss
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

from multi_time_gnn.utils import get_logger

log = get_logger()


class GraphLearningLayer(nn.Module):
    def __init__(self, config, known_captor_features=False):
        super().__init__()
        self.config = config
        # If node_emb is already known with k features, change k accordingly and change node_emb_emitter and receiver
        self.known_captor_features = known_captor_features
        if known_captor_features:
            # Replace node_emb with your own data
            k = 1
            self.node_emb_emitter = nn.Embedding(config.N, k)
            self.node_emb_receiver = nn.Embedding(config.N, k)
            self.theta1 = nn.Parameter(torch.randn(k, config.embedding_dim, device=config.device))
            self.theta2 = nn.Parameter(torch.randn(k, config.embedding_dim, device=config.device))
        else:
            self.node_emb_emitter = nn.Embedding(config.N, config.embedding_dim)
            self.node_emb_receiver = nn.Embedding(config.N, config.embedding_dim)

    def forward(self, v=None):
        """
        v: list of nodes (index) to consider, if None all nodes are considered
        """
        if v is None:
            embed_emitter = self.node_emb_emitter(
                torch.arange(0, self.config.N, dtype=torch.int).to(self.config.device)
            )
            embed_receiver = self.node_emb_receiver(
                torch.arange(0, self.config.N, dtype=torch.int).to(self.config.device)
            )
        else:
            embed_emitter = self.node_emb_emitter[v]
            embed_receiver = self.node_emb_receiver[v]

        log.debug(f"Embed emitter : {embed_emitter.shape}")
        log.debug(f"Embed receiver : {embed_receiver.shape}")
        if self.known_captor_features:
            # We need to adjust the features with theta1 and 2
            M1 = tanh(
                self.config.alpha * embed_emitter @ self.theta1
            )  # N * embedding_dim
            M2 = tanh(
                self.config.alpha * embed_receiver @ self.theta2
            )  # N * embedding_dim
        else:
            M1 = tanh(self.config.alpha * embed_emitter)  # N * embedding_dim
            M2 = tanh(self.config.alpha * embed_receiver)  # N * embedding_dim

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
            [
                nn.Linear(
                    config.residual_channels, config.residual_channels, bias=False
                )
                for _ in range(config.k)
            ]
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
        Atilde = Dmoins1 @ (A + torch.eye(self.config.N, device=self.config.device))  # NxN

        Hprev = Hin
        Hout = 0
        for i in range(self.config.k):
            graph_times_hprev = torch.einsum(
                "n m, bcnt -> b c m t", Atilde, Hprev
            )  # BxCxNxT
            # Here n and m are the same but in order to use einsum we have to give different names
            log.debug(f"graph_times_hp shape :{graph_times_hprev.shape}")
            Hprev = (
                self.config.beta * Hin + (1 - self.config.beta) * graph_times_hprev
            )  # BxCxNxT
            log.debug(f"hprev shape :{Hprev.shape}")
            Hout += rearrange(
                self.mlps[i](rearrange(Hprev, "b c n t -> b n t c")),
                "b n t c -> b c n t",
            )
        return Hout  # BxCxNxT


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
        conv_size = [2, 3, 5, 7]
        self.convs = []
        for size in conv_size:
            self.convs.append(
                nn.Conv2d(
                    config.residual_channels,
                    out_channel,
                    kernel_size=(1, size),
                    dilation=(1, d),
                    # padding=(0, size // 2),
                )
            )
        self.convs = nn.ModuleList(self.convs)

    def forward(self, x):
        log.debug(f"shape x dilated : {x.shape}")
        res = []
        for conv in self.convs:
            res_conv = conv(x)  # BxC//4xNxT'
            res.append(res_conv)
        min_t_size = res[-1].shape[3]
        log.debug(f"min_size_t: {min_t_size}, res_conv_shape: {res_conv.shape}")
        # Truncate to match size along T
        res = [xt[:, :, :, -min_t_size:] for xt in res]
        return torch.cat(res, dim=1)  # along C # BxCxNxT_4'


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
            in_channels=config.skip_layer_channels,
            out_channels=config.intermediate_output_channels,
            kernel_size=1,
        )
        # Change out_channels for multistep prediction
        self.end_conv_2 = nn.Conv2d(
            in_channels=config.intermediate_output_channels,
            out_channels=1,
            kernel_size=1,
        )

    def forward(self, x):
        log.debug(f"output x shape : {x.shape}")
        x = relu(x)
        x = relu(self.end_conv_1(x))
        log.debug(f"output x shape endconv1 : {x.shape}")
        x = self.end_conv_2(x)
        return x


def get_model(config):
    if config.model_kind == "MTGNN":
        return NextStepModelMTGNN(config)
    elif config.model_kind == "statistical":
        return NextStepModelAR()

class NextStepModelAR():
    def __init__(self):
        pass
 
    def forward(self, input, lags, y=None):
        result = np.empty((input.shape[0]))
        for chanel_number, time_serie in enumerate(input):
            model = AutoReg(time_serie, lags=lags[chanel_number], trend='c')
            model_fit = model.fit()
            pred = model_fit.predict(start=len(time_serie), end=len(time_serie), dynamic=False)
            result.append(pred)




class NextStepModelMTGNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # C = config.residual_channels
        self.first_conv = nn.Conv2d(1, self.config.residual_channels, 1)
        self.first_skip = nn.Conv2d(
            1, self.config.skip_layer_channels, (1, config.timepoints_input)
        )
        self.graph_learn = GraphLearningLayer(config)
        self.timeCM = nn.ModuleList(
            [TimeConvolutionModule(config, d=2**i) for i in range(config.m)]
        )
        self.graphCM = nn.ModuleList(
            [GraphConvolutionModule(config) for _ in range(config.m)]
        )
        self.dropouts = nn.ModuleList(
            [nn.Dropout(self.config.p_dropout) for _ in range(config.m)]
        )
        # Here Li is the number of timepoints after the timeCM[i]
        # In fact, with dilation and convolution of 7
        list_Li = [
            config.timepoints_input - sum(6 * 2**j for j in range(i + 1))
            for i in range(config.m)
        ]
        log.debug(f"List Li (number of timepoints after each block) : {list_Li}")
        self.layer_norm = nn.ModuleList(
            [nn.LayerNorm((config.residual_channels, config.N, Li)) for Li in list_Li]
        )
        self.skip_layer = nn.ModuleList(
            [
                nn.Conv2d(
                    self.config.residual_channels,
                    self.config.skip_layer_channels,
                    (1, Li),
                )
                for Li in list_Li
            ]
        )
        self.last_skip = nn.Conv2d(
            self.config.residual_channels,
            self.config.skip_layer_channels,
            (1, list_Li[-1]),
        )
        self.output_module = OutputModule(config)

        self.loss = MSELoss()

    def forward(self, input, y=None, v=None):
        graph = self.graph_learn(v)  # NxN
        log.debug(f"graph shape: {graph.shape}")
        x = self.first_conv(input)  # BxCxNxT
        log.debug(f"x shape first conv: {x.shape}")
        skip = self.first_skip(input)  # BxCxNx1
        log.debug(f"x shape first skip: {skip.shape}")
        for i in range(self.config.m):
            residual = x
            x1 = self.timeCM[i](x)  # BxCxNxT'
            log.debug(f"x shape timeCM, block {i}: {x1.shape}")
            skip += self.skip_layer[i](x1)  # BxCxNx1
            x1 = self.dropouts[i](x1)

            x2 = self.graphCM[i](x1, graph)  # BxCxNxT'
            log.debug(f"x shape graphCM, block {i}: {x2.shape}")
            if self.config.enable_layer_norm:
                x2 = self.layer_norm[i](x2)  # BxCxNxT'

            x = (
                x2 + residual[:, :, :, -x2.size(3) :]
            )  # truncate residual to match T and T'

        x = self.last_skip(x)  # B,C,N,1
        log.debug(f"Skip {skip.shape}")
        log.debug(f"Last Skip {x.shape}")
        next_point = self.output_module(skip + x)  # B,1,N,1
        log.debug(f"Next point : {next_point.shape}")
        if y is None:
            return next_point, None

        log.debug(f"Compared y : {y.shape}")
        loss = self.loss(next_point.squeeze(), y.squeeze())
        return next_point, loss
