import torch
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops


class GCLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCLayer, self).__init__(aggr='max') #  "Max" aggregation.
        self.mlp = nn.Sequential(nn.Linear(2 * in_channels, out_channels),
                                   nn.ReLU(),
                                   nn.Linear(out_channels, out_channels))

    def forward(self, x, adj):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        edge_index,_ = dense_to_sparse(adj)
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)