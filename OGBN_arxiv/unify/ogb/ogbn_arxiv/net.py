import torch
import torch.nn as nn
from graph_masker import DegreeAwareEdgeScoring

import dgl.function as fn
from layer_gcn import ApplyNodeFunc,GCNLayer
import pdb
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch_sparse
from torch_sparse import SparseTensor
import torch.nn.functional as F

from typing import Union, Tuple, Optional, Callable
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType, 
                                    OptTensor, PairTensor)
from math import log
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag, matmul
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
import pdb, pickle, os


msg_mask = fn.src_mul_edge(src='h',edge='mask', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

class net_gcn(nn.Module):
    def __init__(self, embedding_dim, device, spar_wei, spar_adj, use_res, use_bn,coef=None, mode="prune"):
        super().__init__()

        self.mode = mode
        self.n_layers = len(embedding_dim) - 1
        
        self.spar_wei = spar_wei
        self.spar_adj = spar_adj
        self.edge_mask_archive = []
        self.coef = coef
        
        self.use_bn = use_bn
        self.use_res = use_res
        
        self.step = 0
        
        in_dim = embedding_dim[0]
        hidden_dim = embedding_dim[1]
        n_classes = embedding_dim[-1]
        self.n_classes = n_classes
        
  
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        self.device = device
        
        self.gcnlayers = torch.nn.ModuleList()
        
        
        for layer in range(self.n_layers):
            if layer == 0:
                nodeFunc = ApplyNodeFunc(in_dim, hidden_dim)
                self.gcnlayers.append(GCNLayer(nodeFunc,self.dropout,device=self.device,  residual=self.use_res,activate=self.relu))
            elif layer == self.n_layers - 1:
                nodeFunc = ApplyNodeFunc(hidden_dim, n_classes)
                self.gcnlayers.append(GCNLayer(nodeFunc,self.dropout,device=self.device, residual=self.use_res))# non-activate
            else:
                nodeFunc = ApplyNodeFunc(hidden_dim, hidden_dim)
                self.gcnlayers.append(GCNLayer(nodeFunc,self.dropout, device=self.device,residual=self.use_res,activate=self.relu))

        if spar_adj:
            # use edge_learner cal edge score
            self.edge_learner = DegreeAwareEdgeScoring(in_dim,self.device)
    
    
    def forward(self,g, h, edge_mask, **kwargs):
        g.edata['mask'] = torch.ones(size=(g.number_of_edges(),)).to(h.device)
        if self.spar_adj:
            self.edge_score = self.edge_learner(g.edges(),h)
            self.edge_score = self.edge_score*edge_mask
            g.edata['mask'] = self.edge_score
        for i in range(self.n_layers):
            h = self.gcnlayers[i](g, h)
        return h


class SAGEConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
        

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout,edge_num, edge_index,args):
        super(SAGE, self).__init__()
        self.args = args
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.edge_num = edge_num
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, bias=False))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, bias=False))
        self.convs.append(SAGEConv(hidden_channels, out_channels, bias=False))
        self.edge_learner = DegreeAwareEdgeScoring(in_channels)
        self.edge_mask = nn.Parameter(torch.ones(self.edge_num,1),requires_grad=False)
        self.dropout = dropout
        self.edge_index =edge_index
        self.num_nodes = args['n_nodes']

    def forward(self, x, edge_index, val_test=False):
        self.edge_score = self.edge_learner(edge_index,x)
        mask = (~(self.edge_mask==0)).flatten()
        mask = mask.to(x.device)
        edge_index = edge_index[:, mask]
        pruned_values = self.edge_score[mask].flatten()
        adj_pruned = SparseTensor.from_edge_index(edge_index, pruned_values, 
                                sparse_sizes=(self.args['n_nodes'], self.args['n_nodes']))
        
        for conv in self.convs[:-1]:
            x = conv(x, adj_pruned)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_pruned)
        return x

