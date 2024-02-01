import math
import numpy as np
import pickle as pkl
import networkx as nx
import dgl
import scipy.sparse as sp
# from scipy.sparse.linalg.eigen.arpack import eigsh
from typing import Optional
import random
import sys
import os
import pdb
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.utils.num_nodes import maybe_num_nodes
import shutil
from torch_geometric.datasets import WikipediaNetwork
import scipy.optimize as optimize

# from dgl.data import CoraGraphDataset, KarateClubDataset, CiteseerGraphDataset, PubmedGraphDataset
# from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.model_selection import train_test_split

from normalization import fetch_normalization, row_normalize


datadir = "data"


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def get_dataset(name:str, root:str="data/" ):
    if name in ['Chameleon', 'Squirrel']:
        preProcDs = WikipediaNetwork(
            root=root, name=name, geom_gcn_preprocess=False, transform=T.NormalizeFeatures())
        dataset = WikipediaNetwork(
            root=root, name=name, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.edge_index = preProcDs[0].edge_index

    num_nodes = data.y.shape[0]
    all_idx = np.array(range(num_nodes))
    dataset_split = [0.6, 0.2, 0.2] # train/val/test
    np.random.shuffle(all_idx)
    data.train_idx = torch.LongTensor(all_idx[:int(num_nodes*dataset_split[0])])
    data.val_idx = torch.LongTensor(all_idx[int(num_nodes*dataset_split[0]):int(num_nodes*dataset_split[0])+int(num_nodes*dataset_split[1])])
    data.test_idx = torch.LongTensor(all_idx[int(num_nodes*dataset_split[0])+int(num_nodes*dataset_split[1]):])
    print(f"train: {len(data.train_idx)} nodes | val: {len(data.val_idx)} nodes | test: {len(data.test_idx)} nodes")
    return data.edge_index, data.x, data.y, data.train_idx, data.val_idx, data.test_idx,[],[]

def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("./data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    # preprocess feature
    features = preprocess_features(features)
    features = torch.tensor(features, dtype=torch.float32)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # preprocess adj
    adj = sparse_mx_to_torch_sparse_tensor(adj).to_dense()
    # adj = torch_normalize_adj(adj)
    # adj2 = preprocess_adj(adj)
    # adj2 = sparse_mx_to_torch_sparse_tensor(adj2).to_dense()
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    _, l_num = labels.shape
    labels = torch.tensor((labels * range(l_num)).sum(axis=1), dtype=torch.int64)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y)+500))

    #print(f"train: {len(idx_test)} val: {len(idx_val)} test: {len(idx_test)}")

    return adj, features, labels, idx_train, idx_val, idx_test

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    #return sparse_to_tuple(features)
    return features.todense()


def torch_normalize_adj(adj, device):
    # adj = adj + torch.eye(adj.shape[0]).to(device)
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt).to(device)
    return adj.mm(d_mat_inv_sqrt).t().mm(d_mat_inv_sqrt)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    #return sparse_to_tuple(adj_normalized)
    return adj_normalized


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_adj_raw(dataset_str):

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("./data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    adj_raw = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    return adj_raw

def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features

def load_citation(dataset_str="cora", normalization="AugNormAdj", porting_to_torch=True,data_path=datadir, task_type="semi"):
    """
    Load Citation Networks Datasets.
    """
    if not dataset_str in ['cora','pubmed','citeseer']:
        return get_dataset(dataset_str)
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str.lower(), names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # degree = np.asarray(G.degree)
    degree = np.sum(adj, axis=1)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    
    if task_type == "full":
        #print("Load full supervised task.")
        #supervised setting
        idx_test = test_idx_range.tolist()
        idx_train = range(len(ally)- 500)
        idx_val = range(len(ally) - 500, len(ally))
    elif task_type == "semi":
        #print("Load semi-supervised task.")
        #semi-supervised setting
        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)
    else:
        raise ValueError("Task type: %s is not supported. Available option: full and semi.")

    adj, features = preprocess_citation(adj, features, normalization)
    features = np.array(features.todense())
    labels = np.argmax(labels, axis=1)
    # porting to pytorch
    if porting_to_torch:
        features = torch.FloatTensor(features).float()
        labels = torch.LongTensor(labels)
        # labels = torch.max(labels, dim=1)[1]
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        degree = torch.LongTensor(degree)
    learning_type = "transductive"
    
    print(f"train: {len(idx_train)} val: {len(idx_val)} test: {len(idx_test)}")
    return adj, features, labels, idx_train, idx_val, idx_test, degree, learning_type


def fix_seed(seed):    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
            for script in scripts_to_save:
                dst_file = os.path.join(
                    path, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)


EXCLUDED_TYPES = (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d) # exclude the norm layer
EXCLUDED_PARAS = ("adj_mask1_train")

# Get the layer with the attribute weight
def get_weighted_layers(model, i=0, layers=None,paras=None, linear_layers_mask=None, parameters_mask=None):
    if layers is None:
        layers = []  # initialize the layers and linear_layers_masks
    if paras is None:
        paras = []
    if linear_layers_mask is None:
        linear_layers_mask = []
    if parameters_mask is None:
        parameters_mask = []
    items = model._modules.items()
    
    if i == 0:
        items = [(None, model)]
        for para_name,p in model._parameters.items():  # insert parameters
            if p.requires_grad:
                paras.append(p)
                linear_layers_mask.append(0)
                parameters_mask.append(1)
    
    for layer_name, p in items:  # Iterate through all submodules of the model
        if isinstance(p, torch.nn.Linear):  # if submodule is Linear
            layers.append([p])
            linear_layers_mask.append(1)  # linear_layers_masks record whether the submodule is linear_layer
            parameters_mask.append(0)
        elif hasattr(p, 'weight') and type(p) not in EXCLUDED_TYPES:
            layers.append([p])
            linear_layers_mask.append(0)
            parameters_mask.append(0)
        else:
            _,_, linear_layers_mask,parameters_mask, i = get_weighted_layers(p, i=i + 1, layers=layers,paras=paras, linear_layers_mask=linear_layers_mask,parameters_mask=parameters_mask)  #? can be merged with the previous else if
    return layers,paras, linear_layers_mask,parameters_mask, i 


# get the weight list
def get_W(model, return_linear_layers_mask=False):
    layers, paras, linear_layers_mask,parameters_mask, _ = get_weighted_layers(model)
    
    W = []
    for para in paras:
        W.append(para)
    
    for layer in layers:
        idx = 0 if hasattr(layer[0], 'weight') else 1  # if the layer has attribute 
        W.append(layer[idx].weight)
    assert len(W) == len(linear_layers_mask)
    if return_linear_layers_mask:  
        return W, linear_layers_mask,parameters_mask
    return W

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
def add_trainable_mask_noise(model, c=1e-5):
    
    model.adj_mask1_train.requires_grad = False
   
    rand1 = (2 * torch.rand(model.adj_mask1_train.shape) - 1) * c
    rand1 = rand1.to(model.adj_mask1_train.device) 
    rand1 = rand1 * model.adj_mask1_train
    model.adj_mask1_train.add_(rand1)

    model.adj_mask1_train.requires_grad = True

def exp_diversity_loss(v):
    loss = torch.exp(-0.5*torch.abs(v.unsqueeze(1) - v.unsqueeze(0) ).mean())
    return loss

def log_variance_loss(v):
    loss = -torch.log(torch.var(v) + 1e-8)
    return loss



def compute_edge_score_from_edge_index(edge_index, num_nodes,masked,device):
    # cal degree
    row, col = edge_index
    row_mask = row[masked]
    col_mask = col[masked]
    edge_index_masked = torch.stack((row_mask,col_mask),dim=1)
    # through deg to compute 
    deg = torch.zeros(num_nodes, dtype=torch.float,device=device)
    # deg.scatter_add_(0, row, torch.ones(row.size(0), dtype=torch.float,device=device))
    deg.scatter_add_(0, row_mask, torch.ones(row_mask.size(0), dtype=torch.float,device=device))
    # cal deg_inv and deg_inv_sqrt
    deg_inv = deg.pow(-1)
    deg_inv.masked_fill_(deg_inv == float('inf'), 1.21)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 1.1)
    
    edge_importance = deg_inv_sqrt[row] * deg_inv[row] * deg_inv_sqrt[col] * deg_inv[col]
    return edge_importance

def get_topo_criterion(dataset,backbone):
    topo_criterion=torch.load(f"./stat/{dataset}_{backbone}.pt")
    return topo_criterion