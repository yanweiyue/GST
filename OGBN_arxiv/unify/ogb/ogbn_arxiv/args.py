import __init__
import argparse
import uuid
import logging
import time
import os
import sys

from utils.logger import create_exp_dir
import glob

def ed(param_name, default=None):
    return os.environ.get(param_name, default)

def parser_loader():
    parser = argparse.ArgumentParser(description='GST')

    ### pruning settings
    parser.add_argument('--total_epoch', type=int, default=400)
    parser.add_argument('--pretrain_epoch', type=int, default=30)
    parser.add_argument("--retrain_epoch", type=int, default=0)
    parser.add_argument("--spar_wei", default=False, action='store_true')
    parser.add_argument("--spar_adj", default=False, action='store_true')
    parser.add_argument('--s1', type=float, default=1e-6,help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--s2', type=float, default=1e-4,help='scale sparse rate (default: 0.0001)')

    parser.add_argument('--resume_dir', type=str, default='')
    parser.add_argument('--seed', type=int, default=10, help='which seed to use if any (default: 0)')
    parser.add_argument('--mask_epochs', type=int, default=200,
                            help='number of epochs to train (default: 500)')
    parser.add_argument('--fix_epochs', type=int, default=500,
                            help='number of epochs to train (default: 500)')                            
    parser.add_argument('--fixed', default='', type=str, help='{all_fixed, only_adj, only_wei, no_fixed}')

        # parser.add_argument('--baseline', action='store_true')
        # dataset
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv',
                            help='dataset name (default: ogbn-arxiv)')
    parser.add_argument('--self_loop', action='store_true')
        # training & eval settings
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument('--weight-decay', type=float, default=5e-4)

    parser.add_argument('--lr', type=float, default=0.01,
                            help='learning rate set for optimizer.')
    parser.add_argument('--dropout', type=float, default=0.5)
    # model
    parser.add_argument('--num_layers', type=int, default=2,
                            help='the number of layers of the networks')
    parser.add_argument('--mlp_layers', type=int, default=1,
                            help='the number of layers of mlp in conv')
    parser.add_argument('--in_channels', type=int, default=128,
                            help='the dimension of initial embeddings of nodes')
    parser.add_argument('--hidden_channels', type=int, default=128,
                            help='the dimension of embeddings of nodes')
    parser.add_argument('--block', default='res+', type=str,
                            help='graph backbone block type {res+, res, dense, plain}')
    parser.add_argument('--conv', type=str, default='gen',
                            help='the type of GCNs')
    parser.add_argument('--gcn_aggr', type=str, default='softmax_sg',
                            help='the aggregator of GENConv [mean, max, add, softmax, softmax_sg, softmax_sum, power, power_sum]')
    parser.add_argument('--norm', type=str, default='batch',
                            help='the type of normalization layer')
    parser.add_argument('--num_tasks', type=int, default=1,
                            help='the number of prediction tasks')
    # message norm
    parser.add_argument('--msg_norm', action='store_true')
    parser.add_argument('--learn_msg_scale', action='store_true')
    # save model
    parser.add_argument('--model_save_path', type=str, default='model_ckpt',
                            help='the directory used to save models')
    parser.add_argument('--save', type=str, default='CKPTs', help='experiment name')
    # load pre-trained model
    parser.add_argument('--model_load_path', type=str, default='ogbn_arxiv_pretrained_model.pth',
                            help='the path of pre-trained model')

    parser.add_argument('--remain', default=ed('remain'), type=float,
                        help='percentage of remained parameters allowed. if None, pruning will not be used. must be on the interval (0, 1]')
    parser.add_argument('--delta', default=ed('DELTA', 3), type=int,
                        help='delta param for pruning')
    parser.add_argument('--accumulation-n', default=ed('GRAD_ACCUMULATION_N', 2), type=int,
                        help='number of gradients to accumulate before scoring for grow and prune')
    parser.add_argument('--alpha', default=ed('ALPHA', 0.3), type=float,
                        help='alpha param for pruning')
    parser.add_argument('--static-topo', default=ed('STATIC_TOPO', 0), type=int, help='if 1, use random sparsity topo and remain static')
    parser.add_argument('--gamma', type=float, default=ed('GAMMA', 0.7), metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--beta', type=float, default=ed('BETA', 1), metavar='M',
                        help='Initial pruning number(default: 1)')
    parser.add_argument('--warmup_steps', default=ed('WARMUP_STEPS', 0), type=int, help='if 0, completely one shot')
    parser.add_argument('--inductive',default=False, action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_partitions', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--topo_k', type=float, default=ed('TOPO_K', 1), metavar='M',
                        help='Learning rate step gamma (default: 1)')
    parser.add_argument('--sema_k', type=float, default=ed('sema_k', 1e4), metavar='M',
                        help='Initial pruning number(default: 1e4)')
    args = vars(parser.parse_args())
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO,
                        format=log_format,
                        datefmt='%m/%d %I:%M:%S %p')
    return args
