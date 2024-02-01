import argparse
import utils
import os, sys
import logging
import glob
import torch

def ed(param_name, default=None):
    return os.environ.get(param_name, default)

def parser_loader():
    parser = argparse.ArgumentParser(description='GST')
    parser.add_argument('--total_epoch', type=int, default=500)
    parser.add_argument('--pretrain_epoch', type=int, default=0)
    parser.add_argument("--retrain_epoch", type=int, default=0)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:7")
    
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--spar_wei", default=False, action='store_true')
    parser.add_argument("--spar_adj", default=False, action='store_true')
    parser.add_argument('--model_save_path', type=str, default='model_ckpt',)
    parser.add_argument('--save', type=str, default='CKPTs',
                        help='experiment name')
    parser.add_argument("--e1", type=float, default=2e-6)
    parser.add_argument("--e2", type=float, default=2e-3)
    parser.add_argument("--coef", type=float, default=0.01)

    parser.add_argument('--remain', default=ed('REMAIN'), type=float,
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
    parser.add_argument('--topo_k', type=float, default=ed('TOPO_K', 1), metavar='M',
                        help='Learning rate step gamma (default: 1)')
    parser.add_argument('--sema_k', type=float, default=ed('sema_k', 1e4), metavar='M',
                        help='Initial pruning number(default: 1e4)')
    
    args = vars(parser.parse_args())
    seed_dict = {'cora': 1899, 'citeseer': 17889, 'pubmed': 3333,'Chameleon':1111,'Squirrel':2222}
    args['seed'] = seed_dict[args['dataset']]
    torch.cuda.device(int(args['device'][-1]))

    if args['dataset'] == "cora":
        args['embedding_dim'] = [1433, 512, 7]
    elif args['dataset'] == "citeseer":
        args['embedding_dim'] = [3703, 512, 6]
    elif args['dataset'] == "pubmed":
        args['embedding_dim'] = [500, 512, 3]
    elif args['dataset'] == "Chameleon":
        args['embedding_dim'] = [2325,] +  [64,] * (args['num_layers'] - 1) + [5]
    elif args['dataset'] == "Squirrel":
        args['embedding_dim'] = [2089,] +  [64,] * (args['num_layers'] - 1) + [5]
    else:
        raise NotImplementedError("dataset not supported.")

    args["model_save_path"] = os.path.join(
        args["save"], args["model_save_path"])
    utils.create_exp_dir(args["save"], scripts_to_save=glob.glob('*.py'))
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO,
                        format=log_format,
                        datefmt='%m/%d %I:%M:%S %p')

    return args

