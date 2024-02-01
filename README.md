# Usage

## GCN
Please `cd GST_gcn` first, then use the following commands to check the extreme graph sparsity that **GST** is capable of achieving. 
```
python main.py --remain 0.4 --spar_adj --num_layers 2 --dataset cora --lr 0.001 --delta 20 --accumulation-n 3 --pretrain_epoch 200 --warmup_steps 6 --beta 0.3
python main.py --remain 0.4 --spar_adj --num_layers 2 --dataset citeseer --lr 0.001 --delta 20 --accumulation-n 3 --pretrain_epoch 200 --warmup_steps 6 --beta 0.3
python main.py --remain 0.4 --spar_adj --num_layers 2 --dataset pubmed --lr 0.001 --delta 20 --accumulation-n 3 --pretrain_epoch 200 --warmup_steps 6 --beta 0.3
```


## GIN
For GIN demonstration, please `cd GST_gin` first. 
```
python main.py --remain 0.4 --spar_adj --num_layers 2 --dataset cora --lr 0.001 --delta 20 --accumulation-n 3 --pretrain_epoch 200
python main.py --remain 0.4 --spar_adj --num_layers 2 --dataset citeseer --lr 0.001 --delta 20 --accumulation-n 3 --pretrain_epoch 200
python main.py --remain 0.4 --spar_adj --num_layers 2 --dataset pubmed --lr 0.001 --delta 20 --accumulation-n 3 --pretrain_epoch 200
```

## GAT
For GAT demonstration, please `cd GST_GAT` first.
```
python main.py --remain 0.4 --spar_adj --num_layers 2 --dataset cora --lr 0.001 --delta 20 --accumulation-n 3 --pretrain_epoch 200
python main.py --remain 0.4 --spar_adj --num_layers 2 --dataset citeseer --lr 0.001 --delta 20 --accumulation-n 3 --pretrain_epoch 200
python main.py --remain 0.4 --spar_adj --num_layers 2 --dataset pubmed --lr 0.001 --delta 20 --accumulation-n 3 --pretrain_epoch 200
```

## OGBN_arxiv
For OGBN_arxiv demonstration, execute the following commands.
```
python main_graphsage.py --use_gpu --total_epoch 300 --spar_adj --remain 0.4 --pretrain_epoch 600  --delta 3 --accumulation-n 2
```
