import os
import time
import torch
import random
import warnings
import argparse
import numpy as np
import pandas as pd
import pickle as pk
import torch.nn as nn
import torchmetrics as tm
import pytorch_lightning as pl
import torch.nn.functional as F
from tool import METRICS
from tqdm import tqdm,trange
from model import GraphBepi
from dataset import PDB,collate_fn,chain
from collections import defaultdict
from torch.utils.data import DataLoader,Dataset
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback,EarlyStopping,ModelCheckpoint
warnings.simplefilter('ignore')
def seed_everything(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-6, help='learning rate.')
parser.add_argument('--gpu', type=int, default=1, help='gpu.')
parser.add_argument('--fold', type=int, default=-1, help='dataset fold. set it -1 to use the whole trainset')
parser.add_argument('--seed', type=int, default=2022, help='random seed.')
parser.add_argument('--batch', type=int, default=4, help='batch size.')
parser.add_argument('--hidden', type=int, default=256, help='hidden dim.')
parser.add_argument('--epochs', type=int, default=300, help='max number of epochs.')
parser.add_argument('--dataset', type=str, default='BCE_633', help='dataset name.')
parser.add_argument('--logger', type=str, default='./log', help='logger path.')
parser.add_argument('--tag', type=str, default='GraphBepi', help='logger name.')
args = parser.parse_args()

device='cpu' if args.gpu==-1 else f'cuda:{args.gpu}'
seed_everything(args.seed)
root=f'./data/{args.dataset}'

trainset=PDB(mode='train',fold=args.fold,root=root)
valset=PDB(mode='val',fold=args.fold,root=root)
testset=PDB(mode='test',fold=args.fold,root=root)

train_loader=DataLoader(trainset,batch_size=args.batch,shuffle=True,collate_fn=collate_fn,drop_last=True)
val_loader=DataLoader(valset,batch_size=args.batch,shuffle=False,collate_fn=collate_fn)
test_loader=DataLoader(testset,batch_size=args.batch,shuffle=False,collate_fn=collate_fn)
if args.fold==-1:
    val_loader=test_loader
log_name=f'{args.dataset}_{args.tag}'
metrics=METRICS(device)
model=GraphBepi(
    feat_dim=2560,                     # esm2 representation dim
    hidden_dim=args.hidden,            # hidden representation dim
    exfeat_dim=13,                     # dssp feature dim
    edge_dim=51,                       # edge feature dim
    augment_eps=0.05,                  # random noise rate
    dropout=0.2,
    lr=args.lr,                        # learning rate
    metrics=metrics,                   # an implement to compute performance
    result_path=f'./model/{log_name}', # path to save temporary result file of testset
)
es=EarlyStopping('val_AUPRC',patience=40,mode='max')
mc=ModelCheckpoint(
    f'./model/{log_name}/',f'model_{args.fold}',
    'val_AUPRC',
    mode='max',
    save_weights_only=True, 
)
logger = TensorBoardLogger(
    args.logger, 
    name=log_name+f'_{args.fold}'
)
cb=[mc,es]
trainer = pl.Trainer(
    gpus=[args.gpu] if args.gpu!=-1 else None, 
    max_epochs=args.epochs, callbacks=cb,
    logger=logger,check_val_every_n_epoch=1,
)
trainer.fit(model, train_loader, val_loader)
model.load_state_dict(
    torch.load(f'./model/{log_name}/model_{args.fold}.ckpt')['state_dict'],
)
trainer = pl.Trainer(gpus=[args.gpu],logger=None)
result = trainer.test(model,test_loader)
os.rename(f'./model/{log_name}/result.pkl',f'./model/{log_name}/result_{args.fold}.pkl')