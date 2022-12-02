import os
import esm
import time
import torch
import random
import esmfold
import warnings
import argparse
import numpy as np
import pandas as pd
import pickle as pk
import pytorch_lightning as pl
from tool import METRICS
from tqdm import tqdm,trange
from model import GraphBepi
from dataset import PDB,collate_fn,chain
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
parser.add_argument('--gpu', type=int, default=0, help='gpu.')
parser.add_argument('--seed', type=int, default=2022, help='random seed.')
parser.add_argument('-i','--input', type=str, help='input file')
parser.add_argument('-o','--output', type=str, default='./output', help='output path')
group =parser.add_mutually_exclusive_group()
group.add_argument('-p','--pdb', action='store_true', help='pdb format')
group.add_argument("-f","--fasta",action='store_true', help='fasta format')
args = parser.parse_args()
device='cpu' if args.gpu==-1 else f'cuda:{args.gpu}'
seed_everything(args.seed)
if not os.path.exists(args.output):
    os.mkdir(args.output)
with open(args.input,'r') as f:
    if args.pdb:
        pass
    elif args.fasta:
        pass


