import os
import esm
import time
import torch
import random
import warnings
import argparse
import numpy as np
import pandas as pd
import pickle as pk
import pytorch_lightning as pl
from tqdm import tqdm
from tool import METRICS
from utils import process_chain
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
parser.add_argument('-t','--threshold', type=float, default=0.1763, help='threshold.')
parser.add_argument('-i','--input', type=str, help='input file')
parser.add_argument('-o','--output', type=str, default='./output', help='output path')
group =parser.add_mutually_exclusive_group()
group.add_argument('-p','--pdb', action='store_true', help='pdb format')
group.add_argument("-f","--fasta",action='store_true', help='fasta format')
args = parser.parse_args()
device='cpu' if args.gpu==-1 else f'cuda:{args.gpu}'
seed_everything(args.seed)
print('Preparing...')
tmp_root='./data/tmp'
if not os.path.exists(args.output):
    os.makedirs(args.output)
if not os.path.exists(tmp_root):
    os.makedirs(tmp_root)
    os.system(f'cd {tmp_root} && mkdir PDB purePDB feat dssp graph')
esm2, _=esm.pretrained.esm2_t36_3B_UR50D()
esm2=esm2.to(device)
esm2.eval()
if args.pdb:
    os.system(f'cp {args.input} {tmp_root}/purePDB')
    with open(args.input,'r') as f:
        pid=args.input.split('/')[-1].split('.')[0]
        print('Processing...')
        data=chain()
        data.name=pid
        data=process_chain(data,tmp_root,pid,esm2,device)
        chains=[data]
elif args.fasta:
    esmfold = esm.pretrained.esmfold_v1()
    esmfold = esmfold.eval().to(device)
    with open(args.input,'r') as f:
        lines=f.readlines()
        seqs={}
        chains=[]
        for i in range(0,len(lines),2):
            fid=lines[i][1:-1].split('|')[0]
            fasta=lines[i+1][:-1]
            seqs[fid]=fasta
    print('Running ESMfold...')
    err_info=''
    for i,j in tqdm(seqs.items()):
        try:
            with torch.no_grad():
                output = esmfold.infer_pdb(j)
        except RuntimeError as e:
            # if e.args[0].startswith("CUDA out of memory"):
            #     print(f"Failed (CUDA out of memory) on sequence {i} of length {len(j)}.")
            # else:
            #     print(f'Unknown error on sequence {i}.')
            err_info+=f'{i}, '
            continue
        with open(f"{tmp_root}/purePDB/{i}.pdb", "w") as f:
            f.write(output)
    if err_info!='':
        print('Sequences failed to predict structure:'+err_info[:-2])
    print('Processing...')
    for i,j in tqdm(seqs.items()):
        if not os.path.exists(f"{tmp_root}/purePDB/{i}.pdb"):
            continue
        data=chain()
        data.name=i
        data=process_chain(data,tmp_root,i,esm2,device)
        chains.append(data)
idx=np.array(range(len(chains)))
np.save(f'{tmp_root}/cross-validation.npy',idx)
with open(f'{tmp_root}/test.pkl','wb') as f:
    pk.dump(chains,f)
with open(f'{tmp_root}/test.pkl','rb') as f:
    chains=pk.load(f)
print('Predicting...')
testset=PDB(mode='test',root=tmp_root)
test_loader=DataLoader(testset,batch_size=4,shuffle=False,collate_fn=collate_fn)
model=GraphBepi(
    feat_dim=2560,                     # esm2 representation dim
    hidden_dim=256,                    # hidden representation dim
    exfeat_dim=13,                     # dssp feature dim
    edge_dim=51,                       # edge feature dim
    augment_eps=0.05,                  # random noise rate
    dropout=0.2,
    result_path=f'{args.output}',      # path to save temporary result file of testset
)
model.load_state_dict(
    torch.load(f'./model/BCE_633_GraphBepi/model_-1.ckpt',map_location='cpu')['state_dict'],
)
trainer = pl.Trainer(gpus=[args.gpu],logger=None)
result = trainer.test(model,test_loader)
pred=torch.load(f'{args.output}/result.pkl')['pred']
IDX=[]
for i in range(len(testset)):
    IDX+=[i]*len(testset.data[i])
IDX=torch.LongTensor(IDX)
for i in range(len(testset)):
    idx=IDX==i
    predi=pred[idx]
    seqi=testset.data[i].sequence
    labeli=torch.where(predi>args.threshold,1,0).bool()
    df=pd.DataFrame({'resn':list(seqi),'score':predi,'is epitope':labeli})
    df.to_csv(f'{args.output}/{testset.data[i].name}.csv',index=False)
os.remove(f'{args.output}/result.pkl')
print('Fin')

        


