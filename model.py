import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from EGAT import EGAT,AE
from torch.nn.utils.rnn import pad_sequence,pack_sequence,pack_padded_sequence,pad_packed_sequence
class GraphBepi(pl.LightningModule):
    def __init__(
        self, 
        feat_dim=2560, hidden_dim=256, 
        exfeat_dim=13, edge_dim=51, 
        augment_eps=0.05, dropout=0.2, 
        lr=1e-6, metrics=None, result_path=None
    ):
        super().__init__()
        self.metrics=metrics
        self.path=result_path
        # loss function
        self.loss_fn=nn.BCELoss()
        # Hyperparameters
        self.exfeat_dim=exfeat_dim
        self.augment_eps = augment_eps
        self.lr = lr
        self.cls = 1
        bias=False
        self.W_v = nn.Linear(feat_dim, hidden_dim, bias=bias)
        self.W_u1 = AE(exfeat_dim,hidden_dim,hidden_dim, bias=bias)
        self.edge_linear=nn.Sequential(
            nn.Linear(edge_dim,hidden_dim//4, bias=True),
            nn.ELU(),
        )
        self.gat=EGAT(2*hidden_dim,hidden_dim,hidden_dim//4,dropout)
        self.lstm1 = nn.LSTM(hidden_dim,hidden_dim//2,3,batch_first=True,bidirectional=True,dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_dim,hidden_dim//2,3,batch_first=True,bidirectional=True,dropout=dropout)
        # output
        self.mlp=nn.Sequential(
            nn.Linear(4*hidden_dim,hidden_dim,bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim,1,bias=True),
            nn.Sigmoid()
        )
        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, V, edge):
        h=[]
        V = pad_sequence(V, batch_first=True, padding_value=0).float()
        mask=V.sum(-1)!=0
        if self.training and self.augment_eps > 0:
            aug=torch.randn_like(V)
            aug[~mask]=0
            V = V+self.augment_eps * aug
        mask=mask.sum(1)
        feats,dssps=self.W_v(V[:,:,:-self.exfeat_dim]),self.W_u1(V[:,:,-self.exfeat_dim:])
        x_gcns=[]
        for i in range(len(V)):
            E=self.edge_linear(edge[i]).permute(2,0,1)
            x1,x2=feats[i,:mask[i]],dssps[i,:mask[i]]
            x_gcn=torch.cat([x1,x2],-1)
            x_gcn,E=self.gat(x_gcn,E)
            x_gcns.append(x_gcn)
        feats=pack_padded_sequence(feats,mask.cpu(),True,False)
        dssps=pack_padded_sequence(dssps,mask.cpu(),True,False)
        feats=pad_packed_sequence(self.lstm1(feats)[0],True)[0]
        dssps=pad_packed_sequence(self.lstm2(dssps)[0],True)[0]
        x_attns=torch.cat([feats,dssps],-1)
        
        x_attns=[x_attns[i,:mask[i]] for i in range(len(x_attns))]
        h=[torch.cat([x_attn,x_gcn],-1) for x_attn,x_gcn in zip(x_attns,x_gcns)]
        h=torch.cat(h,0)
        return self.mlp(h)
    def training_step(self, batch, batch_idx): 
        feat, edge, y = batch
        pred = self(feat, edge).squeeze(-1)
        loss=self.loss_fn(pred,y.float())
        self.log('train_loss', loss.cpu().item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if self.metrics is not None:
            result=self.metrics.calc_prc(pred.detach().clone(),y.detach().clone())
            self.log('train_auc', result['AUROC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('train_prc', result['AUPRC'], on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        feat, edge, y = batch
        pred = self(feat, edge).squeeze(-1)
        return pred,y
    def validation_epoch_end(self,outputs):
        pred,y=[],[]
        for i,j in outputs:
            pred.append(i)
            y.append(j)
        pred=torch.cat(pred,0)
        y=torch.cat(y,0)
        loss=self.loss_fn(pred,y.float())
        self.log('val_loss', loss.cpu().item(), on_epoch=True, prog_bar=True, logger=True)
        if self.metrics is not None:
            result=self.metrics(pred.detach().clone(),y.detach().clone())
            self.log('val_AUROC', result['AUROC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('val_AUPRC', result['AUPRC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('val_mcc', result['MCC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('val_f1', result['F1'], on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        feat, edge, y = batch
        pred = self(feat, edge).squeeze(-1)
        return pred,y
    def test_epoch_end(self,outputs):
        pred,y=[],[]
        for i,j in outputs:
            pred.append(i)
            y.append(j)
        pred=torch.cat(pred,0)
        y=torch.cat(y,0)
        loss=self.loss_fn(pred,y.float())
        if self.path:
            if not os.path.exists(self.path):
                os.system(f'mkdir -p {self.path}')
            torch.save({'pred':pred.cpu(),'gt':y.cpu()},f'{self.path}/result.pkl')
        if self.metrics is not None:
            result=self.metrics(pred.detach().clone(),y.detach().clone())
            self.log('test_loss', loss.cpu().item(), on_epoch=True, prog_bar=True, logger=True)
            self.log('test_AUROC', result['AUROC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('test_AUPRC', result['AUPRC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('test_recall', result['RECALL'], on_epoch=True, prog_bar=True, logger=True)
            self.log('test_precision', result['PRECISION'], on_epoch=True, prog_bar=True, logger=True)
            self.log('test_f1', result['F1'], on_epoch=True, prog_bar=True, logger=True)
            self.log('test_mcc', result['MCC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('test_bacc', result['BACC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('test_threshold', result['threshold'], on_epoch=True, prog_bar=True, logger=True)
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), betas=(0.9, 0.99), lr=self.lr, weight_decay=1e-5, eps=1e-5)
