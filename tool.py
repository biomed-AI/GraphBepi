import torch
import torch.nn as nn
import torchmetrics as tm
import torch.nn.functional as F
class METRICS:
    def __init__(self,device='cpu'):
        self.device=device
        self.auroc=tm.AUROC().to(device)
        self.prc=tm.PrecisionRecallCurve().to(device)
        self.auc=tm.AUC(True).to(device)
        self.roc=tm.ROC().to(device)
        self.rec=tm.Recall().to(device)
        self.prec=tm.Precision().to(device)
        self.f1=tm.F1Score().to(device)
        self.mcc=tm.MatthewsCorrCoef(num_classes=2).to(device)
        self.bacc=tm.Accuracy(
            num_classes=2,average='macro',multiclass=True
        ).to(device)
    def to(self,pred,y):
        return pred.to(self.device),y.to(self.device)
    def calc_thresh(self,pred,y):
        pred,y=self.to(pred,y)
        prec, rec, thresholds = self.prc(pred,y)
        f1=(2*prec*rec/(prec+rec)).nan_to_num(0)[:-1]
        threshold = thresholds[torch.argmax(f1)]
        return threshold
    def calc_prc(self,pred,y):
        pred,y=self.to(pred,y)
        auroc = self.auroc(pred,y)
        prec, rec, th1 = self.prc(pred,y)
        auprc = self.auc(rec,prec)
        fpr, tpr, th2 = self.roc(pred,y)
        return {
            'AUROC':auroc.cpu().item(),'AUPRC':auprc.cpu().item(),'prc':[rec[:-1],prec[:-1],th1],'roc':[fpr,tpr,th2]
        }
    def __call__(self,pred,y,threshold=None):
        pred,y=self.to(pred,y)
        auroc = self.auroc(pred,y)
        prec, rec, thresholds = self.prc(pred,y)
        auprc = self.auc(rec,prec)
        if threshold is None:
            f1=(2*prec*rec/(prec+rec)).nan_to_num(0)[:-1]
            threshold = thresholds[torch.argmax(f1)]
        threshold=torch.tensor(threshold)
        self.f1.threshold=threshold
        self.rec.threshold=threshold
        self.mcc.threshold=threshold
        self.bacc.threshold=threshold
        self.prec.threshold=threshold
        f1 = self.f1(pred,y)
        rec = self.rec(pred,y)
        mcc = self.mcc(pred,y)
        bacc = self.bacc(pred,y)
        prec = self.prec(pred,y)
        return {
            'AUROC':auroc.cpu().item(),'AUPRC':auprc.cpu().item(),
            'RECALL':rec.cpu().item(),'PRECISION':prec.cpu().item(),
            'F1':f1.cpu().item(),'MCC':mcc.cpu().item(),
            'BACC':bacc.cpu().item(),'threshold':threshold.cpu().item(),
        }