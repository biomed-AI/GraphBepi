import torch
import torch.nn as nn
import torch.nn.functional as F
class AE(nn.Module):
    def __init__(self, dim_in, dim_out, hidden, dropout = 0., bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden, bias=bias),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim_out, bias=bias),
            nn.LayerNorm(dim_out),
        )
    def forward(self, x):
        return self.net(x)
class EGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, edge_attr):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        e = e*edge_attr
        zero_vec = -9e15*torch.ones_like(e)
        e = torch.where(edge_attr > 0, e, zero_vec)
        e = F.softmax(e, dim=1)
        e = F.dropout(e, self.dropout, training=self.training)
        
        h_prime=[]
        for i in range(edge_attr.shape[0]):
            h_prime.append(torch.matmul(e[i],Wh))
        
        if self.concat:
            h_prime = torch.cat(h_prime,dim=1)
        else:
            h_prime = torch.stack(h_prime,dim=0).mean(0)
        return F.elu(h_prime),e

    #compute attention coefficient
    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
class EGAT(nn.Module):
    def __init__(self, nfeat, nhid, efeat, dropout=0.2, alpha=0.2):
        super().__init__()
        self.dropout = dropout
        self.in_att = EGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) 
        self.out_att = EGraphAttentionLayer(nhid*efeat, nfeat, dropout=dropout, alpha=alpha, concat=False)
    def forward(self, x, edge_attr):
        x_cut=x
        x = F.dropout(x, self.dropout, training=self.training)
        x, edge_attr=self.in_att(x, edge_attr)
        x, edge_attr=self.out_att(x, edge_attr)
        return x+x_cut, edge_attr