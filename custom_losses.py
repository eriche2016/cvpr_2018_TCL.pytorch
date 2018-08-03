import torch 
import torch.nn as nn 
import torch.nn.parallel
import torch.nn.functional as F 
from torch.autograd import Variable 
from torch.nn import Parameter 
import numpy as np 

from IPython.core.debugger import Tracer
debug_here = Tracer() 

################################################################
## Triplet related loss 
################################################################
def pdist(A, squared=False, eps=1e-4):
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod) 
    res = (norm + norm.t() - 2 * prod).clamp(min = 0) 
    return res if squared else (res + eps).sqrt() + eps 


class LiftedStructLoss(nn.Module):
    def __init__(self, margin=1.0, eps=1e-4):
        super(LiftedStructLoss, self).__init__() 
        self.margin = margin
        self.eps = eps 

    def forward(self, features, y):
        d = pdist(features, squared = False, eps = eps)
        pos = torch.eq(*[y.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(features)
        neg_i = torch.mul((margin - d).exp(), 1 - pos).sum(1).expand_as(d)

        return torch.sum(torch.mul(pos.triu(1), torch.log(neg_i + neg_i.t()) + d).clamp(min = 0).pow(2)) / (pos.sum() - len(d))

class Triplet(nn.Module):
    def __init__(self, margin=1.0):
        super(Triplet, self).__init__() 
        self.margin = margin 

    def forward(self, features, y):
        d = pdist(features, squared = False)
        # pos[i][j]: if i and j are positive, then it will be 1, otherwise, it is 0 
        pos = torch.eq(*[y.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(features)
        T = d.unsqueeze(1).expand(*(len(d),) * 3) # T[0][0][j]:  first sample to sample j's distance
                                                  # T[0][k][j]:  first sample to sample j's distance
        # M[i][j][k], means POS[i][k][j] * (1 - POS[i][j][k])
        M = pos.unsqueeze(1).expand_as(T) * (1 - pos.unsqueeze(2).expand_as(T))
        # T[i][k][j] - T[i][j][k] + margin 
        return (M * torch.clamp(T - T.transpose(1, 2) + margin, min = 0)).sum() / M.sum()

class TripletRatio(nn.Module):
    def __init__(self, margin=0.1, eps=1e-4):
        super(TripletRatio, self).__init__() 
        self.margin = margin 
        self.eps = eps 

    def forward(self, features, y):
        d = pdist(features, squared = False, eps = eps)
        pos = torch.eq(*[y.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(features)
        T = d.unsqueeze(1).expand(*(len(d),) * 3) # [i][k][j]
        M = pos.unsqueeze(1).expand_as(T) * (1 - pos.unsqueeze(2).expand_as(T))
        return (M * T.div(T.transpose(1, 2) + margin)).sum() / M.sum() #[i][k][j] = 


class Pddm(nn.Module):
    def __init__(self, output_dim, alpha=0.5, beta=1.0, Lambda=0.5):
        super(Pddm, self).__init__()
        self.d = output_dim
        self.Alpha = Alpha 
        self.Beta = Beta 
        self.Lambda = Lambda
        self.wu = nn.Linear(self.d, self.d)
        self.wv = nn.Linear(self.d, self.d)
        self.wc = nn.Linear(2 * self.d, self.d)
        self.ws = nn.Linear(self.d, 1)
        self.dropout = nn.Dropout(p = 0.5)

    def forward(self, features, y):
        d = pdist(features, squared = True)
        pos = torch.eq(*[y.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(features)

        f1, f2 = [features.detach().unsqueeze(dim).expand(len(features), *features.size()) for dim in [0, 1]]
        u = (f1 - f2).abs()
        v = (f1 + f2) / 2
        u_ = F.normalize(F.relu(self.dropout(self.wu(u.view(-1, u.size(-1))))))
        v_ = F.normalize(F.relu(self.dropout(self.wv(v.view(-1, v.size(-1))))))
        s = self.ws(F.relu(self.dropout(self.wc(torch.cat((u_, v_), -1))))).view_as(d)
        
        sneg = s * (1 - pos)
        i, j = min([(s.data[i, j], (i, j)) for i, j in pos.data.nonzero() if i != j])[1]
        k, l = sneg.max(1)[1].data.squeeze(1)[torch.cuda.LongTensor([i, j])]
        assert pos[i, j] == 1 and pos[i, k] == 0 and pos[j, l] == 0

        smin, smax = torch.min(sneg[i], sneg[j]).min().detach(), torch.max(sneg[i], sneg[j]).max().detach()
        s = (s - smin.expand_as(s)) / (smax - smin).expand_as(s)

        E_m = F.relu(self.Alpha + s[i, k] - s[i, j]) + F.relu(self.Alpha + s[j, l] - s[i, j])
        E_e = F.relu(self.Beta + d[i, j] - d[i, k]) + F.relu(self.Beta + d[i, j] - d[j, l])

        return E_m + self.Lambda * E_e
    
#########################################################################
## TripletLoss 
#########################################################################
# note that we can also use nn.CosineSimilarity, for details, see pytorch doc 
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """ return cosine similarity between x1 and x2, computed along dim 
        Args:
            x1 (Variable): first input 
            x2 (Variable): second input (of size matching x1)
            dim (int, optional): dimension of vector. Default: 1
            eps (float, optional): Small value to avoid division by zero. Default: 1e-8 
        Shape: 
            - Input1: `(*1, D, *2)`, where D is at position dim 
            - Input2: `(*1, D, *2)`, same shape as the Input1
            - Output: `(*1, *2)`
    """
    w12 = torch.sum(x1*x2, dim)
    w1 = torch.norm(x1, 2, dim) # l2 norm along dim 
    w2 = torch.norm(x2, 2, dim)
    return (w12/(w1*w2).clamp(min=eps)).squeeze() 

def cos_distance(self, a, b):
    return torch.dot(a, b)/(torch.norm(a)*torch.norm(b))
    
# we can also use nn.TripletMarginLoss(). see pytorch file loss.py for more details
class TripletMarginLoss(nn.Module):
    def __init__(self, margin, use_ohem=False, ohem_bs=128, dist_type=0):
        super(TripletMarginLoss, self).__init__() 
        self.margin = margin 
        self.dist_type = dist_type
        self.use_ohem = use_ohem
        self.ohem_bs = ohem_bs 

    def forward(self, anchor, positive, negative):
        # euclidean distance 
        if self.dist_type == 0:
            dist_p = F.pairwise_distance(anchor, positive)
            dist_n = F.pairwise_distance(anchor, negative)

        if self.dist_type == 1:
            dist_p = cosine_similarity(anchor, positive)
            dist_n = cosine_similarity(anchor, negative)

        dist_hinge = torch.clamp(dist_p-dist_n + self.margin, min=0.0)
        if self.use_ohem:
            v, idx = torch.sort(dist_hinge, descending=True)
            loss = torch.mean(v[0:self.ohem_bs])
        else:
            loss = torch.mean(dist_hinge)

        return loss 

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, loss_weight=1.0):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.loss_weight = loss_weight
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.use_cuda = False

    def forward(self, feat, y):
        # torch.histc can only be implemented on CPU
        # To calculate the total number of every class in one mini-batch. See Equation 4 in the paper
        if self.use_cuda:
            hist = Variable(torch.histc(y.cpu().data.float(),bins=self.num_classes,min=0,max=self.num_classes) + 1).cuda()
        else:
            hist = Variable(torch.histc(y.data.float(),bins=self.num_classes,min=0,max=self.num_classes) + 1)

        centers_count = hist.index_select(0,y.long())


        # To squeeze the Tenosr
        batch_size = feat.size()[0]

        feat = feat.view(batch_size, 1, 1, -1).squeeze() 
        # To check the dim of centers and features
        if feat.size()[1] != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim,feat.size()[1]))

        centers_pred = self.centers.index_select(0, y.long())
        diff = feat - centers_pred
        loss = self.loss_weight * 1 / 2.0 * (diff.pow(2).sum(1) / centers_count).sum()
        return loss

    ####### overriding cuda function ##### 
    def cuda(self, device_id=None):
        """Moves all model parameters and buffers to the GPU.
        Arguments:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        self.use_cuda = True # when calling cuda, we will set a flag, which will be used in forward function 
        return self._apply(lambda t: t.cuda(device_id))


# batch version
# find negative centers based on the labels of the batch   
class TripletCenter40Loss(nn.Module):
    def __init__(self, margin=0):
        super(TripletCenter40Loss, self).__init__() 
        self.margin = margin 
        self.ranking_loss = nn.MarginRankingLoss(margin=margin) 
        self.centers = nn.Parameter(torch.randn(40, 40)) # for modelent40
   
    def forward(self, inputs, targets): 
        batch_size = inputs.size(0) 
        targets_expand = targets.view(batch_size, 1).expand(batch_size, inputs.size(1)) 
        centers_batch = self.centers.gather(0, targets_expand) # centers batch 

        # compute pairwise distances between input features and corresponding centers 
        centers_batch_bz = torch.stack([centers_batch]*batch_size) 
        inputs_bz = torch.stack([inputs]*batch_size).transpose(0, 1) 
        dist = torch.sum((centers_batch_bz -inputs_bz)**2, 2).squeeze() 
        dist = dist.clamp(min=1e-12).sqrt() # for numerical stability 

        # for each anchor, find the hardest positive and negative 
        mask = targets.expand(batch_size, batch_size).eq(targets.expand(batch_size, batch_size).t())
        dist_ap, dist_an = [], [] 
        for i in range(batch_size): # for each sample, we compute distance 
            dist_ap.append(dist[i][mask[i]].max()) # mask[i]: positive samples of sample i
            dist_an.append(dist[i][mask[i]==0].min()) # mask[i]==0: negative samples of sample i 
 
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)

        # generate a new label y
        # compute ranking hinge loss 
        y = dist_an.data.new() 
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        # y_i = 1, means dist_an > dist_ap + margin will casuse loss be zero 
        loss = self.ranking_loss(dist_an, dist_ap, y)

        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0) # normalize data by batch size 
        return loss, prec    

# for all classes  
class TripletCenter40LossAllClass(nn.Module):
    def __init__(self, margin=0):
        super(TripletCenter40LossAllClass, self).__init__()
        self.margin = margin
        self.ranking_loss_center = nn.MarginRankingLoss(margin=self.margin)
        self.centers = nn.Parameter(torch.randn(40, 40)) # for modelent40
        # self.centers = nn.Parameter(torch.randn(40, 40)) # for shapenet55

    def forward(self, inputs, targets):
        n = inputs.size(0)
        m = self.centers.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, m) + \
            torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(m, n).t()
        dist.addmm_(1, -2, inputs, self.centers.t())
        dist = dist.clamp(min=1e-12).sqrt()

        # for each anchor, find the hardest positive and negative
        mask = torch.zeros(dist.size()).byte().cuda()
        for i in range(n):
            mask[i][targets[i].data] = 1

        # mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []

        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())  # hardest positive center
            dist_an.append(dist[i][mask[i] == 0].min())  # hardest negative center

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # generate a new label y
        # compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        # y_i = 1, means dist_an > dist_ap + margin will casuse loss be zero
        loss = self.ranking_loss_center(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)

        # normalize data by batch size
        return loss, prec
