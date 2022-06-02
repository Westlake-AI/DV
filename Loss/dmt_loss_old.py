# a pytorch based lisv2 code

from multiprocessing import Pool

import numpy as np
import torch
import torch.autograd
import torch.nn.functional as F
from scipy import optimize
from torch import nn
from torch.autograd import Variable
from torch.functional import split
from torch.nn.modules import loss
from typing import Any
import scipy
import torch.nn.functional as F
import manifolds.hyperboloid as hyperboloid
import manifolds.poincare as poincare

from Loss.dmt_loss_source import Source

class MyLoss(nn.Module):
    def __init__(
        self,
        v_input,
        SimilarityFunc,
        metric_s = "euclidean",
        metric_e = "euclidean",
        c_input = 1,
        c_latent = 1,
        eta = 1,
        near_bound = 0,
        far_bound = 1,
        pow = 2,
    ):
        super(MyLoss, self).__init__()

        self.v_input = v_input
        self.gamma_input = self._CalGamma(v_input)
        self.ITEM_loss = self._TwowaydivergenceLoss
        self._Similarity = SimilarityFunc
        self.metric_s = metric_s
        self.metric_e = metric_e
        self.c_input = c_input
        self.c_latent = c_latent
        self.eta = eta
        self.near_bound = near_bound
        self.far_bound = far_bound
        self.pow = pow
        

    def forward(self, input_data, latent_data, rho, sigma, v_latent, ):
        
        dis_P = self._DistanceSquared(input_data, metric=self.metric_s, c=self.c_input)
        P = self._Similarity(
                dist=dis_P,
                rho=rho,
                sigma_array=sigma,
                gamma=self.gamma_input,
                v=self.v_input)
        
        dis_Q = self._DistanceSquared(latent_data, metric=self.metric_e, c=self.c_latent)
        Q = self._Similarity(
                dist=dis_Q,
                rho=0,
                sigma_array=1,
                gamma=self._CalGamma(v_latent),
                v=v_latent,
                pow=self.pow
                )
        
        loss_ce = self.ITEM_loss(P_=P,Q_=Q,)
        return loss_ce, dis_P, dis_Q, P, Q

    def ForwardInfo(self, input_data, latent_data, rho, sigma, v_latent, ):
        
        dis_P = self._DistanceSquared(input_data, metric=self.metric_s, c=self.c_input)
        P = self._Similarity(
                dist=dis_P,
                rho=rho,
                sigma_array=sigma,
                gamma=self.gamma_input,
                v=self.v_input)
        
        dis_Q = self._DistanceSquared(latent_data, metric=self.metric_e, c=self.c_latent)
        Q = self._Similarity(
                dist=dis_Q,
                rho=0,
                sigma_array=1,
                gamma=self._CalGamma(v_latent),
                v=v_latent,
                pow=self.pow)
        
        loss_ce = self.ITEM_loss(P_=P,Q_=Q,)
        return loss_ce.detach().cpu().numpy(), dis_P.detach().cpu().numpy(), dis_Q.detach().cpu().numpy(), P.detach().cpu().numpy(), Q.detach().cpu().numpy()

    def _TwowaydivergenceLoss(self, P_, Q_):

        EPS = 1e-12
        losssum1 = (P_ * torch.log(Q_ + EPS))
        losssum2 = ((1-P_) * torch.log(1-Q_ + EPS))
        losssum = -1*(losssum1 + losssum2).mean()

        return losssum

    def _L2Loss(self, P, Q):

        losssum = torch.norm(P-Q, p=2)/P.shape[0]
        return losssum
    
    def _L3Loss(self, P, Q):

        losssum = torch.norm(P-Q, p=3)/P.shape[0]
        return losssum
    
    def _DistanceSquared(
        self,
        x,
        c,
        metric='euclidean'
    ):
        if metric == 'euclidean':
            m, n = x.size(0), x.size(0)
            xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
            yy = xx.t()
            dist = xx + yy
            dist = torch.addmm(dist, mat1=x, mat2=x.t(),beta=1, alpha=-2)
            dist = dist.clamp(min=1e-22)

        if metric == 'cosine':
            dist = torch.mm(x, x.transpose(0, 1))
            dist = 1 - dist
            dist = dist.clamp(min=1e-22)

        if metric == 'poin_dist_mobiusm_v1':
            PoincareBall = poincare.PoincareBall()
            dist = PoincareBall.sqdist_xu_mobius_v1(x, x, c=c)
            dist = dist.clamp(min=1e-22)

        if metric == 'poin_dist_mobiusm_v2':
            PoincareBall = poincare.PoincareBall()
            dist = PoincareBall.sqdist_xu_mobius_v2(x, x, c=c)
            dist = dist.clamp(min=1e-22)

        if metric == 'poin_dist_v1':
            PoincareBall = poincare.PoincareBall()
            dist = PoincareBall.sqdist_xu_v1(x, x)
            dist = dist.clamp(min=1e-22)

        if metric == 'poin_dist_v2':
            PoincareBall = poincare.PoincareBall()
            dist = PoincareBall.sqdist_xu_v2(x, x)
            dist = dist.clamp(min=1e-22)

        if metric == 'lor_dist_v1':
            Hyperboloid = hyperboloid.Hyperboloid()
            dist = Hyperboloid.sqdist_xu_v1(x, x, c=c)
            dist = dist.clamp(min=1e-22)

        if metric == 'lor_dist_v2':
            Hyperboloid = hyperboloid.Hyperboloid()
            dist = Hyperboloid.sqdist_xu_v2(x, x, c=c)
            dist = dist.clamp(min=1e-22)

        dist[torch.eye(dist.shape[0]) == 1] = 1e-22

        return dist

    def _CalGamma(self, v):
        
        a = scipy.special.gamma((v + 1) / 2)
        b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
        out = a / b

        return out

