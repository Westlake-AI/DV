from threading import enumerate
import joblib
import torch
import torchvision.datasets as datasets
from sklearn.metrics import pairwise_distances
import numpy as np
from load_data.sigma import PoolRunner
import scipy
from sklearn.preprocessing import StandardScaler
from pynndescent import NNDescent
import os
import manifolds.hyperboloid as hyperboloid
import manifolds.poincare as poincare
from manifolds.hyperbolic_project import ToEuclidean, ToSphere, ToPoincare, ToLorentz


class Source(torch.utils.data.Dataset):
    
    def __init__(self, DistanceF, SimilarityF, SimilarityNPF, jumpPretreatment=False, **kwargs):
        self.args = kwargs
        self.DistanceF = DistanceF
        self.SimilarityF = SimilarityF
        self.SimilarityNPF = SimilarityNPF
        self.train = True
        self.smooth_sigma = True

        if self.args['pro_s'] == 'eu':
            self.rie_pro_input = ToEuclidean()
        if self.args['pro_s'] == 'sphere':
            self.rie_pro_input = ToSphere()
        if self.args['pro_s'] == 'poin':
            self.rie_pro_input = ToPoincare(c=self.args['c_input'], manifold=self.args['manifold'])
        if self.args['pro_s'] == 'lor':
            self.rie_pro_input = ToLorentz(c=self.args['c_input'], manifold=self.args['manifold'])

        self._LoadData(self.args)
        if not jumpPretreatment:
            filename = 'data_name{}same_sigma{}perplexity{}v_input{}metric{}pow_input{}n_point{}'.format(
                    self.args['data_name'], 
                    self.args['same_sigma'], 
                    self.args['perplexity'], 
                    self.args['v_input'], 
                    self.args['metric_s'], 
                    self.args['pow_input'], 
                    self.args['n_point'], 
                    )

            self._Pretreatment()
            joblib.dump(
                value=[self.sigma, self.rho, self.inputdim], 
                filename='save/'+filename
                )

    def _LoadData(self, ):
        pass
    
    def _Pretreatment(self, ):
        
        if self.args['metric_s'] ==  'cosine' or self.args['metric_s'] ==  'poin_dist_mobiusm_v2' or self.args['metric_s'] == 'poin_dist_v2' or self.args['metric_s'] == 'lor_dist_v2':
            rho, sigma = self._initPairwise(
                self.rie_pro_input(self.data),
                perplexity=self.args['perplexity'],
                v_input=self.args['v_input'])
        else:
            rho, sigma = self._initKNN(
                self.rie_pro_input(self.data),
                perplexity=self.args['perplexity'],
                v_input=self.args['v_input']
                )
            
        self.sigma = sigma
        self.rho = rho
        self.inputdim = self.data[0].shape

    def _DistanceSquared(
        self,
        x,
        metric='euclidean'
    ):
        if metric == 'euclidean_':
            m, n = x.size(0), x.size(0)
            xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
            yy = xx.t()
            dist = xx + yy
            dist = torch.addmm(dist, mat1=x, mat2=x.t(),beta=1, alpha=-2)

        if metric == 'euclidean':
            dist = np.power(pairwise_distances(x.reshape((x.shape[0], -1)), n_jobs=-1, metric=metric), 2)

        if metric == 'cosine':
            dist = torch.mm(x, x.transpose(0, 1))
            print(dist)
            dist = 1 - dist

        if metric == 'poin_dist_mobiusm_v1':
            PoincareBall = poincare.PoincareBall()
            dist = PoincareBall.sqdist_xu_mobius_v1(x, x, c=self.args['c_input'])

        if metric == 'poin_dist_mobiusm_v2':
            PoincareBall = poincare.PoincareBall()
            dist = PoincareBall.sqdist_xu_mobius_v2(x, x, c=self.args['c_input'])

        if metric == 'poin_dist_v1':
            PoincareBall = poincare.PoincareBall()
            dist = PoincareBall.sqdist_xu_v1(x, x)

        if metric == 'poin_dist_v2':
            PoincareBall = poincare.PoincareBall()
            dist = PoincareBall.sqdist_xu_v2(x, x)

        if metric == 'lor_dist_v1':
            Hyperboloid = hyperboloid.Hyperboloid()
            dist = Hyperboloid.sqdist_xu_v1(x, x, c=self.args['c_input'])

        if metric == 'lor_dist_v2':
            Hyperboloid = hyperboloid.Hyperboloid()
            dist = Hyperboloid.sqdist_xu_v2(x, x, c=self.args['c_input'])

        return dist

    def _initPairwise(self, X, perplexity, v_input):
        
        print('use pairwise method to find the sigma')

        dist = self._DistanceSquared(X.reshape((X.shape[0], -1)), metric=self.args['metric_s']).numpy()
        Dist = dist
        Dist[torch.eye(Dist.shape[0]) == 1] = 1e22
        self.neighbors_index = np.argsort(Dist)

        if self.args['model_type'] == 'dsml' or self.args['model_type'] == 'dshl':
            rho = np.zeros(dist.shape[0])
            sigma = np.ones(dist.shape[0])
        else:
            rho = self._CalRho(dist)
            r = PoolRunner(
                similarity_function_nunpy=self.SimilarityNPF,
                number_point = X.shape[0],
                perplexity=perplexity,
                dist=dist,
                rho=rho,
                gamma=self._CalGamma(v_input),
                v=v_input,
                pow=self.args['pow_input'],
                )
            sigma = np.array(r.Getout())
            print("\nMean sigma = " + str(np.mean(sigma)))

            std_dis = np.std(rho) / np.sqrt(X.shape[1])
            print('std_dis', std_dis)
            if std_dis < 0.20 or self.args['same_sigma'] is True:
                sigma[:] = sigma.mean()
        
        return rho, sigma

    def _initKNN(self, X, perplexity, v_input):
        
        print('use kNN method to find the sigma')

        X_rshaped = X.reshape((X.shape[0],-1))
        index = NNDescent(X_rshaped, n_jobs=-1, metric=self.args['metric_s'])
        self.neighbors_index, neighbors_dist = index.query(X_rshaped, k=self.args['K'] )
        neighbors_dist = np.power(neighbors_dist, 2)

        if self.args['model_type'] == 'dsml' or self.args['model_type'] == 'dshl':
            rho = np.zeros(neighbors_dist.shape[0])
            sigma = np.ones(neighbors_dist.shape[0])
        else:
            rho = neighbors_dist[:, 1]
            r = PoolRunner(
                similarity_function_nunpy=self.SimilarityNPF,
                number_point = X.shape[0],
                perplexity=perplexity,
                dist=neighbors_dist,
                rho=rho,
                gamma=self._CalGamma(v_input),
                v=v_input,
                pow=self.args['pow_input'],
                )
            sigma = np.array(r.Getout())
            print("\nMean sigma = " + str(np.mean(sigma)))

            std_dis = np.std(rho) / np.sqrt(X.shape[1])
            print('std_dis', std_dis)
            if std_dis < 0.20 or self.args['same_sigma'] is True:
                sigma[:] = sigma.mean()
            if self.smooth_sigma is True:
                new_sigma = []
                for i in range(sigma.shape[0]):
                    new_sigma.append(
                        np.mean(sigma[self.neighbors_index[i, : int(self.args['perplexity']) ]])
                    )
                sigma = np.array(new_sigma)
        
        return rho, sigma

    def _CalRho(self, dist):
        dist_copy = np.copy(dist)
        row, col = np.diag_indices_from(dist_copy)
        dist_copy[row,col] = 1e16
        rho = np.min(dist_copy, axis=1)
        return rho
    
    def _CalGamma(self, v):
        a = scipy.special.gamma((v + 1) / 2)
        b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
        out = a / b
        return out
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        batch_all_item = self.batch_all[index]
        batch_item = self.batch_hot[index]
            
        if self.args['model_type'] == 'dsml' or self.args['model_type'] == 'dshl':
            return batch_item, batch_all_item, index
        else:
            data_item = self.data[index]
            rho_item = self.rho[index]
            sigma_item = self.sigma[index]
            label_item = self.label[index]
            return data_item, rho_item, sigma_item, label_item, batch_item, batch_all_item, index
