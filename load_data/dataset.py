from copyreg import pickle
from re import S
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math as m
import scanpy as sc
import anndata
import torch
import load_data.source as Source
from torch.nn import functional as F
from scipy.io import mmread
import scanpy.external as sce
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from Bio import SeqIO
import itertools
import pickle
from sklearn.feature_extraction.text import CountVectorizer


def read_mtx(filename, dtype='int32'):
    x = mmread(filename).astype(dtype)
    return x

def multi_one_hot(index_tensor, depth_list):
    one_hot_tensor = F.one_hot(index_tensor[:,0], num_classes=depth_list[0])
    for col in range(1, len(depth_list)):
        next_one_hot = F.one_hot(index_tensor[:,col], num_classes=depth_list[col])
        one_hot_tensor = torch.cat([one_hot_tensor, next_one_hot], 1)

    return one_hot_tensor

class CD14DataModule(Source.Source):
    
    def _LoadData(self, args):
        print('load CD14DataModule')

        # mtx = './data/cd14_monocyte_erythrocyte.mtx'
        # data = read_mtx(mtx)
        # data = data.transpose().todense()
        # sadata = anndata.AnnData(X=data)
        # sc.pp.normalize_per_cell(sadata, counts_per_cell_after=1e4)
        # sadata = sc.pp.log1p(sadata, copy=True)
        # sc.tl.pca(sadata, n_comps=50)
        # sadata.write('./data/cd14_monocyte_erythrocyte.h5ad')

        sadata = sc.read('./data/cd14_monocyte_erythrocyte.h5ad')
        data = sadata.obsm['X_pca'].copy()
        
        batch_all = np.zeros((data.shape[0], 1)) * -1
        n_batch = [1]
        batch_hot = batch_all
        len_n_batch = len(n_batch)
        len_batch = sum([n_batch[i] for i in range(len_n_batch)])

        label_celltype = pd.read_csv('data/cd14_monocyte_erythrocyte_celltype.tsv', sep='\t', header=None).values
        sadata.obs['celltype'] = label_celltype
        label_id = list(np.squeeze(label_celltype))
        label_id_set = list(set(label_id))
        label_id = np.array([label_id_set.index(i) for i in label_id])

        self.len_n_batch = len_n_batch
        self.len_batch = len_batch
        self.batch_hot = torch.tensor(batch_hot).long()
        self.batch_all = torch.tensor(batch_all)
        self.n_batch = n_batch
        if args['batch_invariant'] == 'batch_invariant':
            self.data = torch.cat((torch.tensor(data), self.batch_hot), 1)
        else:
            self.data = torch.tensor(data)
        self.sadata = sadata
        self.label = torch.tensor(label_id)
        self.label_str = [np.array(label_celltype)]
        self.inputdim = self.data[0].shape
        self.same_sigma = False
        print('shape = ', self.data.shape)

class UCSTROMALCORRECTIONDataModule(Source.Source):

    def _LoadData(self, args):
        print('load UCSTROMALCORRECTIONDataModule')

        # mtx = './data/uc_stromal.mtx'
        # data = read_mtx(mtx)
        # data = data.transpose().todense()
        # sadata = anndata.AnnData(X=data)
        # sc.pp.normalize_per_cell(sadata, counts_per_cell_after=1e4)
        # sadata = sc.pp.log1p(sadata, copy=True)
        # sc.tl.pca(sadata, n_comps=50)

        # batch_p = np.array(pd.read_csv('data/uc_stromal_batch_patient.tsv', header=None))
        # batch_h = np.array(pd.read_csv('data/uc_stromal_batch_health.tsv', header=None))
        # sadata.obs['batch_p'] = batch_p
        # sadata.obs['batch_h'] = batch_h

        # label_celltype = pd.read_csv('data/uc_stromal_celltype.tsv', sep='\t', header=None).values
        # label_id = list(np.squeeze(label_celltype))
        # label_id_set = list(set(label_id))
        # label_id = np.array([label_id_set.index(i) for i in label_id])
        # sadata.obs['celltype'] = label_celltype
        # sadata.obs['celltype_id'] = label_id

        # sadata.write('./data/uc_stromal_batch_correction.h5ad')

        sadata = sc.read('./data/uc_stromal_batch_correction.h5ad')
        data = sadata.obsm['X_pca'].copy()

        batch_p = sadata.obs['batch_p']
        batch_h = sadata.obs['batch_h']
        batch_all = np.array(pd.concat([batch_p, batch_h], axis=1))
        n_batch = [30, 3]
        batch_hot = multi_one_hot(torch.tensor(batch_all), n_batch)
        len_n_batch = len(n_batch)
        len_batch = sum([n_batch[i] for i in range(len_n_batch)])

        self.len_n_batch = len_n_batch
        self.len_batch = len_batch
        self.batch_hot = torch.tensor(batch_hot).long()
        self.batch_all = torch.tensor(batch_all)
        self.n_batch = n_batch
        self.data = torch.tensor(data)
        self.sadata = sadata
        self.label = torch.tensor(sadata.obs['celltype_id'])
        self.label_str = [np.array(sadata.obs['celltype'])]
        self.inputdim = self.data[0].shape
        self.same_sigma = False
        print('shape = ', self.data.shape)