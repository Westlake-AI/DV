from torch import optim
import os
from re import L, X
import pandas

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import pytorch_lightning as pl
import numpy as np
import functools
from torch.optim.lr_scheduler import StepLR
from matplotlib.projections import register_projection

import Similarity.sim_Gaussian_1 as Sim_use
import Loss.dmt_loss_old as Loss_use
import Loss.dmt_loss_aug as Loss_use_aug
import nuscheduler
import plotly.express as px
import matplotlib.pylab as plt
import pandas as pd

import load_data.dataset as datasetfunc
import load_disF.disfunc as disfunc
import load_simF.simfunc as simfunc
import sys
import git
import scanpy as sc
import manifolds
from manifolds.hyperbolic_project import ToEuclidean, ToSphere, ToPoincare, FromPoincare, ToLorentz

torch.set_num_threads(1)


class Generator(nn.Module):
    def __init__(self, dims, num_batch):
        super().__init__()

        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.1))
            return layers

        self.model = nn.Sequential(
            *block(int(np.array(dims)), 500),
            *block(500, 300),
            *block(300, 100),
        )

    def forward(self, x):
        x = self.model(x)
        return x

class Classifier(nn.Module):
    def __init__(self, num_latent_dim):
        super().__init__()

        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.1))
            return layers

        self.model = nn.Sequential(
            *block(100, 300),
            *block(300, 100),
            nn.Linear(100, num_latent_dim),
        )

    def forward(self, x):
        x = self.model(x)
        
        return x


class DMT_Model(pl.LightningModule):
    
    def __init__(
        self,
        dataset,
        DistanceF,
        SimilarityF,
        data_dir='./data',
        **kwargs,
        ):

        super().__init__()
        self.save_hyperparameters()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.learning_rate = self.hparams.lr
        self.train_batchsize = self.hparams.batch_size
        self.test_batchsize = self.hparams.batch_size
        self.num_latent_dim = 2
        self.dim_plot = '2d'
        if self.hparams.pro_e == 'sphere' or self.hparams.pro_e == 'lor':
            self.num_latent_dim += 1
            self.hparams.NetworkStructure[-1] = self.num_latent_dim
            if self.hparams.pro_e == 'sphere':
                self.dim_plot = 'equal_earth'

        self.dims = dataset.data[0].shape
        self.log_interval = self.hparams.log_interval
        self.c_input = self.hparams.c_input
        self.c_latent = self.hparams.c_latent
        self.manifold = self.hparams.manifold
        self.act = self.hparams.act
        self.dropout = 0.0
        self.weight_decay = 5e-4

        self.train_dataset = dataset
        self.DistanceF = DistanceF
        self.SimilarityF = SimilarityF
        
        self.label = dataset.label
        self.data = dataset.data
        self.labelstr = dataset.label_str
        self.sadata = dataset.sadata
        self.n_batch = dataset.n_batch
        self.len_n_batch = dataset.len_n_batch
        self.len_batch = dataset.len_batch
        self.batch_hot = dataset.batch_hot

        self.generator = Generator(dims=self.dims, num_batch=self.len_batch)
        self.classifier = Classifier(num_latent_dim=self.num_latent_dim)

        if self.hparams.pro_s == 'eu':
            self.rie_pro_input = ToEuclidean()
        if self.hparams.pro_s == 'sphere':
            self.rie_pro_input = ToSphere()
        if self.hparams.pro_s == 'poin':
            self.rie_pro_input = ToPoincare(c=self.c_input, manifold=self.manifold)
        if self.hparams.pro_s == 'lor':
            self.rie_pro_input = ToLorentz(c=self.c_input, manifold=self.manifold)

        if self.hparams.pro_e == 'eu':
            self.rie_pro_latent = ToEuclidean()
        if self.hparams.pro_e == 'sphere':
            self.rie_pro_latent = ToSphere()
        if self.hparams.pro_e == 'poin':
            self.rie_pro_latent = ToPoincare(c=self.c_latent, manifold=self.manifold)
        if self.hparams.pro_e == 'lor':
            self.rie_pro_latent = ToLorentz(c=self.c_latent, manifold=self.manifold)

        if self.hparams.method == 'dmt':
            self.Loss = Loss_use.MyLoss(
                v_input=self.hparams.v_input,
                SimilarityFunc=SimilarityF,
                metric_s=self.hparams.metric_s,
                metric_e=self.hparams.metric_e,
                c_input = self.hparams.c_input,
                c_latent = self.hparams.c_latent,
                pow = self.hparams.pow_latent
                )
        
        if self.hparams.method == 'dmt_aug':
            self.Loss = Loss_use_aug.MyLoss(
                v_input=self.hparams.v_input,
                SimilarityFunc=SimilarityF,
                metric_s=self.hparams.metric_s,
                metric_e=self.hparams.metric_e,
                c_input = self.hparams.c_input,
                c_latent = self.hparams.c_latent,
                pow = self.hparams.pow_latent,
                augNearRate=self.hparams.augNearRate,
                batchRate=self.hparams.batchRate,
                )
        
        self.nushadular = nuscheduler.Nushadular(
            nu_start=self.hparams.vs,
            nu_end=self.hparams.ve,
            epoch_start=self.hparams.epochs*2//6,
            epoch_end=self.hparams.epochs*5//6,
            )

        self.criterion = nn.MSELoss()

        self.data_name = self.hparams.data_name
        self.save_path = self.data_name + '/path_' + self.hparams.metric_s + '_' + self.hparams.metric_e + '_' + self.hparams.model_type \
                            + '_' + str(self.hparams.v_input) + '_' + str(self.hparams.vs) + '_' + str(self.hparams.ve) + '_' + str(self.hparams.K) \
                            + '_' + self.hparams.pro_s + '_' + self.hparams.pro_e + '_' + str(self.hparams.pow_input) + '_' + str(self.hparams.batch_size) \
                            + '_' + str(self.hparams.pow_latent) + '_' + self.hparams.optimizer + '_' + self.hparams.act + '_' + str(self.hparams.perplexity) \
                            + '_' + str(self.c_input) + '_' + str(self.c_latent)\
                            + '_' + str(self.hparams.augNearRate) + '_' + str(self.hparams.mid_batch_invariant) + '_' + str(self.hparams.latent_batch_invariant) \
                            + '_' + self.hparams.method + '_' + str(self.hparams.batch_invariant) + '_' + str(self.hparams.lr) + '_' + str(self.hparams.alpha) \
                            + '_' + str(self.hparams.beta) + '_' + str(self.hparams.batchRate) + '_500_300_100_batch_correction'

        if not os.path.exists('log_' + self.save_path):
            os.makedirs('log_' + self.save_path)
            os.makedirs('log_' + self.save_path + '/result')
            os.makedirs('log_' + self.save_path + '/figures')
        os.chdir(r'log_' + self.save_path)

        print(self.generator)
        print(self.classifier)

    def forward(self, x):
        return self.generator(x)

    def aug_near_mix(self, index, dataset, k=10):
        r = (torch.arange(start=0, end=index.shape[0])*k + torch.randint(low=1, high=k, size=(index.shape[0],)))
        random_select_near_index = dataset.neighbors_index[index.cpu()][:,:k].reshape((-1,))[r]
        random_select_near_data_2 = dataset.data[random_select_near_index]
        random_select_near_data_batch_hot_2 = dataset.batch_hot[random_select_near_index]
        random_rate = torch.rand(size = (index.shape[0], 1)) / 2
        new_data = random_rate*random_select_near_data_2 + (1-random_rate)*dataset.data[index.cpu()]
        new_batch_hot = random_rate*random_select_near_data_batch_hot_2 + (1-random_rate)*dataset.batch_hot[index.cpu()]
        return new_data.to(index.device), new_batch_hot.to(index.device)

    def false_batch_hot(self, data_1):
        for i in range(self.len_n_batch):
            batch_false = np.expand_dims(np.random.randint(self.n_batch[i], size=data_1.shape[0]), 1)
            if i == 0:
                batch_hot_false = torch.tensor(batch_false)
            else:
                batch_hot_false = torch.cat((torch.tensor(batch_hot_false), torch.tensor(batch_false)), dim=1)
        return batch_hot_false.to(data_1.device)

    def multi_one_hot(SELF, index_tensor, depth_list):
        one_hot_tensor = F.one_hot(index_tensor[:,0], num_classes=depth_list[0])
        for col in range(1, len(depth_list)):
            next_one_hot = F.one_hot(index_tensor[:,col], num_classes=depth_list[col])
            one_hot_tensor = torch.cat([one_hot_tensor, next_one_hot], 1)

        return one_hot_tensor

    def training_step(self, batch, batch_idx):

        batch_hot, batch_all, index = batch

        data_1 = self.data_train.data[index].to(index.device)
        data_2, batch_hot_2 = self.aug_near_mix(index, self.data_train, k=self.hparams.K)
        mid_1 = self(data_1)
        mid_2 = self(data_2)
        latent_1 = self.classifier(mid_1)
        latent_2 = self.classifier(mid_2)
        data = torch.cat([data_1, data_2])
        mid = torch.cat((mid_1, mid_2))
        latent = torch.cat([latent_1, latent_2])
        batch_hot = torch.cat((batch_hot, batch_hot_2))

        latent_pro = self.rie_pro_latent(latent)

        loss_gsp, self.dis_p, self.dis_q, self.P, self.Q = self.Loss(
            input_data=mid.reshape(data.shape[0], -1),
            latent_data=latent_pro.reshape(data.shape[0], -1),
            batch_hot = batch_hot,
            rho=0.0,
            sigma=1.0,
            v_latent=self.nushadular.Getnu(self.current_epoch),
        )
        loss = loss_gsp

        logs={
            'loss': loss,
            'nv': self.nushadular.Getnu(self.current_epoch),
            'lr': self.trainer.optimizers[0].param_groups[0]['lr'],
            'epoch': self.current_epoch,
        }

        return loss

    def validation_step(self, batch, batch_idx):

        batch_hot, batch_all, index = batch
        data = self.data_train.data[index].to(index.device)
        label = self.data_train.label[index].to(index.device)

        mid = self(data)
        latent = self.classifier(mid)
        latent = self.rie_pro_latent(latent)
        
        return (
            data.detach().cpu().numpy(),
            latent.detach().cpu().numpy(),
            label.detach().cpu().numpy(),
            index.detach().cpu().numpy(),
            )

    def log_dist(self, dist):
        self.dist = dist

    def validation_epoch_end(self, outputs):
        
        if (self.current_epoch+1) % self.log_interval == 0:

            self.current_nu = self.nushadular.vListForEpoch[self.current_epoch]
            self.title_color_1 = self.hparams.color_1 + ' Nu=' + str(self.current_nu)
            self.title_color_2 = self.hparams.color_2 + ' Nu=' + str(self.current_nu)
            
            data = np.concatenate([ data_item[0] for data_item in outputs ])
            latent = np.concatenate([ data_item[1] for data_item in outputs ])
            label = np.concatenate([ data_item[2] for data_item in outputs ])
            index = np.concatenate([ data_item[3] for data_item in outputs ])

            if (self.current_epoch+1) == self.hparams.epochs:
                np.save('result/latent_epoch_{}.npy'.format(self.current_epoch), latent)

            if self.hparams.pro_e == 'lor':
                latent = latent[:, 1:3] / np.expand_dims(1 + latent[:, 0], axis=1)

            if self.dim_plot == 'equal_earth':
                self.sadata.obsm['X_tsne'] = latent
                sc.settings.set_figure_params(dpi=600)
                sc.pl.tsne(self.sadata, title=self.title_color_1, size=30, color=self.hparams.color_1, projection='3d', legend_fontsize='xx-small', save='pbmc3k_{}_{}_sphere.png'.format(self.hparams.color_1, self.current_epoch))
                sc.pl.tsne(self.sadata, title=self.title_color_2, size=30, color=self.hparams.color_2, projection='3d', legend_fontsize='xx-small', save='pbmc3k_{}_{}_sphere.png'.format(self.hparams.color_2, self.current_epoch))

            self.sadata.obsm['X_tsne'] = latent
            sc.settings.set_figure_params(dpi=600)

            if len(latent) > 5000:
                sc.pl.tsne(self.sadata, title=self.title_color_1, color=self.hparams.color_1, projection=self.dim_plot, legend_fontsize='xx-small', save='{}_{}_{}.png'.format(self.data_name, self.hparams.color_1, self.current_epoch))
                sc.pl.tsne(self.sadata, title=self.title_color_2, color=self.hparams.color_2, projection=self.dim_plot, legend_fontsize='xx-small', save='{}_{}_{}.png'.format(self.data_name, self.hparams.color_2, self.current_epoch))
                if self.dim_plot == '2d':
                    sc.pl.tsne(self.sadata, title=self.title_color_1, color=self.hparams.color_1, projection=self.dim_plot, legend_fontsize='xx-small', legend_loc="on data", save='{}_{}_on_data_{}.png'.format(self.data_name, self.hparams.color_1, self.current_epoch))
                    sc.pl.tsne(self.sadata, title=self.title_color_2, color=self.hparams.color_2, projection=self.dim_plot, legend_fontsize='xx-small', legend_loc="on data", save='{}_{}_on_data_{}.png'.format(self.data_name, self.hparams.color_2, self.current_epoch))
            else:
                sc.pl.tsne(self.sadata, title=self.title_color_1, size=30, color=self.hparams.color_1, projection=self.dim_plot, legend_fontsize='xx-small', save='{}_{}_{}.png'.format(self.data_name, self.hparams.color_1, self.current_epoch))
                sc.pl.tsne(self.sadata, title=self.title_color_2, size=30, color=self.hparams.color_2, projection=self.dim_plot, legend_fontsize='xx-small', save='{}_{}_{}.png'.format(self.data_name, self.hparams.color_2, self.current_epoch))
                if self.dim_plot == '2d':
                    sc.pl.tsne(self.sadata, title=self.title_color_1, size=30, color=self.hparams.color_1, projection=self.dim_plot, legend_fontsize='xx-small', legend_loc="on data", save='{}_{}_on_data_{}.png'.format(self.data_name, self.hparams.color_1, self.current_epoch))
                    sc.pl.tsne(self.sadata, title=self.title_color_2, size=30, color=self.hparams.color_2, projection=self.dim_plot, legend_fontsize='xx-small', legend_loc="on data", save='{}_{}_on_data_{}.png'.format(self.data_name, self.hparams.color_2, self.current_epoch))

    def configure_optimizers(self):
        opt = torch.optim.AdamW([
            {'params': self.generator.parameters()},
            {'params': self.classifier.parameters()}
        ], lr=self.learning_rate, weight_decay=1e-4)
        scheduler = StepLR(opt, step_size=self.hparams.epochs//10, gamma=0.5)

        return [opt], [scheduler]

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        if stage == 'fit' or stage is None:
            self.data_train = self.train_dataset
            self.data_val = self.train_dataset

            self.train_da = DataLoader(
                self.data_train,
                shuffle=True,
                batch_size=self.train_batchsize,
                num_workers=1,
                persistent_workers=True,
            )
            self.test_da = DataLoader(
                self.data_val,
                batch_size=self.train_batchsize,
                num_workers=1,
                persistent_workers=True,
            )

        if stage == 'test' or stage is None:
            self.data_test = self.train_dataset

    def train_dataloader(self):
        return self.train_da

    def val_dataloader(self):
        return self.test_da

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.test_batchsize)

def main(args):

    pl.utilities.seed.seed_everything(args.__dict__['seed'])
    info = [str(s) for s in sys.argv[1:]]
    runname = '_'.join(['dmt', args.data_name, ''.join(info)])
    
    disfunc_use = getattr(disfunc, 'EuclideanDistanceNumpy')
    simfunc_use = getattr(simfunc, 'UMAPSimilarity')
    simfunc_npuse = getattr(simfunc, 'UMAPSimilarityNumpy')
    dm_class = getattr(datasetfunc, args.__dict__['data_name'] + 'DataModule')

    dataset = dm_class(
        DistanceF=disfunc_use,
        SimilarityF=simfunc_use,
        SimilarityNPF=simfunc_npuse,
        **args.__dict__,
        )

    model = DMT_Model(
        DistanceF=disfunc_use,
        SimilarityF=simfunc_use,
        dataset=dataset,
        **args.__dict__,
        )

    trainer = pl.Trainer.from_argparse_args(
        args=args,
        gpus=1, 
        max_epochs=args.epochs, 
        progress_bar_refresh_rate=10,
        logger=False,
        checkpoint_callback=False,
        gradient_clip_val=args.gradient_clip
        )

    trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='digits_T', )

    # data set param
    parser.add_argument('--data_name', type=str, default='CD14', choices=['CD14', 'UCSTROMALCORRECTION'])
    parser.add_argument('--n_point', type=int, default=60000000, )

    # model param
    parser.add_argument('--v_input', type=float, default=100)
    parser.add_argument('--same_sigma', type=bool, default=False)
    parser.add_argument('--show_detail', type=bool, default=False)
    parser.add_argument('--perplexity', type=int, default=20)
    parser.add_argument('--vs', type=float, default=5e-3)
    parser.add_argument('--ve', type=float, default=-1) # ve=-1 == nu=1e-3
    parser.add_argument('--eta', type=float, default=0)
    parser.add_argument('--NetworkStructure', type=list, default=[-1, 500, 300, 100, 2], help="[-1, 1000, 500, 300, 2]")
    parser.add_argument('--pow_input', type=float, default=2)
    parser.add_argument('--pow_latent', type=float, default=2)
    parser.add_argument('--method', type=str, default='dmt_aug', choices=['dmt', 'dmt_mask', 'dmt_aug'])
    parser.add_argument('--near_bound', type=float, default=0.0)
    parser.add_argument('--far_bound', type=float, default=1.0)

    # aug param
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--augNearRate', type=float, default=1000)

    # batch_invariant param
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--beta', type=float, default=0) 
    parser.add_argument('--batchRate', type=float, default=100)
    parser.add_argument('--batch_invariant', type=str, default='no', help="batch_invariant, no")
    parser.add_argument('--mid_batch_invariant', type=str, default='no', help="mid, no")
    parser.add_argument('--latent_batch_invariant', type=str, default='no', help="latent, no")

    # hypersphere/hyperbolic param
    parser.add_argument('--model_type', type=str, default='dsml')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--metric_s', type=str, default="euclidean", )
    parser.add_argument('--metric_e', type=str, default="euclidean", help="euclidean, cosine, poin_dist_v1, poin_dist_v2, poin_dist_mobiusm_v1, poin_dist_mobiusm_v2, lor_dist_v1, lor_dist_v2")
    parser.add_argument('--pro_s', type=str, default='eu', help="whether to projrction in input ['eu', 'sphere', 'poin', 'lor']")
    parser.add_argument('--pro_e', type=str, default='eu', help="whether to projrction in output ['eu', 'sphere', 'poin', 'lor']")
    parser.add_argument('--act', type=str, default='leaky_relu', help="which activation function to use [relu, leaky_relu] (or None for no activation)")
    parser.add_argument("--c_input", type=float, default=1.0, help="hyperbolic radius, set to None for trainable curvature")
    parser.add_argument("--c_latent", type=float, default=1.0, help="hyperbolic radius, set to None for trainable curvature")
    parser.add_argument('--manifold', type=str, default="Euclidean", help="which manifold to use, can be any of [Euclidean, Sphere, PoincareBall, Hyperboloid]")
    parser.add_argument('--bias', type=int, default=1)
    parser.add_argument('--gradient_clip', type=float, default=0)
    parser.add_argument('--Gauss_rescale', type=float, default=1)

    # train param
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR')
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--computer', type=str, default=os.popen('git config user.name').read()[:-1])

    # plot param
    parser.add_argument('--color_1', type=str, default='celltype', help="label [celltype, time, cluster]")
    parser.add_argument('--color_2', type=str, default='celltype', help="label [celltype, time, cluster]")

    args = pl.Trainer.add_argparse_args(parser)
    args = args.parse_args()
    
    main(args)

