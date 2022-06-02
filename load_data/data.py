# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from scipy.sparse import csgraph

import pandas as pd
import numpy as np
import torch
import os

import timeit


def prepare_data(fin, with_labels=True, normalize=False, n_pca=0):
	"""
	Reads a dataset in CSV format from the ones in datasets/
	"""
	df = pd.read_csv(fin + '.csv', sep=',')
	n = len(df.columns)

	if with_labels:
		x = np.double(df.values[:, 0:n - 1])
		labels = df.values[:, (n - 1)]
		labels = labels.astype(str)
		colnames = df.columns[0:n - 1]
	else:
		x = np.double(df.values)
		labels = ['unknown'] * np.size(x, 0)
		colnames = df.columns

	n = len(colnames)

	idx = np.where(np.std(x, axis=0) != 0)[0]
	x = x[:, idx]

	if normalize:
		s = np.std(x, axis=0)
		s[s == 0] = 1
		x = (x - np.mean(x, axis=0)) / s

	if n_pca:
		if n_pca == 1:
			n_pca = n

		nc = min(n_pca, n)
		pca = PCA(n_components=nc)
		x = pca.fit_transform(x)

	labels = np.array([str(s) for s in labels])

	return x, labels

