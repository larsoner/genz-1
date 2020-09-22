from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Dimension reduction and clustering libraries
import umap
import hdbscan
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

sns.set(style='white', rc={'figure.figsize': (10, 8)})

mnist = fetch_openml('Fashion-MNIST', version=1)
mnist.target = mnist.target.astype(int)
standard_embedding = umap.UMAP(random_state=42).fit_transform(mnist.data)
plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1],
            c=mnist.target.astype(int), s=0.1, cmap='Spectral')


import numpy as np
import pandas as pd
import time# For plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
% matplotlib inline
#PCA
from sklearn.decomposition import PCA
#TSNE
from sklearn.manifold import TSNE
#UMAP
import umap
train = pd.read_csv(
    '/Users/ktavabi/Github/Projects/badbaby/badbaby/payload/cdi-meg_55_lbfgs_tidy_07072020.csv')
# Setting the label and the feature columns
train['c'] = pd.Series(train.condition.map(
    {'aspirative': 1, 'plosive': 2, 'mmn': 3, 'deviant': 4}))
y = train['AUC_zscore'].values
x = train[['ses', 'age', 'headSize', 'birthWeight', 'c']].values
print(np.unique(y))

# PCA
start = time.time()
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
print('Duration: {} seconds'.format(time.time() - start))
plt.style.use('dark_background')
plt.scatter(principalComponents[:, 0],
            principalComponents[:, 1], c=y, cmap='viridis')
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(df['AUC_zscore'].max())
             ).set_ticks(np.arange(df['AUC_zscore'].max()))
plt.title('Responset', fontsize=24)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')


start = time.time()
reducer = umap.UMAP(random_state=42, n_components=2)
embedding = reducer.fit_transform(x)
print('Duration: {} seconds'.format(time.time() - start))

# Visualising UMAP in 2d
fig = plt.figure(figsize=(12, 8))
plt.scatter(reducer.embedding_[:, 0], reducer.embedding_[
            :, 1], c=y, cmap='gist_rainbow')
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(y.max())
             ).set_ticks(np.arange(y.max()))
plt.title('Response', fontsize=24)
plt.xlabel('umap 1')
plt.ylabel('umap 2')
