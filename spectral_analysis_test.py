from sklearn.datasets import load_breast_cancer
from sklearn.cluster import spectral_clustering
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA

import numpy as np

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt


# First things first, load data and perform PCA on it.
dat = load_breast_cancer()

y = dat['target']
X = dat['data']
labels = dat['target_names']

# PCA
pca = PCA(n_components=6)
pca.fit(X)
features = pca.transform(X)

# Find out how much is explained by the the components
print('explained variance in first 6 principle components: ',pca.explained_variance_ratio_)


# Next, going to perform spectral analysis on the data

# Generate the a-symmetric connection matrix with k-nearest neighbor, n = 9 seems to produce a solid graph
connectivity = kneighbors_graph(X, n_neighbors=9, include_self=False)

# Make the connectivity graph symmetric
connectivity = 0.5 * (connectivity + connectivity.T)

# Apply the spectral clustering from Sci-kit learn, n = 2 for malignant and benign
sr = spectral_clustering(affinity=connectivity, n_clusters=2).astype(int)

# try to give a label the spectral clustering (it's an unsupervised method).
# Still uncertain how to automatically associate unsupervised data with labels. That's probably semi-supervised?
# Regardless, it seems the first data point is a pretty "standard" malignant.
# Because of that I'm going to assume that if the spectral clustering puts it into "1", then it probably means 0 in the
# context of the labels.
#
# There has to be a better way to automatically label things though
if sr[0] == 1:
    sr = -(sr-1)

# Axes3D seems to have the best legend if you plot the data sets separately, also confusion matrices are nice
TP, FP, TN, FN = [], [], [], []
for i in range(len(y)):
    if y[i] == 0 and y[i] == sr[i]:
        TP.append(features[i, :])
    if y[i] == 0 and y[i] != sr[i]:
        FP.append(features[i, :])
    if y[i] == 1 and y[i] == sr[i]:
        TN.append(features[i, :])
    if y[i] == 1 and y[i] != sr[i]:
        FN.append(features[i, :])

# Convert back to numpy array
TP = np.array(TP)
FP = np.array(FP)
TN = np.array(TN)
FN = np.array(FN)

plot = plt.figure()
ax = Axes3D(plot)

# Not going to fiddle too much with this, sometimes the items are listed backwards
ax.scatter(TP.T[:][0], TP.T[:][1], TP.T[:][2], label='Malignant: True Positive')
ax.scatter(FP.T[:][0], FP.T[:][1], FP.T[:][2], label='Malignant: False Positive')
ax.scatter(TN.T[:][0], TN.T[:][1], TN.T[:][2], label='Benign: True Negative')
ax.scatter(FN.T[:][0], FN.T[:][1], FN.T[:][2], label='Benign: False Negative')

ax.legend()
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')

plt.show()

# This was a fun little project, it's interesting that the malignant cells go hay-wire in regards to
# to several of the principle components, and that the benign cells for the most part "stay similar".
#
