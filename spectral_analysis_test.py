from sklearn.datasets import load_breast_cancer
from sklearn.cluster import spectral_clustering
from sklearn.neighbors import kneighbors_graph

import matplotlib.pyplot as plt

dat = load_breast_cancer()

y = dat['target']
X = dat['data']

# Okay, so gotta make a connectivity matrix for X.
# Connectivity matrix: 1 if nodes i,j are connected, 0 if not.
# Diagonals at position i,i are the sum of all the connections for node i.

# Generate the connections for each node: (Non-symmetric)
connectivity = kneighbors_graph(X, n_neighbors=9, include_self=False)

# Make the connectivity graph symmetric
connectivity = 0.5 * (connectivity + connectivity.T)

# Normally connectivity matrices need the diagonals to have the negative sum for their diagonal
# However, sklearn's spectral clustering doesn't seem to need that.

bc_spectral_results = spectral_clustering(affinity=connectivity, n_clusters=3).astype(int)

colorkey=['r','b','g']

#Not going to fiddle too much with this
plt.scatter(X[:,0],X[:,1],c=bc_spectral_results)

plt.show()

