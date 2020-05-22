#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import sklearn_tda as tda
from sklearn.cluster import AgglomerativeClustering
np.random.seed(42)


# In[2]:


# some helper functions. Feel free to modify them.

def sw_kernel(diags1, diags2):
    """ check more kernels at
     https://github.com/MathieuCarriere/sklearn-tda/blob/master/example/ex_diagrams.py

     :param: diags1: a list of np.array (shape (n, 2))
     :param diags2: a list of np.array (shape (m, 2))
     :return a np.array of shape n * m
    """
    SW = tda.SlicedWassersteinKernel(num_directions=100, bandwidth=1.)
    X = SW.fit(diags1)
    Y = SW.transform(diags2)
    return Y.T


def plot_kernel(kernel, title='Kernel'):
    plt.matshow(kernel)
    plt.title(title)
    plt.colorbar()
    plt.show()


def kernel2dist(kernel):
    """ convert kernel matrix to distance matrix """
    assert isinstance(kernel, np.ndarray)
    assert (kernel == kernel.T).all(), 'Your kernel is supposed to be symmetric'

    dist_matrix = 2 - 2 * kernel
    assert np.min(dist_matrix) >= 0, 'Dist Matrix is supposed to be postive'
    dist_matrix = np.sqrt(dist_matrix)
    return dist_matrix


def cluster(X):
    """ implement clustering algorithm to cluster persistence diagrams.
        see more examples at https://scikit-learn.org/stable/modules/clustering.html
    """
    clustering = AgglomerativeClustering(affinity='precomputed', linkage='average').fit(X)
    return clustering.labels_


# # Generate persistence diagrams
# generate 40 persistence diagrams. Note that I delibrately generate two types of diagrams that are quite different.
diags = [np.random.random((10, 2)) for _ in range(20)]
diags += [1 + np.random.random((10, 2)) for _ in range(20)]

# show one persistence diagram in the form of persistence image
PI = tda.PersistenceImage(bandwidth=.1, weight=lambda x: x[1], im_range=[0,1,0,1], resolution=[100,100])
pi = PI.fit_transform(diags[:1])
plt.imshow(np.flip(np.reshape(pi[0], [100,100]), 0))
plt.title("Persistence Image")
plt.show()


# # Compute kernel for persistence diagrams
# compute sliced wasserstein kernels
kernel = sw_kernel(diags, diags)
plot_kernel(kernel, title='Kernel')


# # Convert Kernel to Distance Matrix
# plot distance matrix
dist = kernel2dist(kernel)
plot_kernel(dist, title='Distance Matrix')


# # Cluster Persistence Diagrams
# cluster and print out labels. Since we knew there are two "types" of persistence diagrams by construction,
# we expect that clustering algorithm can differentiate them, which is indeed the case!
labels = cluster(dist)
print(labels)






