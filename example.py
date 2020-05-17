# Created at 2020-05-16
# Summary: a short tutorial for clustering persistence diagrams

import matplotlib.pyplot as plt
import numpy as np
import sklearn_tda as tda
from sklearn.cluster import AgglomerativeClustering


def sw_kernel(diags1, diags2):
    """ check more kernels at
     https://github.com/MathieuCarriere/sklearn-tda/blob/master/example/ex_diagrams.py

     :param: diags1: a list of np.array (shape (n, 2))
     :param diags2: a list of np.array (shape (m, 2))
     :return a np.array of shape k * m
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
    assert isinstance(kernel, np.ndarray)
    assert (kernel == kernel.T).all(), 'Your kernel is supposed to be symmetric'

    dist_matrix = 2 - 2 * kernel
    assert np.min(dist_matrix) >= 0, 'Dist Matrix is supposed to be postive'
    dist_matrix = np.sqrt(dist_matrix)
    return dist_matrix


def cluster(X):
    """ implement clustering algorithm to cluster persistence diagrams """
    clustering = AgglomerativeClustering(affinity='precomputed', linkage='average').fit(X)
    return clustering.labels_


if __name__ == '__main__':
    np.random.seed(42)
    # generate 40 persistence diagrams
    diags = [np.random.random((10, 2)) for _ in range(20)]
    diags += [1 + np.random.random((10, 2)) for _ in range(20)]

    # compute sliced wasserstein kernels
    kernel = sw_kernel(diags, diags)
    plot_kernel(kernel, title='Kernel')

    # plot distance matrix
    dist = kernel2dist(kernel)
    plot_kernel(dist, title='Distance Matrix')

    # cluster and print out labels
    labels = cluster(dist)
    print(labels)
