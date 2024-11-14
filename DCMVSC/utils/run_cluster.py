"""
此模块是对子空间聚类的稀疏矩阵进行谱聚类的过程
"""
import numpy as np
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize

from sklearn.cluster import KMeans
from utils.evaluation import get_cluster_sols


def kmeans_clustering(z, n_clusters):
    """对潜在表示进行kmeans聚类后，计算准确度等指标"""
    y_pred, _ = get_cluster_sols(z, ClusterClass=KMeans, n_clusters=n_clusters, init_args={'n_init': 10})
    return y_pred


def spectral_clustering_without_post(L, K):
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack',
                                          affinity='precomputed', assign_labels='discretize')
    spectral.fit(L)
    y = spectral.fit_predict(L)
    return y


def spectral_clustering(C, K, d=3, alpha=0.2, ro=1):
    """C是系数矩阵，K是簇数，d是子空间维度，alpha是去噪保留的比例，ro是增强指数"""
    C = thrC(C, alpha)
    y, _ = post_proC(C, K, d, ro)
    return y


def thrC(C, alpha):
    if alpha < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while (stop == False):
                csum = csum + S[t, i]
                if csum > alpha * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C
    return Cp


def post_proC(C, K, d, ro):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    n = C.shape[0]
    r = d * K + 1
    print(r)
    U, S, _ = svds(C, r, v0=np.ones(n))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** ro)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K,
                                          eigen_solver='arpack',
                                          affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L)
    return grp, L
