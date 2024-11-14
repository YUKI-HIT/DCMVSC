"""
该模块主要是建立图关系的，其中相似度计算get_similarity_matrix，
计划改成动态的，需要把使用它的地方替换，复用其他函数代码重写get_miss_adjacency，
"""
import torch
import numpy as np
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize
import scipy.sparse as sp


def normalization_adj(adjacency):
    """计算 L=D^-0.5 * (A+I) * D^-0.5,
    Args:
        adjacency: sp.csr_matrix.
    Returns:
        归一化后的邻接矩阵，类型为 torch.sparse.FloatTensor
    """
    adjacency += sp.eye(adjacency.shape[0])  # 增加自连接
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    L = d_hat.dot(adjacency).dot(d_hat).tocoo()

    # 转换为 torch.sparse.FloatTensor
    indices = torch.from_numpy(np.asarray([L.row, L.col])).long()
    values = torch.from_numpy(L.data.astype(np.float32))
    tensor_adjacency = torch.sparse.FloatTensor(indices, values, L.shape)
    return tensor_adjacency


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_similarity_matrix(features, method='heat'):
    """得到相似度矩阵"""
    dist = None
    if method == 'heat':
        dist = -0.5 * pair(features) ** 2
        dist = np.exp(dist)
    elif method == 'cos':
        # features[features > 0] = 1
        dist = np.dot(features, features.T)
    elif method == 'ncos':
        # features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)
    return dist


def get_edges(dist, topk=10):
    """使用不同的相似度方法，选择topk生成图邻接边关系，但是不是矩阵形式，是数组对形式[[ni,nj],[],……]形式"""
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)
    edges_unordered = []
    for i, ks_i in enumerate(inds):
        for k_i in ks_i:
            if k_i != i:
                edges_unordered.append([i, k_i])
                # edges_unordered.append([i, k_i, dist[i, k_i]])

    return edges_unordered


def get_negative_edges(dist, topk=10):
    _inds = []
    for i in range(dist.shape[0]):
        _ind = np.argpartition(dist[i, :], topk)[:topk]
        _inds.append(_ind)
    _edges_unordered = []
    for i, ks_i in enumerate(_inds):
        for k_i in ks_i:
            if k_i != i:
                _edges_unordered.append([i, k_i])
    return _edges_unordered


def graph2adj(edges_unordered, n, self_join=True):
    """将建立的边关系转换成GCN需要的邻接矩阵"""
    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.array(edges_unordered, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    # adj = sp.coo_matrix((edges[:, 2], (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    """此时的adj是稀疏矩阵的形式"""

    # build symmetric adjacency matrix,使矩阵对称
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)  # 取交集
    adj = adj + adj.T.multiply(adj.T < adj) - adj.multiply(adj.T < adj)  # 取并集
    # adj = 0.5 * (adj + adj.T)
    if self_join:
        adj = adj + sp.eye(adj.shape[0])  # 加入自联结矩阵
    raw_adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj = sparse_mx_to_torch_sparse_tensor(normalize(adj))
    return adj, raw_adj


# 建立多视图的正负边
def positive_negative_adjacency(features=[], topk=10, bottom_k=10 * 5, diffusion_epoch=100):
    """得到所有数据(包括缺失数据)的邻接矩阵,注意这里加入了自连接矩阵，因为数据进行处理了"""
    n = features[0].shape[0]
    p_adjs, n_adjs = [], []
    adjs = []
    for feature in features:
        dist = get_similarity_matrix(feature, 'heat')
        p_edges_unordered = get_edges(dist, topk)
        n_edges_unordered = get_negative_edges(dist, bottom_k)
        adj, _ = graph2adj(p_edges_unordered, n, True)
        _, p_adj = graph2adj(p_edges_unordered, n, False)
        _, n_adj = graph2adj(n_edges_unordered, n, False)
        adjs.append(adj)
        p_adjs.append(p_adj.to_dense())
        n_adjs.append(n_adj.to_dense())
    for k in range(diffusion_epoch):
        print(k)
        n_adjs = negative_diffusion(p_adjs, n_adjs)
    return adjs, p_adjs, n_adjs


def negative_diffusion(adjs, n_adjs):
    """负对扩散策略"""
    negs = []
    for v in range(len(adjs)):
        adj = adjs[v]
        p_adj = n_adjs[v]
        for i in range(adj.shape[0]):
            flag = p_adj[i].bool()
            p_neighbor = torch.sum(adj[flag], dim=0)
            p_adj[i] = p_adj[i] + p_neighbor
            p_adj[i] = torch.where(p_adj[i] != 0, torch.tensor([1], device=p_adj[i].device, dtype = torch.float), p_adj[i])
        negs.append(p_adj)
    return negs


def getIntersection(adj1, adj2):
    """获取交集"""
    adj = adj1 + adj2
    adj = torch.where(adj == 1, torch.tensor([0], device=adj.device), adj)
    adj = torch.where(adj == 2, torch.tensor([1], device=adj.device), adj)
    return adj


def getUnion(adj1, adj2):
    """获取并集"""
    adj = adj1 + adj2
    adj = torch.where(adj == 2, torch.tensor([1], device=adj.device, dtype = torch.float), adj)
    return adj

