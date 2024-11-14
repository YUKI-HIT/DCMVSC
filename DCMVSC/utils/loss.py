import math
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn

EPS = sys.float_info.epsilon


class InstanceLoss(nn.Module):
    """实例级别的对比损失"""

    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class ClusterLoss(nn.Module):
    """类簇级别的对比损失"""

    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j, alpha=1.0):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss + alpha * ne_loss


def cdist(X, Y):
    """Pairwise distance between rows of X and rows of Y"""
    xyT = X @ torch.t(Y)
    x2 = torch.sum(X ** 2, dim=1, keepdim=True)
    y2 = torch.sum(Y ** 2, dim=1, keepdim=True)
    d = x2 - 2 * xyT + torch.t(y2)
    return d


def CS_divergence(Y, K):
    eps = 1E-9
    n_clusters = Y.shape[1]
    nom = torch.t(Y) @ K @ Y
    dnom_squared = torch.unsqueeze(torch.diagonal(nom), -1) @ torch.unsqueeze(torch.diagonal(nom), 0)
    nom = torch.where(nom < eps, nom.new_tensor(eps), nom)
    dnom_squared = torch.where(dnom_squared < eps, dnom_squared.new_tensor(eps ** 2), dnom_squared)
    d = 2 / (n_clusters * (n_clusters - 1)) * torch.sum(torch.triu(nom / torch.sqrt(dnom_squared), diagonal=1))
    return d


def CS_divergence_loss_E(Y, K):
    n_clusters = Y.shape[1]
    eye = torch.eye(n_clusters, device=Y.device)
    m = torch.exp(-cdist(Y, eye))
    return CS_divergence(m, K)


def L2_penalty(C):
    return (C ** 2).sum()


class cluster_RINCE(nn.Module):
    """类簇级别的对比损失"""

    def __init__(self, class_num, temperature, device, q=1, lam=0.01):
        super(cluster_RINCE, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device
        self.q = q
        self.lam = 0.01

        self.mask = self.mask_correlated_clusters(class_num)
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j, alpha=1.0):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)
        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)
        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        pos_c = torch.exp(positive_clusters)
        neg_c = torch.exp(negative_clusters)
        neg_c = torch.sum(neg_c, dim=1)
        loss = (-pos_c ** self.q / self.q + (self.lam * (pos_c + neg_c)) ** self.q / self.q).sum()
        loss /= N
        return loss + alpha * ne_loss


# 公式直接实现
def contrastive_loss(C, positive, negative, temp=0.5):
    """C是相似度矩阵。positive,negative是正负样本指示矩阵"""
    Po = positive.bool()
    Ne = negative.bool()
    N = C.shape[0]
    C = torch.exp(C / temp)
    C_po = C[Po]
    index = torch.nonzero(Po, as_tuple=False)[:, 0]
    C_ne = (C * Ne).sum(dim=1)
    C_ne = torch.index_select(C_ne, dim=0, index=index)
    loss = -torch.log(C_po / (C_po + C_ne)).sum()
    return loss / N


# 交叉熵公式直接实现
def contrastive_loss2(C, positive, negative, temp=0.01):
    """C是相似度矩阵。positive,negative是正负样本指示矩阵"""
    Po = positive.bool()
    Ne = negative.bool()
    C_po = C[Po]
    N = C_po.shape[0]
    C = torch.exp(C / temp)
    index = torch.nonzero(Po, as_tuple=False)[:, 0]
    C_ne = (C * Ne).sum(dim=1)
    C_ne = torch.index_select(C_ne, dim=0, index=index)
    loss = (-C_po + torch.log(C[Po] + C_ne)).sum()
    return loss / N


# 自带交叉熵实现
def contrastive_loss3(C, positive, negative, temp=0.001):
    Po = positive.bool()
    Ne = negative.bool()

    C = C / temp
    C_po = C[Po]
    C_ne = C * Ne
    C_ne = torch.where(C_ne == 0, torch.tensor([-torch.inf], device=C_ne.device), C_ne)
    index = torch.nonzero(Po, as_tuple=False)[:, 0]
    C_ne = torch.index_select(C_ne, dim=0, index=index)

    N = C_po.shape[0]
    labels = torch.zeros(N).to(C_po.device).long()
    logits = torch.cat((C_po.reshape(N, 1), C_ne), dim=1)
    loss = F.cross_entropy(logits, labels)
    loss = loss / N
    return loss


# RINCE实现
def contrastive_loss4(C, positive, negative, temp=1, q=1, lam=0.01):
    """C是相似度矩阵。positive,negative是正负样本指示矩阵"""
    Po = positive.bool()
    Ne = negative.bool()

    C = torch.exp(C / temp)
    C_po = C[Po]
    N = C_po.shape[0]
    index = torch.nonzero(Po, as_tuple=False)[:, 0]
    C_ne = (C * Ne).sum(dim=1)
    C_ne = torch.index_select(C_ne, dim=0, index=index)
    loss = (-C_po ** q / q + (lam * (C_po + C_ne)) ** q / q).sum()
    return loss / N




