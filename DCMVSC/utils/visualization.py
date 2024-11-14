"""
该模块是可视化画图函数，直接复用性比较低，依赖数据格式，部分需重写
"""
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

base_dir = './imgs/'

# 颜色可能较少，需要时添加
# colorList = [
#     '#F27970',
#     '#BB9727',
#     '#54B345',
#     '#05B9E2',
#     '#8983BF',
#     '#C76DA2',
#     '#9ac9db'
# ]

colorList = [
    '#F27970',
    '#BB9727',
    '#54B345',
    '#32B897',
    '#05B9E2',
    '#8983BF',
    '#C76DA2',
    '#8ECFC9',
    '#FFBE7A',
    '#FA7F6F',
    '#82B0D2',
    '#BEB8DC',
    '#E7DAD2',
    '#2878b5',
    '#9ac9db',
    '#f8ac8c',
    '#c82423',
    '#ff8884',
    '#A1A9D0',
    '#F0988C',
    '#B883D4',
    '#9E9E9E',
    '#CFEAF1',
    '#C4A5DE',
    '#F6CAE5',
    '#96CCCB',
    '#63b2ee',
    '#76da91',
    '#f8cb7f',
    '#f89588',
    '#7cd6cf',
    '#9192ab',
    '#7898e1',
    '#efa666',
    '#eddd86',
    '#9987ce',
    '#63b2ee',
    '#76da91',
    '#002c53',
    '#ffa510',
    '#0c84c6',
    '#ffbd66',
    '#f74d4d',
    '#2455a4',
    '#41b7ac'
]


def TSNE_show2D(z, y, name=''):
    """use t-SNE to visualize the latent representation"""
    t_sne = TSNE(n_components=2, learning_rate='auto')
    data = t_sne.fit_transform(z)
    data = pd.DataFrame(data, index=y)
    color = [colorList[i - 1] for i in data.index]
    plt.scatter(data[0], data[1], c=color, marker='.', s=180)
    # plt.xticks(fontsize=10)
    # plt.yticks(fontsize=10)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(base_dir + 'TSNE2D' + name + '.pdf')
    plt.show()


def TSNE_show3D(z, y):
    """use t-SNE to visualize 3D the latent representation"""
    t_sne = TSNE(n_components=3, learning_rate='auto')
    data = t_sne.fit_transform(z)
    data = pd.DataFrame(data, index=y)
    color = [colorList[i - 1] for i in data.index]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[0], data[1], data[2], c=color, s=12)

    # ax.set_xlim3d(min(data[0].sort_values()[5:-5]), max(data[0].sort_values()[5:-5]))
    # ax.set_ylim3d(min(data[1].sort_values()[5:-5]), max(data[1].sort_values()[5:-5]))
    # ax.set_zlim3d(min(data[2].sort_values()[5:-5]), max(data[2].sort_values()[5:-5]))
    ax.set_xlim3d(min(data[0]), max(data[0]))
    ax.set_ylim3d(min(data[1]), max(data[1]))
    ax.set_zlim3d(min(data[2]), max(data[2]))

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    plt.tick_params(labelsize=8)
    ax.grid(True)
    # ax.view_init(10, 185)#设置角度
    plt.subplots_adjust(left=-0., right=1., top=1., bottom=-0.)
    plt.tight_layout()
    plt.savefig(base_dir + 'TSNE3D' + str(datetime.strftime(datetime.now(), '%d %H-%M-%S')) + '.pdf')
    plt.show()


def loss_plot(loss, acc, nmi, ari,img=''):
    """ 画收敛分析图，loss、acc、nmi、ari随epoch的变化"""
    epochs = range(1, len(loss) + 1)
    fig = plt.figure(figsize=(8, 5))
    ax_left = fig.add_subplot(111)
    ax_right = ax_left.twinx()

    ax_left.set_xlabel('Epoch', fontsize=16)
    ax_left.set_ylabel('Clustering Performance', fontsize=16)
    ax_right.set_ylabel('Loss', fontsize=16)

    a1 = ax_right.plot(epochs, loss, color='#c82423', label='Loss')
    a2 = ax_left.plot(epochs, acc, color='#54B345', label='ACC')
    a3 = ax_left.plot(epochs, nmi, color='#05B9E2', label='NMI')
    a4 = ax_left.plot(epochs, ari, color='#675083', label='ARI')

    lns = a1 + a2 + a3 + a4
    labs = [l.get_label() for l in lns]
    ax_left.legend(lns, labs, loc='center right', fontsize=12)
    ax_left.tick_params(labelsize=12)
    ax_right.tick_params(labelsize=12)

    plt.tight_layout()
    #plt.savefig(base_dir+ img + 'loss.pdf')
    plt.show()


def miss_plot(flag=1):
    """画差带图"""
    # 设置缺失率和准确率的均值和标准差
    missing_rates = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9])

    data_name = None
    if flag == 1:
        data_name = 'acc'
    elif flag == 2:
        data_name = 'nmi'
    elif flag == 3:
        data_name = 'ari'

    # 读取数据
    data = pd.read_csv('D:/bbbbFile/桌面学习/myWork/Img-v/' + data_name + '.csv')
    data = np.array(data)
    # 获取每个算法的均值和标准差
    BSV_mean, BSV_std = data[:, 0], data[:, 1]
    Concat_mean, Concat_std = data[:, 2], data[:, 3]
    PVC_mean, PVC_std = data[:, 4], data[:, 5]
    MIC_mean, MIC_std = data[:, 6], data[:, 7]
    DAIMC_mean, DAIMC_std = data[:, 8], data[:, 9]
    Completer_mean, Completer_std = data[:, 10], data[:, 11]
    DSIMVC_mean, DSIMVC_std = data[:, 12], data[:, 13]
    DIMVC_mean, DIMVC_std = data[:, 14], data[:, 15]
    Ours_mean, Ours_std = data[:, 16], data[:, 17]

    # 自定义颜色
    colors = ['#F27970', '#BB9727', '#54B345', '#32B897', '#05B9E2', '#8983BF', '#C76DA2', '#2878b5', '#c82423']
    plt.figure(figsize=(8, 5))
    # 绘制误差带图
    plt.plot(missing_rates, BSV_mean, color=colors[0], label='BSV')
    plt.fill_between(missing_rates, BSV_mean - BSV_std, BSV_mean + BSV_std, alpha=0.2, color=colors[0])

    plt.plot(missing_rates, Concat_mean, color=colors[1], label='Concat')
    plt.fill_between(missing_rates, Concat_mean - Concat_std, Concat_mean + Concat_std, alpha=0.2, color=colors[1])

    plt.plot(missing_rates, PVC_mean, color=colors[2], label='PVC')
    plt.fill_between(missing_rates, PVC_mean - PVC_std, PVC_mean + PVC_std, alpha=0.2, color=colors[2])

    plt.plot(missing_rates, MIC_mean, color=colors[3], label='MIC')
    plt.fill_between(missing_rates, MIC_mean - MIC_std, MIC_mean + MIC_std, alpha=0.2, color=colors[3])

    plt.plot(missing_rates, DAIMC_mean, color=colors[4], label='DAIMC')
    plt.fill_between(missing_rates, DAIMC_mean - DAIMC_std, DAIMC_mean + DAIMC_std, alpha=0.2, color=colors[4])

    plt.plot(missing_rates, Completer_mean, color=colors[5], label='Completer')
    plt.fill_between(missing_rates, Completer_mean - Completer_std, Completer_mean + Completer_std, alpha=0.2,
                     color=colors[5])

    plt.plot(missing_rates, DSIMVC_mean, color=colors[6], label='DSIMVC')
    plt.fill_between(missing_rates, DSIMVC_mean - DSIMVC_std, DSIMVC_mean + DSIMVC_std, alpha=0.2, color=colors[6])

    plt.plot(missing_rates, DIMVC_mean, color=colors[7], label='DIMVC')
    plt.fill_between(missing_rates, DIMVC_mean - DIMVC_std, DIMVC_mean + DIMVC_std, alpha=0.2, color=colors[7])

    plt.plot(missing_rates, Ours_mean, color=colors[8], label='Ours')
    plt.fill_between(missing_rates, Ours_mean - Ours_std, Ours_mean + Ours_std, alpha=0.2, color=colors[8])

    plt.xlabel('Missing rate', fontsize=22)
    if flag == 1:
        plt.ylabel('Accuracy (%)', fontsize=22)
        plt.title('Accuracy with different missing rates', fontsize=22)
    elif flag == 2:
        plt.ylabel('NMI (%)', fontsize=22)
        plt.title('NMI with different missing rates', fontsize=22)
    elif flag == 3:
        plt.ylabel('ARI (%)', fontsize=22)
        plt.title('ARI with different missing rates', fontsize=18)

    plt.legend(fontsize=10, loc='upper right')
    plt.tick_params(labelsize=12)
    # plt.xticks(missing_rates)

    # 保存为PDF格式并边距极小
    plt.tight_layout()
    plt.savefig(base_dir + 'miss' + data_name + '.pdf')


def k_anl_plot():
    """K值敏感度分析图"""
    # 读取数据
    data = pd.read_csv('D:/bbbbFile/桌面学习/myWork/Img-v/K.csv')
    data = np.array(data)
    K = range(1, len(data[:, 0]) + 1)

    plt.figure(figsize=(8, 5))
    # 自定义颜色
    colors = ['#54B345', '#05B9E2', '#c82423']
    plt.plot(K, data[:, 0], color=colors[0], label='ACC', marker='o')
    plt.plot(K, data[:, 1], color=colors[1], label='NMI', marker='+')
    plt.plot(K, data[:, 2], color=colors[2], label='ARI', marker='*')

    plt.xlabel('Number of K', fontsize=16)
    plt.ylabel('Clustering Performance (%)', fontsize=16)

    plt.legend(fontsize=10, loc='upper right')
    plt.tick_params(labelsize=12)
    plt.xticks(K[::2])

    # 保存为PDF格式并边距极小
    plt.tight_layout()
    plt.savefig(base_dir + 'K_anl.pdf')


def heatmap(z, img=''):
    """画热力图"""
    colors = ['RdBu_r', 'Blues', 'GnBu', 'PuBu', 'YlGnBu''YlOrRd']
    color = colors[0]
    # mat = cosine_similarity(z)

#     sns.heatmap(z * 25, annot=False, vmax=0.5, vmin=0.07, cmap=color, cbar_kws={"ticks": []})  # MSRC_v1
#     sns.heatmap(z * 50, annot=False, vmax=1.0, vmin=0.0, cmap=color, cbar_kws={"ticks": []})  # ORL_mtv
#     sns.heatmap(z * 500, annot=False, vmax=0.7, vmin=0.2, cmap=color, cbar_kws={"ticks": []})  # handwritten
    # sns.heatmap(z * 500, annot=False, vmax=0.8, vmin=0.15, cmap=color, cbar_kws={"ticks": []})  # landUse
    # sns.heatmap(z * 1000, annot=False, vmax=1.0, vmin=0.2, cmap=color, cbar_kws={"ticks": []})  # scene
    sns.heatmap(z * 4000, annot=False, vmax=0.5, vmin=0.35, cmap=color, cbar_kws={"ticks": []})  # noisyMNIST

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(base_dir + 'similarity' + img + '.png')
    plt.show()