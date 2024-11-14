"""
其他用到的工具函数，包括缺失指示矩阵生成get_mask，获得日志记录get_logger，获得设备get_device和固定随机种子set_seed
"""
import numpy as np
import torch
import logging
import datetime
import os
import random
from numpy.random import randint


def normalize2sum2one(X):
    T = torch.sum(X, dim=1).unsqueeze(1)
    return X / T


def CosSimilarity(h1, h2):
    sim = h1 @ h2.T
    h1_l = torch.norm(h1, p=2, dim=1)
    h2_l = torch.norm(h1, p=2, dim=1)
    sim_l = h1_l * h2_l.unsqueeze(1)
    return sim / sim_l


def get_logger(config, main_dir='./logs/'):
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: - %(message)s', datefmt='%m-%d %H:%M:%S')

    plt_name = str(config['dataset']) + ' ' + str(
        datetime.datetime.strftime(datetime.datetime.now(), '%m-%d %H-%M-%S'))

    fh = logging.FileHandler(
        main_dir + str(config['dataset']) + ' ' + str(
            datetime.datetime.strftime(datetime.datetime.now(), '%m-%d %H-%M-%S')) + '.logs')

    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, plt_name


def get_device():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # GPU设备排号
    os.environ["CUDA_VISIBLE_DEVICES"] = str('0')  # 设置device
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    return device


def setup_seed(seed):
    """set up random seed"""
    np.random.seed(seed)
    random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
