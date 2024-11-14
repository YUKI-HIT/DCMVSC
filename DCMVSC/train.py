import torch.optim
import time
import torch.optim
from utils.util import *
from utils.std_utils import *
from utils.datasets import *
import collections
from utils.graph_adjacency import *
from config import get_config

from models.ModelS import ModelS

data_dict = {
    1: 'Scene-15',
    2: 'LandUse-21',
    3: 'handwritten',
    4: 'MSRC_v1',
    5: 'ORL_mtv',
    6: 'NoisyMNIST',
}


def main():
    for flag in [1]:

        # prepare
        test_time = 5
        device = get_device()
        config = get_config(flag=flag)
        config['print_num'] = 50
        config['training']['epoch'] = 500

        # logger
        logger, _ = get_logger(config)
        logger.info('Dataset:' + str(config['dataset']))

        for k in [7]:
            for de in [1]:
                X_list, Y_list = load_data(config)
                print('load data')
                X1 = X_list[0]
                X2 = X_list[1]
                adjs, p_adjs, n_adjs = positive_negative_adjacency([X1, X2],
                                                                    config['topk'],
                                                                    config['bottom_k'],
                                                                    config['diffusion_epoch'])

                # revise config
                config['Autoencoder']['gcnEncoder1'][0] = X1.shape[1]
                config['Autoencoder']['gcnEncoder2'][0] = X2.shape[1]
                config['n'] = X1.shape[0]

                # data and mask to device
                X1 = torch.from_numpy(X1).float().to(device)
                X2 = torch.from_numpy(X2).float().to(device)

                # get adjacency
                adj1 = adjs[0].to(device)
                adj2 = adjs[1].to(device)
                p_adj1 = p_adjs[0].float().to(device)
                p_adj2 = p_adjs[1].float().to(device)
                n_adj1 = n_adjs[0].float().to(device)
                n_adj2 = n_adjs[1].float().to(device)

                config['alpha1'] = 100
                config['alpha2'] = 0.001
                config['alpha3'] = 100
                config['alpha4'] = 10
                config['alpha5'] = 10
                fold_acc, fold_nmi, fold_ari = [], [], []
                for data_seed in [0, 1, 2, 3, 4]:
                    start = time.time()
                    setup_seed(data_seed)
                    accumulated_metrics = collections.defaultdict(list)  # Accumulated metrics
                    # Build model
                    model = ModelS(config)
                    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
                    model.to(device)
                    # Training
                    acc, nmi, ari = model.run_train([X1, X2], Y_list,
                                                    [adj1, adj2], [p_adj1, p_adj2], [n_adj1, n_adj2],
                                                    optimizer, logger, accumulated_metrics, device)
                    fold_acc.append(acc), fold_nmi.append(nmi), fold_ari.append(ari)
                    print(time.time() - start)
                logger.info('--------------------Training over--------------------')
                acc, nmi, ari = cal_std(logger, fold_acc, fold_nmi, fold_ari)
                print('acc:', acc, ',nmi:', nmi, ',ari:', ari)
                logger.handlers.clear()


if __name__ == '__main__':
    main()






