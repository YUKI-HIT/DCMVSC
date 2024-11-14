import torch.optim
from utils.evaluation import evaluation
from models.baseModels import *
from utils.loss import *
import torch.optim
import warnings
from utils.graph_adjacency import *
from utils.util import *
from utils.visualization import *
from utils.run_cluster import spectral_clustering, kmeans_clustering, spectral_clustering_without_post

warnings.simplefilter("ignore")



class ModelS(nn.Module):

    def __init__(self, config):
        """定义常用参数和搭建模型的模块"""
        super(ModelS, self).__init__()
        self._config = config
        self._input_dim1 = config['Autoencoder']['gcnEncoder1'][0]
        self._input_dim2 = config['Autoencoder']['gcnEncoder2'][0]

        self._latent_dim = config['Autoencoder']['gcnEncoder1'][-1]

        self._n_clusters = config['n_clusters']
        self.n = config['n']
        # encoder
        self.gcnEncoder1 = GraphEncoder(config['Autoencoder']['gcnEncoder1'], 'relu', True, False)
        self.gcnEncoder2 = GraphEncoder(config['Autoencoder']['gcnEncoder2'], 'relu', True, False)

        #self.gcnEncoder1 = Encoder(config['Autoencoder']['gcnEncoder1'], 'relu', batchnorm =True)
        #self.gcnEncoder2 = Encoder(config['Autoencoder']['gcnEncoder1'], 'relu', batchnorm =True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        t = self.gcnEncoder1().to(device)

        # predict(graph decoder)
        self.predict1 = InnerProductDecoder()
        self.predict2 = InnerProductDecoder()
        # cluster
        self.cluster1 = ClusterProject(self._latent_dim, self._n_clusters)
        self.cluster2 = ClusterProject(self._latent_dim, self._n_clusters)
        # self-expression
        self.C = nn.Parameter(-1e4 * torch.ones(self.n, self.n, dtype=torch.float32))
        # weight
        self.w = torch.nn.Parameter(torch.ones(self.n, 2))

    def forward(self, x1, x2, adj1, adj2):
        h1 = self.gcnEncoder1(x1, adj1, True)
        h2 = self.gcnEncoder2(x2, adj2, True)
        # h1 = self.gcnEncoder1(x1)
        # h2 = self.gcnEncoder2(x2)

        y1, _ = self.cluster1(h1)
        y2, _ = self.cluster2(h2)
        return h1, h2, y1, y2

    def run_train(self, x_train, Y_list, adj, p_adj, n_adj, optimizer, logger, accumulated_metrics, device):
        epochs = self._config['training']['epoch']
        print_num = self._config['print_num']

        alpha1 = self._config['alpha1']
        alpha2 = self._config['alpha2']
        alpha3 = self._config['alpha3']
        alpha4 = self._config['alpha4']
        alpha5 = self._config['alpha5']

        LOSS = []
        criterion_cluster = ClusterLoss(self._n_clusters, 1.0, device).to(device)  # train the model
        for k in range(epochs):
            h1, h2, y1, y2 = self(x_train[0], x_train[1], adj[0], adj[1])
            sim1 = self.predict1(h1)
            sim2 = self.predict2(h2)
            beta = self.w / torch.sum(self.w, dim=1, keepdim=True)
            Y = beta[:, 0].unsqueeze(1) * y1 + beta[:, 1].unsqueeze(1) * y2

            # representation learning
            loss_repr = (contrastive_loss3(sim1, p_adj[0], n_adj[0]) + contrastive_loss3(sim2, p_adj[1], n_adj[1]))
            loss = alpha1 * loss_repr

            # self-expression loss
            C = F.softmax(self.C, dim=1)
            loss_selfExp = F.mse_loss(C @ h1.detach(), h1) + F.mse_loss(C @ h2.detach(), h2) + L2_penalty(C)
            loss += alpha2 * loss_selfExp

            loss_coef = (contrastive_loss3(C, p_adj[0], n_adj[0]) + contrastive_loss3(C, p_adj[1], n_adj[0]))
            loss += alpha3 * loss_coef

            # spectral cluster CS-Divergence loss
            W = 0.5 * (C + C.T)
            loss_cluster = criterion_cluster(y1, y2)#对比学习
            loss += alpha4 * CS_divergence(Y, W.detach())
            loss += alpha5 * loss_cluster

            # parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            LOSS.append(loss.item())

            # evaluation
            if k == 0 or (k + 1) % print_num == 0:  #
                output = ("Epoch:{:.0f}/{:.0f}===>loss={:.4f}".format((k + 1), epochs, loss.item()))
                logger.info("\033[2;29m" + output + "\033[0m")
                self.run_eval(x_train, Y_list, adj, accumulated_metrics)

        loss_plot(LOSS, accumulated_metrics['acc'], accumulated_metrics['nmi'], accumulated_metrics['ARI'],
                  self._config['dataset'])
        return accumulated_metrics['acc'][-1], accumulated_metrics['nmi'][-1], accumulated_metrics['ARI'][-1]

    def run_eval(self, x_train, Y_list, adj, accumulated_metrics):
        """ this method is used to evluation and output the result"""
        with torch.no_grad():
            h1, h2, y1, y2 = self(x_train[0], x_train[1], adj[0], adj[1])
            beta = self.w / torch.sum(self.w, dim=1, keepdim=True)
            Y = (beta[:, 0].unsqueeze(1) * y1 + beta[:, 1].unsqueeze(1) * y2).data.cpu().numpy().argmax(1)
            scores = evaluation(y_pred=Y, y_true=Y_list[0], accumulated_metrics=accumulated_metrics)
            print(scores, sep='\n')

