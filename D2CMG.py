from models.embedder import embedder
from tqdm import tqdm
from evaluate import evaluate
import torch.nn.functional as F
import torch
import torch.nn as nn
from models.Layers import make_mlplayers
from utils.process import get_clones, get_feature_dis, get_A_r, local_preserve, cluster


class Model(nn.Module):
    def __init__(self, n_in, view_num, cfg=None, dropout=0.2, num_community=10):
        super(Model, self).__init__()
        self.view_num = view_num
        MLP = make_mlplayers(n_in, cfg, batch_norm=False)
        self.encoders = get_clones(MLP, self.view_num)
        self.init = nn.ParameterList(
            nn.Parameter(torch.rand(num_community, cfg[-1], dtype=torch.float32)) for _ in range(view_num))
        for param in self.init:
            param.requires_grad = False

        self.dropout = dropout
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Parameter)):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, adjs):
        x_list = [F.dropout(x, self.dropout, training=self.training) for i in range(len(adjs))]
        embed_list = [self.encoders[i](x_list[i]) for i in range(self.view_num)]

        s_list = [get_feature_dis(embed_list[i]) for i in range(self.view_num)]

        embed_fusion = sum(embed_list) / self.view_num
        return embed_list, s_list, embed_fusion

    def embed(self, x):
        embed_list = [self.encoders[i](x) for i in range(self.view_num)]
        embed_fusion = sum(embed_list) / self.view_num
        return embed_fusion.detach()


class Trainer(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args
        self.cfg = args.cfg

    def training(self):

        features = self.features.to(self.args.device)
        adj_list = [adj.to(self.args.device) for adj in self.adj_list]
        adj_label_list = [get_A_r(adj, self.args.A_r) for adj in adj_list]

        d, n = self.cfg[-1], self.args.nb_nodes
        I = torch.eye(d).to(self.args.device)

        model = Model(self.args.ft_size, self.args.view_num, cfg=self.cfg, dropout=self.args.dropout,
                      num_community=self.args.num_community).to(self.args.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-5)

        model.train()

        for _ in tqdm(range(1, self.args.nb_epochs + 1)):
            optimiser.zero_grad()

            embed_list, s_list, z_fusion = model(features, adj_list)

            loss_local = 0
            for i in range(self.args.view_num):
                loss_local += 1 * local_preserve(s_list[i], adj_label_list[i], tau=self.args.tau)

            R = 0
            R_c = 0
            centerList = []
            R_cList = []
            for i in range(self.args.view_num):
                mu_init, _, _ = cluster(embed_list[i], self.args.num_community, 1, self.args.cluster_temp,
                                        model.init[i])
                centerList.append(mu_init.detach().clone())
                _, prob, dist = cluster(embed_list[i], self.args.num_community, 1, self.args.cluster_temp,
                                        mu_init.detach().clone())

                """loss of code reduction"""
                # coding rate of Entire Dataset
                embed = F.normalize(embed_list[i], p=1)
                R += torch.logdet(I + self.args.gama * d / (n * self.args.eps) * (embed.T).matmul(embed)) / 2.0

                # coding rate of group
                pi = prob
                tmp_list = []
                for j in range(self.args.num_community):
                    pi_k = pi[::, j]
                    trace_pi = pi_k.sum()
                    ZPiZ_T = torch.matmul((embed * pi_k.view(-1, 1)).T, embed)
                    log_det = torch.logdet(I + d / (trace_pi * self.args.eps) * ZPiZ_T)
                    R_c += log_det * trace_pi / (n * 2.0)
                    tmp_list.append(log_det * trace_pi / (n * 2.0))
                R_cList.append(torch.stack(tmp_list))
            loss_rd = (R_c - R * self.args.gama)

            """loss of align"""
            loss_align = 0
            for i in range(self.args.view_num):
                for j in range(self.args.view_num):
                    if i == j:
                        continue
                    embed = F.normalize(embed_list[i], p=1)
                    _, pi, _ = cluster(embed_list[i], self.args.num_community, 1, self.args.cluster_temp,
                                       centerList[j].detach().clone())

                    tmp_list = []
                    for k in range(self.args.num_community):
                        pi_k = pi[::, k]
                        trace_pi = pi_k.sum()
                        ZPiZ_T = torch.matmul((embed * pi_k.view(-1, 1)).T, embed)
                        log_det = torch.logdet(I + d / (trace_pi * self.args.eps) * ZPiZ_T)
                        tmp_list.append(log_det * trace_pi / (n * 2.0))
                    loss_align += F.mse_loss(torch.stack(tmp_list), R_cList[j].detach())

            loss = loss_local + loss_rd * self.args.weight_rd + loss_align * self.args.weight_ag
            loss.backward()
            optimiser.step()

        model.eval()
        embed = model.embed(features)
        acc, acc_std, macro_f1, macro_f1_std, micro_f1, micro_f1_std, nmi, nmi_std, acc_cluster, acc_cluster_std, st = evaluate(
            embed, self.idx_train, self.idx_val, self.idx_test, self.labels,
            epoch=self.args.test_epo, lr=self.args.test_lr)
        return acc, acc_std, macro_f1, macro_f1_std, micro_f1, micro_f1_std, nmi, nmi_std, acc_cluster, acc_cluster_std, st
