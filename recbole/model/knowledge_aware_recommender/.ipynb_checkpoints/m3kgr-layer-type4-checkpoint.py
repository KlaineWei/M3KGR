# -*- coding: utf-8 -*-
# @Time   : 2020/9/15
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

r"""
KGAT
##################################################
Reference:
    Xiang Wang et al. "KGAT: Knowledge Graph Attention Network for Recommendation." in SIGKDD 2019.

Reference code:
    https://github.com/xiangwang1223/knowledge_graph_attention_network
"""

import numpy as np
import pandas
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_normal_initialization, constant_, xavier_normal_
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

import os


class Aggregator(nn.Module):
    """ GNN Aggregator layer
    """

    def __init__(self, input_dim, output_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)

        if self.aggregator_type == 'gcn':
            self.W = nn.Linear(self.input_dim, self.output_dim)
        elif self.aggregator_type == 'graphsage':
            self.W = nn.Linear(self.input_dim * 2, self.output_dim)
        elif self.aggregator_type == 'bi':
            self.W1 = nn.Linear(self.input_dim, self.output_dim)
            self.W2 = nn.Linear(self.input_dim, self.output_dim)
        else:
            raise NotImplementedError

        self.activation = nn.LeakyReLU()

    def forward(self, norm_matrix, ego_embeddings):
        side_embeddings = torch.sparse.mm(norm_matrix, ego_embeddings)

        if self.aggregator_type == 'gcn':
            ego_embeddings = self.activation(self.W(ego_embeddings + side_embeddings))
        elif self.aggregator_type == 'graphsage':
            ego_embeddings = self.activation(self.W(torch.cat([ego_embeddings, side_embeddings], dim=1)))
        elif self.aggregator_type == 'bi':
            add_embeddings = ego_embeddings + side_embeddings
            sum_embeddings = self.activation(self.W1(add_embeddings))
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = self.activation(self.W2(bi_embeddings))
            ego_embeddings = bi_embeddings + sum_embeddings
        else:
            raise NotImplementedError

        ego_embeddings = self.message_dropout(ego_embeddings)

        return ego_embeddings


class M3KGR(KnowledgeRecommender):
    r"""KGAT is a knowledge-based recommendation model. It combines knowledge graph and the user-item interaction
    graph to a new graph called collaborative knowledge graph (CKG). This model learns the representations of users and
    items by exploiting the structure of CKG. It adopts a GNN-based architecture and define the attention on the CKG.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(M3KGR, self).__init__(config, dataset)

        # load dataset info
        self.ckg = dataset.ckg_graph(form='dgl', value_field='relation_id')
        self.all_hs = torch.LongTensor(dataset.ckg_graph(form='coo', value_field='relation_id').row).to(self.device)
        self.all_ts = torch.LongTensor(dataset.ckg_graph(form='coo', value_field='relation_id').col).to(self.device)
        self.all_rs = torch.LongTensor(dataset.ckg_graph(form='coo', value_field='relation_id').data).to(self.device)
        self.matrix_size = torch.Size([self.n_users + self.n_entities, self.n_users + self.n_entities])

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.kg_embedding_size = config['kg_embedding_size']
        self.layers = [self.embedding_size] + config['layers']
        self.aggregator_type = config['aggregator_type']
        self.mess_dropout = config['mess_dropout']
        self.reg_weight = config['reg_weight']

        # moco
        self.dim = config['dim']
        self.K = config['k']
        self.m = config['m']
        self.T = config['t']
        self.phi = config['phi']

        # multimodal
        self.multimodal_size = config['multimodal_size']
        self.use_att = config['use_att']
        self.use_image = config['use_image']
        self.use_text = config['use_text']
        self.multi_step = config['multi_step']
        self.att_type = config['att_type']

        # generate intermediate data
        self.A_in = self.init_graph()  # init the attention matrix by the structure of ckg
        self.A_in_1 = self.A_in
        self.A_in_2 = self.A_in
        affine = True
        self.projection_head = torch.nn.ModuleList()
        inner_size = self.layers[-1] * 2
        print("inner size:", inner_size)
        self.projection_head.append(torch.nn.Linear(inner_size, inner_size * 4, bias=False))
        self.projection_head.append(torch.nn.BatchNorm1d(inner_size * 4, eps=1e-12, affine=affine))
        self.projection_head.append(torch.nn.Linear(inner_size * 4, inner_size, bias=False))
        self.projection_head.append(torch.nn.BatchNorm1d(inner_size, eps=1e-12, affine=affine))
        self.mode = 0
        self.projection_head.to(self.device)

        # multimodal projection head img
        self.projection_head_multi_1 = torch.nn.ModuleList()
        self.projection_head_multi_1.append(torch.nn.Linear(self.multimodal_size, self.embedding_size * 4, bias=False))
        self.projection_head_multi_1.append(torch.nn.BatchNorm1d(self.embedding_size * 4, eps=1e-12, affine=affine))
        self.projection_head_multi_1.append(torch.nn.Linear(self.embedding_size * 4, self.embedding_size, bias=False))
        self.projection_head_multi_1.append(torch.nn.BatchNorm1d(self.embedding_size, eps=1e-12, affine=affine))
        self.mode_multi_1 = 0
        self.projection_head_multi_1.to(self.device)
        
        # multimodal projection head text
        self.projection_head_multi_2 = torch.nn.ModuleList()
        self.projection_head_multi_2.append(torch.nn.Linear(self.multimodal_size, self.embedding_size * 4, bias=False))
        self.projection_head_multi_2.append(torch.nn.BatchNorm1d(self.embedding_size * 4, eps=1e-12, affine=affine))
        self.projection_head_multi_2.append(torch.nn.Linear(self.embedding_size * 4, self.embedding_size, bias=False))
        self.projection_head_multi_2.append(torch.nn.BatchNorm1d(self.embedding_size, eps=1e-12, affine=affine))
        self.mode_multi_2 = 0
        self.projection_head_multi_2.to(self.device)
        
        # unify projection head img+text
        # self.projection_head_uni_1 = torch.nn.ModuleList()
        # self.projection_head_uni_1.append(torch.nn.Linear(self.embedding_size * 2, self.embedding_size * 2, bias=False))
        # self.projection_head_uni_1.append(torch.nn.BatchNorm1d(self.embedding_size * 2, eps=1e-12, affine=affine))
        # self.projection_head_uni_1.append(torch.nn.Linear(self.embedding_size * 2, self.embedding_size, bias=False))
        # self.projection_head_uni_1.append(torch.nn.BatchNorm1d(self.embedding_size, eps=1e-12, affine=affine))
        # self.mode_uni_1 = 0
        # self.projection_head_uni_1.to(self.device)
        
        # unify projection head entity+multimodal
        self.projection_head_uni_2 = torch.nn.ModuleList()
        self.projection_head_uni_2.append(torch.nn.Linear(self.embedding_size * 3, self.embedding_size * 3, bias=False))
        self.projection_head_uni_2.append(torch.nn.BatchNorm1d(self.embedding_size * 3, eps=1e-12, affine=affine))
        self.projection_head_uni_2.append(torch.nn.Linear(self.embedding_size * 3, self.embedding_size, bias=False))
        self.projection_head_uni_2.append(torch.nn.BatchNorm1d(self.embedding_size, eps=1e-12, affine=affine))
        self.mode_uni_2 = 0
        self.projection_head_uni_2.to(self.device)

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations, self.kg_embedding_size)
        self.trans_w = nn.Embedding(self.n_relations, self.embedding_size * self.kg_embedding_size)
        # self.i_trans_w = nn.Embedding(self.n_relations, self.embedding_size * self.kg_embedding_size)
        # self.t_trans_w = nn.Embedding(self.n_relations, self.embedding_size * self.kg_embedding_size)
        self.aggregator_layers = nn.ModuleList()
        for idx, (input_dim, output_dim) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.aggregator_layers.append(Aggregator(input_dim, output_dim, self.mess_dropout, self.aggregator_type))

        # multimodal
        self.dataset_name = config['dataset']
        self.dataset_path = config['data_path']

        self.tanh = nn.Tanh()
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.restore_user_e = None
        self.restore_entity_e = None
        
        self.contra_pairs_dis = None

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_entity_e', 'contra_pairs_dis']

        self.aggregator_layers_k = nn.ModuleList()
        for idx, (input_dim, output_dim) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.aggregator_layers_k.append(Aggregator(input_dim, output_dim, self.mess_dropout, self.aggregator_type))
        for param, param_k in zip(self.aggregator_layers.parameters(), self.aggregator_layers_k.parameters()):
            param_k.data.copy_(param.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue_u", torch.randn(self.dim, self.K))
        self.queue_u = nn.functional.normalize(self.queue_u, dim=0)

        self.register_buffer("queue_u_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_e", torch.randn(self.dim, self.K))
        self.queue_e = nn.functional.normalize(self.queue_e, dim=0)

        self.register_buffer("queue_e_ptr", torch.zeros(1, dtype=torch.long))
        
        self.dataset = dataset
        
        self.multi_step_cur = 0


    def init_graph(self):
        r"""Get the initial attention matrix through the collaborative knowledge graph

        Returns:
            torch.sparse.FloatTensor: Sparse tensor of the attention matrix
        """
        import dgl
        adj_list = []
        for rel_type in range(1, self.n_relations, 1):
            edge_idxs = self.ckg.filter_edges(lambda edge: edge.data['relation_id'] == rel_type)
            sub_graph = dgl.edge_subgraph(self.ckg, edge_idxs, preserve_nodes=True). \
                adjacency_matrix(transpose=False, scipy_fmt='coo').astype('float')
            rowsum = np.array(sub_graph.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(sub_graph).tocoo()
            adj_list.append(norm_adj)

        final_adj_matrix = sum(adj_list).tocoo()
        indices = torch.LongTensor([final_adj_matrix.row, final_adj_matrix.col])
        values = torch.FloatTensor(final_adj_matrix.data)
        adj_matrix_tensor = torch.sparse.FloatTensor(indices, values, self.matrix_size)
        return adj_matrix_tensor.to(self.device)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.aggregator_layers.parameters(), self.aggregator_layers_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue_u(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        # print("batch size:{}", batch_size)
        # print("queue shape:{}", self.queue.shape)

        ptr = int(self.queue_u_ptr)
        # print("ptr:{}", ptr)
        # assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size >= self.K:
            self.queue_u[:, ptr:self.K] = keys[:(self.K - ptr)].T
            self.queue_u[:, :(ptr + batch_size - self.K)] = keys[(self.K - ptr):].T
        else:
            self.queue_u[:, ptr:(ptr + batch_size)] = keys.T
        # out_ids = torch.arange(batch_size).cuda()
        # out_ids = torch.fmod(out_ids + ptr, self.K).long()
        # self.queue.index_copy_(1, out_ids, keys)

        ptr = (ptr + batch_size) % self.K  # move pointer
        # print("ptr:{}", ptr)

        self.queue_u_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_e(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        # print("batch size:{}", batch_size)
        # print("queue shape:{}", self.queue.shape)

        ptr = int(self.queue_e_ptr)
        # print("ptr:{}", ptr)
        # assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size >= self.K:
            self.queue_e[:, ptr:self.K] = keys[:(self.K - ptr)].T
            self.queue_e[:, :(ptr + batch_size - self.K)] = keys[(self.K - ptr):].T
        else:
            self.queue_e[:, ptr:(ptr + batch_size)] = keys.T
        # out_ids = torch.arange(batch_size).cuda()
        # out_ids = torch.fmod(out_ids + ptr, self.K).long()
        # self.queue.index_copy_(1, out_ids, keys)

        ptr = (ptr + batch_size) % self.K  # move pointer
        # print("ptr:{}", ptr)

        self.queue_e_ptr[0] = ptr

    def load_image_text_embeddings(self):
        # img_entity_path = os.path.join(self.dataset_path, f'{self.dataset_name}.iet')
        # des_entity_path = os.path.join(self.dataset_path, f'{self.dataset_name}.det')
        # entity_path = os.path.join(self.dataset_path, f'{self.dataset_name}.et')
        # img_path = os.path.join(self.dataset_path, 'image.pt')
        # text_path = os.path.join(self.dataset_path, 'text.pt')
        # i_emb_df = pandas.read_csv(img_entity_path)
        # d_emb_df = pandas.read_csv(des_entity_path)
        # emb_df = pandas.read_csv(entity_path)
        # i_tokens = i_emb_df['entity'].values.tolist()
        # d_tokens = d_emb_df['entity'].values.tolist()
        # tokens = emb_df['entity_id:token'].values
        # token2id_dict = dict(zip(tokens, self.dataset.token2id(field='entity_id', tokens=tokens)))
        # image_embs = torch.load(img_path).to(self.device)
        # text_embs = torch.load(text_path).to(self.device)
        image_embs = self.dataset.image_embs.to(self.device)
        text_embs = self.dataset.text_embs.to(self.device)
        image_embs = self.projection_head_map(image_embs, self.mode_multi_1, self.projection_head_multi_1)
        text_embs = self.projection_head_map(text_embs, self.mode_multi_2, self.projection_head_multi_2)
        self.mode_multi_1 = 1 - self.mode_multi_1
        self.mode_multi_2 = 1 - self.mode_multi_2

        # img_ids = []
        # text_ids = []
        # for t in i_tokens:
        #     img_ids.append(token2id_dict[t])
        # for t in d_tokens:
        #     text_ids.append(token2id_dict[t])
            
        image_embedding = nn.Embedding(self.n_entities, self.embedding_size).to(self.device)
        # constant_(image_embedding.weight.data, 0)
        xavier_normal_(image_embedding.weight.data)
        text_embedding = nn.Embedding(self.n_entities, self.embedding_size).to(self.device)
        # constant_(text_embedding.weight.data, 0)
        xavier_normal_(text_embedding.weight.data)
        for param_i in image_embedding.parameters():
            param_i.requires_grad = False  # not update by gradient
        for param_t in text_embedding.parameters():
            param_t.requires_grad = False
            
        img_ids = self.dataset.img_ids
        text_ids = self.dataset.text_ids

        if self.use_image:
            image_embedding.weight[img_ids] = image_embs
        if self.use_text:
            text_embedding.weight[text_ids] = text_embs
        
        return image_embedding.weight, text_embedding.weight


    def _get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        entity_embeddings = self.entity_embedding.weight
        
        if self.multi_step_cur < self.multi_step:
            image_embeddings, text_embeddings = self.load_image_text_embeddings()
            # multi_embs = torch.cat((image_embeddings, text_embeddings), 1).to(self.device)
            # multi_embs = self.projection_head_map(multi_embs, self.mode_uni_1, self.projection_head_uni_1)
            # self.mode_uni_1 = 1 - self.mode_uni_1
            uni_embs = torch.cat((entity_embeddings, image_embeddings, text_embeddings), 1).to(self.device)
            entity_embeddings = self.projection_head_map(uni_embs, self.mode_uni_2, self.projection_head_uni_2)
            self.mode_uni_2 = 1 - self.mode_uni_2
            self.multi_step_cur += 1
        
        ego_embeddings = torch.cat([user_embeddings, entity_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        ego_embeddings = self._get_ego_embeddings()
        embeddings_list = [ego_embeddings]
        for aggregator in self.aggregator_layers:
            ego_embeddings = aggregator(self.A_in, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
        user_all_embeddings, entity_all_embeddings = torch.split(kgat_all_embeddings, [self.n_users, self.n_entities])
        return user_all_embeddings, entity_all_embeddings

    def forward_1(self):
        ego_embeddings = self._get_ego_embeddings()
        embeddings_list = [ego_embeddings]
        for aggregator in self.aggregator_layers:
            ego_embeddings = aggregator(self.A_in_1, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            # x = 0.01 + torch.zeros_like(norm_embeddings,
            #                             dtype=torch.float32, device=norm_embeddings.device)  # add guassian noise
            # noise = torch.normal(mean=torch.tensor([0.0]).to(norm_embeddings.device), std=x).to(
            #     norm_embeddings.device)  # add guassian noise.
            # norm_embeddings += noise
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
        user_all_embeddings, entity_all_embeddings = torch.split(kgat_all_embeddings, [self.n_users, self.n_entities])
        return user_all_embeddings, entity_all_embeddings

    def forward_2(self):
        ego_embeddings = self._get_ego_embeddings()
        embeddings_list = [ego_embeddings]
        for aggregator in self.aggregator_layers:
            ego_embeddings = aggregator(self.A_in_2, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            # x = 0.01 + torch.zeros_like(norm_embeddings,
            #                             dtype=torch.float32, device=norm_embeddings.device)  # add guassian noise
            # noise = torch.normal(mean=torch.tensor([0.0]).to(norm_embeddings.device), std=x).to(
            #     norm_embeddings.device)  # add guassian noise.
            # norm_embeddings += noise
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
        user_all_embeddings, entity_all_embeddings = torch.split(kgat_all_embeddings, [self.n_users, self.n_entities])
        return user_all_embeddings, entity_all_embeddings

    def forward_k(self):
        ego_embeddings = self._get_ego_embeddings()
        embeddings_list = [ego_embeddings]
        for aggregator in self.aggregator_layers_k:
            ego_embeddings = aggregator(self.A_in_2, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            # x = 0.01 + torch.zeros_like(norm_embeddings,
            #                             dtype=torch.float32, device=norm_embeddings.device)  # add guassian noise
            # noise = torch.normal(mean=torch.tensor([0.0]).to(norm_embeddings.device), std=x).to(
            #     norm_embeddings.device)  # add guassian noise.
            # norm_embeddings += noise
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
        user_all_embeddings, entity_all_embeddings = torch.split(kgat_all_embeddings, [self.n_users, self.n_entities])
        return user_all_embeddings, entity_all_embeddings

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def _get_kg_embedding(self, h, r, pos_t, neg_t):
        h_e = self.entity_embedding(h).unsqueeze(1)
        pos_t_e = self.entity_embedding(pos_t).unsqueeze(1)
        neg_t_e = self.entity_embedding(neg_t).unsqueeze(1)
        r_e = self.relation_embedding(r)
        r_trans_w = self.trans_w(r).view(r.size(0), self.embedding_size, self.kg_embedding_size)

        h_e = torch.bmm(h_e, r_trans_w).squeeze(1)
        pos_t_e = torch.bmm(pos_t_e, r_trans_w).squeeze(1)
        neg_t_e = torch.bmm(neg_t_e, r_trans_w).squeeze(1)

        return h_e, r_e, pos_t_e, neg_t_e

    def cts_loss(self, z_i, z_j, temp, batch_size):  # B * D    B * D

        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)  # 2B * D

        sim = torch.mm(z, z.T) / temp  # 2B * 2B

        sim_i_j = torch.diag(sim, batch_size)  # B*1
        sim_j_i = torch.diag(sim, -batch_size)  # B*1

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        mask = self.mask_correlated_samples(batch_size)

        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)  # N * C
        loss = self.ce_loss(logits, labels)
        return loss

    def moco_loss(self, q, k, u=True):
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # print("q's shape:{}, k's shape(): {}, l_pos's shape(): {}", q.shape, k.shape, l_pos.shape)
        # negative logits: NxK
        if u:
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue_u.clone().detach()])
        else:
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue_e.clone().detach()])
        # print("queue's shape: {}, l_neg's shape: {}", self.queue.shape, l_neg.shape)

        weights = torch.where(l_neg > self.phi, 0, 1)

        l_neg = l_neg * weights

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        if u:
            self._dequeue_and_enqueue_u(k)
        else:
            self._dequeue_and_enqueue_e(k)

        loss = self.ce_loss(logits, labels)
        return loss

    def projection_head_map(self, state, mode, projection_head):
        for i, l in enumerate(projection_head):  # 0: Linear 1: BN (relu)  2: Linear 3:BN (relu)
            if i % 2 != 0:
                if mode == 0:
                    l.train()  # set BN to train mode: use a learned mean and variance.
                else:
                    l.eval()  # set BN to eval mode: use a accumulated mean and variance.
            state = l(state)
            if i % 2 != 0:
                state = F.relu(state)
        return state

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None
            
        if self.contra_pairs_dis is not None:
            self.contra_pairs_dis = None

        # get loss for training rs
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, entity_all_embeddings = self.forward()
        kgat_all_embeddings = torch.cat((user_all_embeddings, entity_all_embeddings), 0)

        user_all_embeddings_1, entity_all_embeddings_1 = self.forward_1()
        user_all_embeddings_2, entity_all_embeddings_2 = self.forward_2()

        user_all_embeddings_q, entity_all_embeddings_q = self.forward_1()
        with torch.no_grad():
            self._momentum_update_key_encoder()
            user_all_embeddings_k, entity_all_embeddings_k = self.forward_k()

        user_rand_samples = self.rand_sample(user_all_embeddings_1.shape[0], size=user.shape[0] // 8, replace=False)
        entity_rand_samples = self.rand_sample(entity_all_embeddings_1.shape[0], size=user.shape[0], replace=False)

        user_rand_samples2 = self.rand_sample(user_all_embeddings_1.shape[0], size=user.shape[0] // 8, replace=False)
        entity_rand_samples2 = self.rand_sample(entity_all_embeddings_1.shape[0], size=user.shape[0], replace=False)

        cts_embedding_1 = user_all_embeddings_1[torch.tensor(user_rand_samples)]
        cts_embedding_2 = user_all_embeddings_2[torch.tensor(user_rand_samples)]

        e_cts_embedding_1 = entity_all_embeddings_1[torch.tensor(entity_rand_samples)]
        e_cts_embedding_2 = entity_all_embeddings_2[torch.tensor(entity_rand_samples)]

        cts_embedding_q = user_all_embeddings_q[torch.tensor(user_rand_samples2)]
        cts_embedding_k = user_all_embeddings_k[torch.tensor(user_rand_samples2)]

        e_cts_embedding_q = entity_all_embeddings_q[torch.tensor(entity_rand_samples2)]
        e_cts_embedding_k = entity_all_embeddings_k[torch.tensor(entity_rand_samples2)]

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = entity_all_embeddings[pos_item]
        neg_embeddings = entity_all_embeddings[neg_item]

        cts_embedding_1 = self.projection_head_map(cts_embedding_1, self.mode, self.projection_head)
        cts_embedding_2 = self.projection_head_map(cts_embedding_2, 1 - self.mode, self.projection_head)
        e_cts_embedding_1 = self.projection_head_map(e_cts_embedding_1, self.mode, self.projection_head)
        e_cts_embedding_2 = self.projection_head_map(e_cts_embedding_2, 1 - self.mode, self.projection_head)

        cts_embedding_q = self.projection_head_map(cts_embedding_q, self.mode, self.projection_head)
        cts_embedding_k = self.projection_head_map(cts_embedding_k, 1 - self.mode, self.projection_head)
        e_cts_embedding_q = self.projection_head_map(e_cts_embedding_q, self.mode, self.projection_head)
        e_cts_embedding_k = self.projection_head_map(e_cts_embedding_k, 1 - self.mode, self.projection_head)
        
        with torch.no_grad():
            self.contra_pairs_dis = F.pairwise_distance(e_cts_embedding_q, e_cts_embedding_k, p=2)
            self.contra_pairs_dis.requires_grad = False

        u_embeddings = self.projection_head_map(u_embeddings, self.mode, self.projection_head)
        pos_embeddings = self.projection_head_map(pos_embeddings, 1 - self.mode, self.projection_head)

        self.mode = 1 - self.mode

        cts_loss = self.cts_loss(cts_embedding_1, cts_embedding_2, temp=1.0,
                                 batch_size=cts_embedding_1.shape[0])

        e_cts_loss = self.cts_loss(e_cts_embedding_1, e_cts_embedding_2, temp=1.0,
                                   batch_size=e_cts_embedding_1.shape[0])

        ui_cts_loss = self.cts_loss(u_embeddings, pos_embeddings, temp=1.0,
                                    batch_size=u_embeddings.shape[0])

        cts_moco_loss = self.moco_loss(cts_embedding_1, cts_embedding_2, u=True)
        e_cts_moco_loss = self.moco_loss(e_cts_embedding_1, e_cts_embedding_2, u=False)

        #        cts_loss_1 = self.cts_loss(cts_embedding, cts_embedding_1, temp=0.1,
        #                                                        batch_size=cts_embedding_1.shape[0])
        #        cts_loss_2 = self.cts_loss(cts_embedding, cts_embedding_2, temp=0.1,
        #                                                        batch_size=cts_embedding_1.shape[0])

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = entity_all_embeddings[pos_item]
        neg_embeddings = entity_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)
        reg_loss = self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)
        #        print("cts_loss:", cts_loss, e_cts_loss, ui_cts_loss)
        loss = mf_loss + self.reg_weight * reg_loss + 0.01 * (
                    cts_loss + e_cts_loss + ui_cts_loss + cts_moco_loss + e_cts_moco_loss)
        return loss

    def calculate_kg_loss(self, interaction):
        r"""Calculate the training loss for a batch data of KG.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """

        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        # get loss for training kg
        h = interaction[self.HEAD_ENTITY_ID]
        r = interaction[self.RELATION_ID]
        pos_t = interaction[self.TAIL_ENTITY_ID]
        neg_t = interaction[self.NEG_TAIL_ENTITY_ID]

        h_e, r_e, pos_t_e, neg_t_e = self._get_kg_embedding(h, r, pos_t, neg_t)
        pos_tail_score = ((h_e + r_e - pos_t_e) ** 2).sum(dim=1)
        neg_tail_score = ((h_e + r_e - neg_t_e) ** 2).sum(dim=1)
        kg_loss = F.softplus(pos_tail_score - neg_tail_score).mean()
        kg_reg_loss = self.reg_loss(h_e, r_e, pos_t_e, neg_t_e)
        loss = kg_loss + self.reg_weight * kg_reg_loss

        return loss

    def generate_transE_score(self, hs, ts, r):
        r"""Calculating scores for triples in KG.
            对应kgat公式4,5

        Args:
            hs (torch.Tensor): head entities
            ts (torch.Tensor): tail entities
            r (int): the relation id between hs and ts

        Returns:
            torch.Tensor: the scores of (hs, r, ts)
        """

        all_embeddings = self._get_ego_embeddings()
        image_embeddings, text_embeddings = self.load_image_text_embeddings()
        # image_embeddings, text_embeddings = self.image_embedding, self.text_embedding
        shs = hs - self.n_users
        sts = ts - self.n_users
        h_e = all_embeddings[hs]
        t_e = all_embeddings[ts]
        hi_e = image_embeddings[shs]
        ti_e = image_embeddings[sts]
        ht_e = text_embeddings[shs]
        tt_e = text_embeddings[sts]
        r_e = self.relation_embedding.weight[r]
        r_trans_w = self.trans_w.weight[r].view(self.embedding_size, self.kg_embedding_size)
        # r_i_trans_w = self.i_trans_w.weight[r].view(self.embedding_size, self.kg_embedding_size)
        # r_t_trans_w = self.t_trans_w.weight[r].view(self.embedding_size, self.kg_embedding_size)

        h_e = torch.matmul(h_e, r_trans_w)
        t_e = torch.matmul(t_e, r_trans_w)
        # trans 1,2,3,10
        # hi_e = torch.matmul(hi_e, r_trans_w)
        # ti_e = torch.matmul(ti_e, r_trans_w)
        # ht_e = torch.matmul(ht_e, r_trans_w)
        # tt_e = torch.matmul(tt_e, r_trans_w)
        # trans 4,5,6,11
        # hi_e = torch.matmul(hi_e, r_i_trans_w)
        # ti_e = torch.matmul(ti_e, r_i_trans_w)
        # ht_e = torch.matmul(ht_e, r_t_trans_w)
        # tt_e = torch.matmul(tt_e, r_t_trans_w)
        # trans 7,8,9,12 no trans
        
        # kg_score = torch.mul(t_e, self.tanh(h_e + r_e)).sum(dim=1)

        # kg_score = torch.mul(t_e, self.tanh(h_e + r_e) + torch.mul(hi_e, tt_e) + torch.mul(ht_e, ti_e)).sum(dim=1) # trans1,4,7
        # kg_score = torch.mul(t_e, self.tanh(h_e + r_e)).sum(dim=1) + torch.mul(ti_e, self.tanh(hi_e + r_e)).sum(dim=1) + torch.mul(tt_e, self.tanh(ht_e + r_e)).sum(dim=1) # trans2,5,8
        # kg_score = torch.mul(t_e, self.tanh(h_e + r_e)).sum(dim=1) + torch.mul(ti_e, self.tanh(hi_e + r_e)).sum(dim=1) + torch.mul(tt_e, self.tanh(ht_e + r_e)).sum(dim=1) + torch.mul(hi_e, tt_e).sum(dim=1) + torch.mul(ht_e, ti_e).sum(dim=1) # trans3,6,9
        if self.use_att:
            if self.att_type == 1:
                kg_score = torch.mul(t_e, self.tanh(h_e + r_e) + torch.mul(hi_e, tt_e) + torch.mul(ht_e, ti_e)).sum(dim=1) + torch.mul(ti_e, self.tanh(hi_e + r_e)).sum(dim=1) + torch.mul(tt_e, self.tanh(ht_e + r_e)).sum(dim=1) # trans10,11,12
            elif self.att_type == 2:
                kg_score = torch.mul(t_e, self.tanh(h_e + r_e) + torch.mul(hi_e, tt_e) + torch.mul(ht_e, ti_e)).sum(dim=1)
            else:
                kg_score = torch.mul(t_e, self.tanh(h_e + r_e)).sum(dim=1) + torch.mul(ti_e, self.tanh(hi_e + r_e)).sum(dim=1) + torch.mul(tt_e, self.tanh(ht_e + r_e)).sum(dim=1)
        else:
            kg_score = torch.mul(t_e, self.tanh(h_e + r_e)).sum(dim=1)

        return kg_score

    def rand_sample(self, high, size=None, replace=True):
        r"""Randomly discard some points or edges.

        Args:
            high (int): Upper limit of index value
            size (int): Array size after sampling

        Returns:
            numpy.ndarray: Array index after sampling, shape: [size]
        """

        a = np.arange(high)
        sample = np.random.choice(a, size=size, replace=replace)
        return sample

    def update_attentive_A(self):
        r"""Update the attention matrix using the updated embedding matrix

        """

        kg_score_list, row_list, col_list = [], [], []
        # To reduce the GPU memory consumption, we calculate the scores of KG triples according to the type of relation
        for rel_idx in range(1, self.n_relations, 1):
            triple_index = torch.where(self.all_rs == rel_idx)
            kg_score = self.generate_transE_score(self.all_hs[triple_index], self.all_ts[triple_index], rel_idx)
            row_list.append(self.all_hs[triple_index])
            col_list.append(self.all_ts[triple_index])
            kg_score_list.append(kg_score)
        kg_score = torch.cat(kg_score_list, dim=0)
        row = torch.cat(row_list, dim=0)
        col = torch.cat(col_list, dim=0)
        indices = torch.cat([row, col], dim=0).view(2, -1)
        # Current PyTorch version does not support softmax on SparseCUDA, temporarily move to CPU to calculate softmax
        A_in = torch.sparse.FloatTensor(indices, kg_score, self.matrix_size).cpu()
        A_in = torch.sparse.softmax(A_in, dim=1).to(self.device)

        drop_edge_1 = self.rand_sample(indices.shape[1], size=int(indices.shape[1] * 0.1), replace=False)
        indices_1 = indices.view(-1, 2)[torch.tensor(drop_edge_1)].view(2, -1)
        kg_score_1 = kg_score[torch.tensor(drop_edge_1)]
        A_in_1 = torch.sparse.FloatTensor(indices_1, kg_score_1, self.matrix_size).cpu()
        A_in_1 = torch.sparse.softmax(A_in_1, dim=1).to(self.device)

        drop_edge_2 = self.rand_sample(indices.shape[1], size=int(indices.shape[1] * 0.1), replace=False)
        indices_2 = indices.view(-1, 2)[torch.tensor(drop_edge_2)].view(2, -1)
        kg_score_2 = kg_score[torch.tensor(drop_edge_2)]
        A_in_2 = torch.sparse.FloatTensor(indices_2, kg_score_2, self.matrix_size).cpu()
        A_in_2 = torch.sparse.softmax(A_in_2, dim=1).to(self.device)

        self.A_in = A_in
        self.A_in_1 = A_in_1
        self.A_in_2 = A_in_2

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, entity_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = entity_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_entity_e is None:
            self.restore_user_e, self.restore_entity_e = self.forward()
        u_embeddings = self.restore_user_e[user]
        i_embeddings = self.restore_entity_e[:self.n_items]

        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))

        return scores.view(-1)
