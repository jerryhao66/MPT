'''
create on Oct 12, 2020
tensorflow implemention of graph neural networks
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = 'haobowen'

import warnings

warnings.filterwarnings("ignore")
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
import numpy as np
from tensorflow.python.client import device_lib
from utility.helper import *
from utility.batch_test import *
from utility.nce_loss import nt_xent
import scipy.sparse as sp
import logging
import tqdm
from scipy.stats import pearsonr
from trainer import *
from utility import metrics
import heapq
from utility.Config import Model_Config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
cpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'CPU']


class Recommender(object):
    def __init__(self, data_config, cldel_config, clrep_config, pretrain_data, initial_data, conv_name, Config,
                 is_training, scope=None):
        # gnn parameters
        # self.model_type = 'LightGCN'
        self.Config = Config
        self.data_config = data_config
        self.cldel_config = cldel_config  # contrastive learning deletion
        self.clrep_config = clrep_config  # contrastive learning replace
        self.adj_type = Config.adj_type
        # self.alg_type = Config.alg_type
        self.dataset = Config.dataset
        self.pretrain_data = pretrain_data
        self.initial_data = initial_data
        self.n_users = self.data_config['n_users']
        self.n_items = self.data_config['n_items']
        self.n_fold = 100
        self.norm_adj = self.data_config['norm_adj']
        self.norm_adj_cld = self.cldel_config['norm_adj']
        self.norm_adj_clr = self.clrep_config['norm_adj']
        self.lr = Config.lr
        self.emb_dim = Config.embed_size
        self.batch_size = Config.batch_size
        self.weight_size = eval(Config.layer_size)
        self.weight_size_single = eval(Config.layer_size1)
        self.n_layers = len(self.weight_size)
        self.regs = eval(Config.regs)
        self.decay = self.regs[0]
        self.verbose = Config.verbose
        # self.Ks = eval(Config.Ks)
        self.Ks = Config.Ks
        self.conv_name = str(conv_name)
        self.num_users = Config.num_users
        self.num_items = Config.num_items
        # transformer encoder structure
        self.dropout_rate = Config.dropout_rate
        self.num_heads = Config.num_heads
        self.d_ff = Config.d_ff
        self.num_blocks = Config.num_blocks
        self.vocab_size = Config.vocab_size
        # bulid training meta embedding structure
        self.create_placeholders_pretext_tasks()
        self.weights = self._init_weights()


        ############# step 1 embedding reconstruction task ##########
        ##### train meta aggregator ####
        self.train_user_meta_aggregator('user')
        self.train_item_meta_aggregator('item')
        ##### train embedding reconstruction ####
        self.target_user_ebd = self.pretrain_data['user_embed']
        self.target_item_ebd = self.pretrain_data['item_embed']
        self.user_embedding_reconstruct()
        self.item_embedding_reconstruct()

        self.ua_embeddings, self.ia_embeddings = self.GeneralGCN(n_layers=self.n_layers, option='meta')

        ############ embedding contrastive learning task ###########
        self.un_embeddins, self.in_embeddings = self.GeneralGCN(n_layers=self.n_layers, option='normal')
        self.uanchor_embeddings, self.ianchor_embeddings = self.GeneralGCN(n_layers=self.n_layers, option='anchor')
        self.udelete_embeddings, self.idelete_embeddings = self.GeneralGCN(n_layers=self.n_layers,
                                                                           option='contrastive_delete')
        self.ureplace_embeddings, self.ireplace_embeddings = self.GeneralGCN(n_layers=self.n_layers,
                                                                             option='contrastive_replace')

        self.user_embedding_contrastive()
        self.item_embedding_contrastive()

        ########### embedding reconstruction with Transformer #####
        self.Transformer_reconstruct()
        self.Transformer_contrastive()

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """

        # self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        # self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        # self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)
        self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)

        # # only embedding reconstruction with GNN
        # self.u_g_embeddings = tf.nn.embedding_lookup( tf.concat([self.user_reconstruct_ebd, self.ua_embeddings], axis=1), self.users)
        # self.pos_i_g_embeddings = tf.nn.embedding_lookup(tf.concat([self.item_reconstruct_ebd, self.ia_embeddings], axis=1), self.pos_items)
        # self.neg_i_g_embeddings = tf.nn.embedding_lookup(tf.concat([self.item_reconstruct_ebd, self.ia_embeddings], axis=1), self.neg_items)

        # # only embedding contrastive with GNN
        # self.u_g_embeddings = tf.nn.embedding_lookup(
        #     tf.concat([self.uanchor_embeddings, self.udelete_embeddings, self.ureplace_embeddings], axis=1), self.users)
        # self.pos_i_g_embeddings = tf.nn.embedding_lookup(
        #     tf.concat([self.ianchor_embeddings, self.idelete_embeddings, self.ireplace_embeddings], axis=1),
        #     self.pos_items)
        # self.neg_i_g_embeddings = tf.nn.embedding_lookup(
        #     tf.concat([self.ianchor_embeddings, self.idelete_embeddings, self.ireplace_embeddings], axis=1),
        #     self.neg_items)

        # # Reconstruction and Contrastive Learning with GNNs
        # self.u_g_embeddings = tf.nn.embedding_lookup(
        #     tf.concat([self.ua_embeddings, self.user_reconstruct_ebd, self.uanchor_embeddings, self.udelete_embeddings, self.ureplace_embeddings], axis=1), self.users)
        # self.pos_i_g_embeddings = tf.nn.embedding_lookup(
        #     tf.concat([self.ia_embeddings, self.item_reconstruct_ebd, self.ianchor_embeddings, self.idelete_embeddings, self.ireplace_embeddings], axis=1),
        #     self.pos_items)
        # self.neg_i_g_embeddings = tf.nn.embedding_lookup(
        #     tf.concat([self.ia_embeddings, self.item_reconstruct_ebd, self.ianchor_embeddings, self.idelete_embeddings, self.ireplace_embeddings], axis=1),
        #     self.neg_items)

        # # only embedding reconstruction with Transofmer
        # self.u_g_embeddings = tf.nn.embedding_lookup( tf.concat([self.weights['bert_r_user_embedding'], self.un_embeddins], axis=1), self.users)
        # self.pos_i_g_embeddings = tf.nn.embedding_lookup(tf.concat([self.weights['bert_r_item_embedding'], self.in_embeddings], axis=1), self.pos_items)
        # self.neg_i_g_embeddings = tf.nn.embedding_lookup(tf.concat([self.weights['bert_r_item_embedding'], self.in_embeddings], axis=1), self.neg_items)


        # Reconstruction with GNN and Transformer + Contrastive learning with GNNs

        # self.u_g_embeddings = tf.nn.embedding_lookup(
        #     tf.concat([self.ua_embeddings, self.user_reconstruct_ebd, self.uanchor_embeddings, self.udelete_embeddings, self.ureplace_embeddings, self.weights['bert_r_user_embedding']], axis=1), self.users)
        # self.pos_i_g_embeddings = tf.nn.embedding_lookup(
        #     tf.concat([self.ia_embeddings, self.item_reconstruct_ebd, self.ianchor_embeddings, self.idelete_embeddings, self.ireplace_embeddings, self.weights['bert_r_item_embedding']], axis=1),
        #     self.pos_items)
        # self.neg_i_g_embeddings = tf.nn.embedding_lookup(
        #     tf.concat([self.ia_embeddings, self.item_reconstruct_ebd, self.ianchor_embeddings, self.idelete_embeddings, self.ireplace_embeddings, self.weights['bert_r_item_embedding']], axis=1),
        #     self.neg_items)

        # all
        self.u_g_embeddings = tf.nn.embedding_lookup(
            tf.concat([self.ua_embeddings, self.user_reconstruct_ebd, self.uanchor_embeddings, self.udelete_embeddings, self.ureplace_embeddings, self.weights['bert_r_user_embedding'], self.weights['bert_c_user_embedding']], axis=1), self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(
            tf.concat([self.ia_embeddings, self.item_reconstruct_ebd, self.ianchor_embeddings, self.idelete_embeddings, self.ireplace_embeddings, self.weights['bert_r_item_embedding'], self.weights['bert_c_item_embedding']], axis=1),
            self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(
            tf.concat([self.ia_embeddings, self.item_reconstruct_ebd, self.ianchor_embeddings, self.idelete_embeddings, self.ireplace_embeddings, self.weights['bert_r_item_embedding'], self.weights['bert_c_item_embedding']], axis=1),
            self.neg_items)

        """
        *********************************************************
        Inference for the testing phase.
        """
        self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False,
                                       transpose_b=True)

        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings,
                                                                          self.u_g_embeddings_pre,
                                                                          self.pos_i_g_embeddings_pre,
                                                                          self.neg_i_g_embeddings_pre)
        self.loss = self.mf_loss + self.emb_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def Transformer_reconstruct(self):
        self.input_ids = tf.placeholder(tf.int32, shape=[None, None])  # [b, n]
        self.masked_positions = tf.placeholder(tf.int32, shape=[None, None])  # [b]
        self.masked_ids = tf.placeholder(tf.int32, shape=[None])  # [b, n]
        self.gen_ebd_mask_position = tf.placeholder(tf.int32)

        self.embedding_table = self.weights['bert_embedding']

        self.target_embedding_table = self.pretrain_data['bert_embed']

        self.input_ebd = tf.nn.embedding_lookup(self.embedding_table, self.input_ids)  # [b, n, e]
        self.encode_ebd = self._encode_mask_lm(self.input_ebd)  # [b, n, e]

        self.mask_generate_ebd = tf.squeeze(
            self.encode_ebd[:, self.gen_ebd_mask_position:self.gen_ebd_mask_position + 1, :], axis=1)  # [b, e]
        self.bert_batch_target_ebd = tf.nn.embedding_lookup(self.target_embedding_table, self.masked_ids)  # [b, e]

        self.bert_batch_loss_reconstruct = -tf.reduce_mean(
            Cosine_similarity(tf.cast(self.mask_generate_ebd, tf.float32), tf.cast(self.bert_batch_target_ebd, tf.float32)))

        self.optimizer_reconstruct = tf.train.AdagradOptimizer(learning_rate=self.lr,
                                                               initial_accumulator_value=1e-8).minimize(
            self.bert_batch_loss_reconstruct)


    def Transformer_contrastive(self):
        self.cl_input_ids1 = tf.placeholder(tf.int32, shape=[None, None])  # [b, n]
        self.cl_masked_positions1 = tf.placeholder(tf.int32, shape=[None, None])  # [b]
        self.cl_masked_ids1 = tf.placeholder(tf.int32, shape=[None])  # [b, n]
        self.cl_gen_ebd_mask_position1 = tf.placeholder(tf.int32)

        self.cl_input_ids2 = tf.placeholder(tf.int32, shape=[None, None])  # [b, n]
        self.cl_masked_positions2 = tf.placeholder(tf.int32, shape=[None, None])  # [b]
        self.cl_masked_ids2 = tf.placeholder(tf.int32, shape=[None])  # [b, n]
        self.cl_gen_ebd_mask_position2 = tf.placeholder(tf.int32)


        cl_input_ebd1 = tf.nn.embedding_lookup(self.embedding_table, self.cl_input_ids1)  # [b, n, e]
        encode_ebd1 = self._encode_contrastive(cl_input_ebd1)  # [b, n, e]

        cl_input_ebd2 = tf.nn.embedding_lookup(self.embedding_table, self.cl_input_ids2)  # [b, n, e]
        encode_ebd2 = self._encode_contrastive(cl_input_ebd2)  # [b, n, e]

        z1 = tf.squeeze(encode_ebd1[:, self.cl_gen_ebd_mask_position1:self.cl_gen_ebd_mask_position1 + 1, :], axis=1)  # [b, e]
        z2 = tf.squeeze(encode_ebd2[:, self.cl_gen_ebd_mask_position2:self.cl_gen_ebd_mask_position2 + 1, :], axis=1)  # [b, e]
        self.contrastive_transformer_loss = nt_xent(z1, z2, self.Config.batch_size_tranformer, self.Config.temperature, self.emb_dim)
        self.optimizer_contrastive_bert_task = tf.train.AdagradOptimizer(learning_rate=self.lr,
                                                                         initial_accumulator_value=1e-8).minimize(
            self.contrastive_transformer_loss)

    def user_embedding_contrastive(self):
        user_aug_ebd1 = tf.nn.embedding_lookup(self.udelete_embeddings, self.cl_pos_user)  # [b, n1, 3e]
        user_aug_ebd2 = tf.nn.embedding_lookup(self.ureplace_embeddings, self.cl_pos_user)  # [b, n2, 3e]
        z1 = tf.contrib.layers.fully_connected(user_aug_ebd1, self.emb_dim, tf.nn.leaky_relu)
        z2 = tf.contrib.layers.fully_connected(user_aug_ebd2, self.emb_dim, tf.nn.leaky_relu)
        self.user_contrastive_loss = nt_xent(z1, z2, self.Config.batch_size, self.Config.temperature, self.emb_dim)
        self.optimizer_user_contrastive_task = tf.train.AdagradOptimizer(learning_rate=self.lr,
                                                                         initial_accumulator_value=1e-8).minimize(
            self.user_contrastive_loss)

    def item_embedding_contrastive(self):
        item_aug_ebd1 = tf.nn.embedding_lookup(self.idelete_embeddings, self.cl_pos_item)  # [b, n1, 3e]
        item_aug_ebd2 = tf.nn.embedding_lookup(self.ireplace_embeddings, self.cl_pos_item)  # [b, n2, 3e]
        z1 = tf.contrib.layers.fully_connected(item_aug_ebd1, self.emb_dim, tf.nn.leaky_relu)
        z2 = tf.contrib.layers.fully_connected(item_aug_ebd2, self.emb_dim, tf.nn.leaky_relu)
        self.item_contrastive_loss = nt_xent(z1, z2, self.Config.batch_size, self.Config.temperature, self.emb_dim)
        self.optimizer_item_contrastive_task = tf.train.AdagradOptimizer(learning_rate=self.lr,
                                                                         initial_accumulator_value=1e-8).minimize(
            self.item_contrastive_loss)


    def _encode(self, input, training=True):
        '''
        input: [b, n, e]
        output: [b, n, e]
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            enc = input
            enc *= self.emb_dim ** 0.5

            ## Blocks
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.dropout_rate,
                                              training=training,
                                              causality=False)  # [b, q, e]
                    # feed forward
                    enc = ff(enc, num_units=[self.d_ff, self.emb_dim])  # [b, q, e]
        memory = enc
        return memory


    def _encode_mask_lm(self, input, training=True):
        '''
        input: [b, n, e]
        output: [b, n, e]
        '''
        with tf.variable_scope("Transformer_Encoder"):
            enc = input
            enc *= self.emb_dim ** 0.5

            ## Blocks
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.dropout_rate,
                                              training=training,
                                              causality=False)  # [b, q, e]
                    # feed forward
                    enc = ff(enc, num_units=[self.d_ff, self.emb_dim])  # [b, q, e]
        memory = enc
        return memory

    def _encode_contrastive(self, input, training=True):

        with tf.variable_scope("Transformer_CL_Encoder", reuse=tf.AUTO_REUSE):
            enc = input
            enc *= self.emb_dim ** 0.5
            ## Blocks
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.dropout_rate,
                                              training=training,
                                              causality=False)  # [b, q, e]
                    # feed forward
                    enc = ff(enc, num_units=[self.d_ff, self.emb_dim])  # [b, q, e]
        memory = enc
        return memory

    def train_user_meta_aggregator(self, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.c1 = tf.constant(0.0, tf.float32, [1, self.emb_dim], name='c2')
            self.context_embedding_i = tf.concat([self.weights['item_embedding'], self.c1], 0, name='embedding_item')
            self.support_ebd = tf.nn.embedding_lookup(self.context_embedding_i, self.support_item)  # [b, n, e]
            self.support_encode_user = self._encode(self.support_ebd,
                                                    training=self.training_phrase_user_task)  # [b, n, e]
            self.final_support_encode_user_task = tf.reduce_mean(self.support_encode_user,
                                                                 axis=1)  # [b, n, e] -> [b, e]
            self.batch_loss_user_task = Cosine_similarity(self.final_support_encode_user_task, self.target_user)

            self.loss_user_task = -tf.reduce_mean(
                Cosine_similarity(self.final_support_encode_user_task, self.target_user))

            self.optimizer_user_task = tf.train.AdagradOptimizer(learning_rate=self.lr,
                                                                 initial_accumulator_value=1e-8).minimize(
                self.loss_user_task)

    def train_item_meta_aggregator(self, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.c2 = tf.constant(0.0, tf.float32, [1, self.emb_dim], name='c2')
            self.context_embedding_u = tf.concat([self.weights['user_embedding'], self.c2], 0, name='embedding_user')
            self.support_ebd = tf.nn.embedding_lookup(self.context_embedding_u, self.support_user)  # [b, n, e]
            self.support_encode_item = self._encode(self.support_ebd,
                                                    training=self.training_phrase_item_task)  # [b, n, e]
            self.final_support_encode_item_task = tf.reduce_mean(self.support_encode_item,
                                                                 axis=1)  # [b, n, e] -> [b, e]
            self.batch_loss_item_task = Cosine_similarity(self.final_support_encode_item_task, self.target_item)

            self.loss_item_task = -tf.reduce_mean(
                Cosine_similarity(self.final_support_encode_item_task, self.target_item))

            self.optimizer_item_task = tf.train.AdagradOptimizer(learning_rate=self.lr,
                                                                 initial_accumulator_value=1e-8).minimize(
                self.loss_item_task)


    def create_placeholders_pretext_tasks(self):
        #################### embedding reconstruction with GNN #######################
        # user task
        self.support_item = tf.placeholder(tf.int32, shape=[None, None])
        self.target_user = tf.placeholder(tf.float32, shape=[None, self.emb_dim])
        self.training_phrase_user_task = tf.placeholder(tf.bool, name='training-flag')
        self.test_support_item = tf.placeholder(tf.int32, shape=[1, None])

        # item task
        self.support_user = tf.placeholder(tf.int32, shape=[None, None])
        self.target_item = tf.placeholder(tf.float32, shape=[None, self.emb_dim])
        self.training_phrase_item_task = tf.placeholder(tf.bool, name='training-flag')
        self.test_support_user = tf.placeholder(tf.int32, shape=[1, None])

        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        self.node_dropout_flag = False
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

        #################### embedding reconstruction with GNN #######################

        # # contrastive user
        self.cl_anchor_user = tf.placeholder(tf.int32, shape=[None, None])
        self.cl_pos_user = tf.placeholder(tf.int32, shape=[None, None])
        self.cl_neg_user = tf.placeholder(tf.int32, shape=[None, None])

        # # contrastive item
        self.cl_anchor_item = tf.placeholder(tf.int32, shape=[None, None])
        self.cl_pos_item = tf.placeholder(tf.int32, shape=[None, None])
        self.cl_neg_item = tf.placeholder(tf.int32, shape=[None, None])

    def create_placeholders_convolution(self, Config):

        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        self.node_dropout_flag = Config.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

    def user_embedding_reconstruct(self):
        self.user_reconstruct_ebd_, _ = self.GeneralGCN(n_layers=4, option='meta')
        self.user_reconstruct_ebd = tf.matmul(self.user_reconstruct_ebd_, self.weights['linear'])

        self.n_layers = len(self.weight_size)

        self.batch_predict_u_ebd = tf.nn.embedding_lookup(self.user_reconstruct_ebd, self.users)
        self.batch_target_u_ebd = tf.nn.embedding_lookup(self.target_user_ebd, self.users)
        self.batch_loss_u_reconstruct = -tf.reduce_mean(
            Cosine_similarity(self.batch_predict_u_ebd, self.batch_target_u_ebd))
        self.optimizer_u_reconstruct = tf.train.AdagradOptimizer(learning_rate=self.lr,
                                                                 initial_accumulator_value=1e-8).minimize(
            self.batch_loss_u_reconstruct)

    def item_embedding_reconstruct(self):
        _, self.item_reconstcuct_ebd_ = self.GeneralGCN(n_layers=4, option='meta')
        self.item_reconstruct_ebd = tf.matmul(self.item_reconstcuct_ebd_, self.weights['linear'])

        self.n_layers = len(self.weight_size)

        self.batch_predict_i_ebd = tf.nn.embedding_lookup(self.item_reconstruct_ebd, self.pos_items)
        # self.batch_predict_i_ebd = tf.contrib.layers.fully_connected(self.batch_predict_i_ebd_, 32, tf.nn.leaky_relu)
        self.batch_target_i_ebd = tf.nn.embedding_lookup(self.target_item_ebd, self.pos_items)
        self.batch_loss_i_reconstruct = -tf.reduce_mean(
            Cosine_similarity(self.batch_predict_i_ebd, self.batch_target_i_ebd))
        self.optimizer_i_reconstruct = tf.train.AdagradOptimizer(learning_rate=self.lr,
                                                                 initial_accumulator_value=1e-8).minimize(
            self.batch_loss_i_reconstruct)

    def _init_weights(self):
        all_weights = dict()
        initializer = tf.random_normal_initializer(stddev=0.01)  # tf.contrib.layers.xavier_initializer()

        # if self.pretrain_data is None:
        all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                    name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                    name='item_embedding')

        all_weights['meta_user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                         name='meta_user_embedding')

        all_weights['meta_item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                         name='meta_item_embedding')

        all_weights['linear'] = tf.Variable(initializer([2 * self.emb_dim, self.emb_dim]),
                                            name='linear_item_embedding')

        all_weights['cld_user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                        name='cld_user_embedding')
        all_weights['cld_item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                        name='cld_item_embedding')

        all_weights['clr_user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                        name='clr_user_embedding')
        all_weights['clr_item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                        name='clr_item_embedding')

        all_weights['bert_r_user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                         name='bert_r_user_embedding')

        all_weights['bert_r_item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                         name='bert_r_item_embedding')

        all_weights['bert_c_user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                         name='bert_c_user_embedding')

        all_weights['bert_c_item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                         name='bert_c_item_embedding')


        # n_users + n_items + [PAD] + [MASK]
        all_weights['bert_embedding'] = tf.Variable(initializer([self.vocab_size, self.emb_dim]), name='bert_embedding')


        print('using random initialization')


        return all_weights

    def GeneralGCN(self, n_layers, option):
        print('convolution layer(s) is %d' % n_layers)
        if self.conv_name == 'lightgcn':
            ua_embeddings, ia_embeddings = self._create_lightgcn_embed(n_layers=n_layers, option=option)
        else:
            raise Exception('option not exists !')

        return ua_embeddings, ia_embeddings

    '''
    downstream recommendation task
    '''

    def create_bpr_loss(self, users, pos_items, neg_items, u_g_embeddings_pre, pos_i_g_embeddings_pre,
                        neg_i_g_embeddings_pre):

        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        regularizer = tf.nn.l2_loss(u_g_embeddings_pre) + tf.nn.l2_loss(
            pos_i_g_embeddings_pre) + tf.nn.l2_loss(neg_i_g_embeddings_pre)
        regularizer = regularizer / self.batch_size

        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))

        emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _create_lightgcn_embed(self, n_layers, option):
        if option == 'contrastive_delete':
            A_fold_hat = self._split_A_hat(self.norm_adj_cld)
            ego_embeddings = tf.concat([self.weights['cld_user_embedding'], self.weights['cld_item_embedding']], axis=0)
        elif option == 'contrastive_replace':
            A_fold_hat = self._split_A_hat(self.norm_adj_clr)
            ego_embeddings = tf.concat([self.weights['clr_user_embedding'], self.weights['clr_item_embedding']], axis=0)
        elif option == 'normal' or option == 'anchor':
            A_fold_hat = self._split_A_hat(self.norm_adj)
            ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        elif option == 'meta':
            A_fold_hat = self._split_A_hat(self.norm_adj)
            ego_embeddings_user = tf.concat(
                [self.weights['user_embedding'], self.weights['meta_user_embedding']], axis=1)
            ego_embeddings_item = tf.concat(
                [self.weights['item_embedding'], self.weights['meta_item_embedding']], axis=1)
            ego_embeddings = tf.concat([ego_embeddings_user, ego_embeddings_item], axis=0)
        else:
            raise Exception('option not exists !')

        all_embeddings = [ego_embeddings]

        for k in range(0, n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            side_embeddings = tf.concat(temp_embed, 0)

            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
            if k == 0:
                self.temp1 = ego_embeddings
            if k == 1:
                self.temp2 = ego_embeddings
            if k == 2:
                self.temp3 = ego_embeddings
            if k == 3:
                self.temp4 = ego_embeddings

        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings


def load_pretrained_data(Config):
    pretrain_data = {}
    pretrain_data['user_embed'] = np.load(Config.pretrain_user_ebd_path)
    pretrain_data['item_embed'] = np.load(Config.pretrain_item_ebd_path)
    concat_embed = np.concatenate((pretrain_data['user_embed'], pretrain_data['item_embed']), 0)
    initial_ebd = np.zeros((Config.vocab_size, Config.embedding_size))
    pretrain_data['bert_embed'] = np.concatenate((concat_embed, initial_ebd), axis=0)
    return pretrain_data


def load_initial_data(Config):
    initial_data = {}
    initial_data['user_embed'] = np.load(Config.initial_user_ebd_path)
    initial_data['item_embed'] = np.load(Config.initial_item_ebd_path)
    return initial_data


def load_meta_data(Config):
    meta_data = {}
    meta_data['user_embed'] = np.load(Config.meta_user_ebd_path)
    meta_data['item_embed'] = np.load(Config.meta_item_ebd_path)
    return meta_data

def load_bert_reconstruction_data(Config):
    bert_r_data = {}
    bert_r_data['user_embed'] = np.load(Config.bert_user_ebd_path)
    bert_r_data['item_embed'] = np.load(Config.bert_item_ebd_path)
    return bert_r_data

def load_bert_contrastive_data(Config):
    bert_c_data = {}
    bert_c_data['user_embed'] = np.load(Config.bert_c_user_ebd_path)
    bert_c_data['item_embed'] = np.load(Config.bert_c_item_ebd_path)
    return bert_c_data


def Cosine_similarity(support_encode, query_encode):
    '''
    support_enocde [b, e]
    query_encode [b, e]
    return [b, 1]
    '''
    support_encode = tf.nn.l2_normalize(support_encode, axis=1)
    query_encode = tf.nn.l2_normalize(query_encode, axis=1)
    similarity = tf.reduce_sum(support_encode * query_encode, axis=1, keep_dims=True) * 0.5 + 0.5

    return similarity


def Pearson_correlation(support_encode, query_encode):
    batch_pearson = 0.0

    batch_row = support_encode.shape[0]
    for row in range(batch_row):
        support_encode_row = support_encode[row]
        query_encode_row = query_encode[row]
        pccs_row = pearsonr(support_encode_row, query_encode_row)[0]
        # pccs_row = 0.5 * pccs_row + 0.5
        batch_pearson += pccs_row
    return batch_pearson / batch_row


def contrastive_generate_training_file(epoch, Config):
    # np.random.seed(epoch)
    user_dict = {}
    with open(Config.data_path + Config.dataset + '/mask_train.txt', 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(' ')
            user = int(arr[0])
            user_dict[user] = []
            for item in arr[1:]:
                user_dict[user].append(int(item))
            line = f.readline()

    with open(Config.data_path + Config.dataset + '/mask_train.txt', 'r') as f:
        with open(Config.data_path + Config.dataset + '/replace_train.txt', 'w') as writer_replace:
            with open(Config.data_path + Config.dataset + '/delete_train.txt', 'w') as writer_delete:
                line = f.readline()
                while line != "" and line != None:
                    current_item_replace_list, current_item_delete_list = [], []
                    arr = line.strip().split(' ')
                    user = int(arr[0])
                    writer_replace.write(str(user) + ' ')
                    writer_delete.write(str(user) + ' ')
                    # replace
                    for item in arr[1:]:
                        prob = np.random.random()
                        if prob < Config.replace_ratio:
                            current_item_replace_list.append(np.random.choice(user_dict[user], 1)[0])
                            # print(np.random.choice(user_dict[user], 1))
                        else:
                            current_item_replace_list.append(item)

                    for index in range(len(current_item_replace_list)):
                        if index != len(current_item_replace_list) - 1:
                            writer_replace.write(str(current_item_replace_list[index]) + ' ')
                        else:
                            writer_replace.write(str(current_item_replace_list[index]) + '\n')

                    # delete
                    for item in arr[1:]:
                        prob = np.random.random()
                        if prob < Config.delete_ratio:
                            continue
                        else:
                            current_item_delete_list.append(item)

                    if len(current_item_delete_list) == 0:
                        current_item_delete_list.append(np.random.choice(user_dict[user], 1)[0])

                    for index in range(len(current_item_delete_list)):
                        if index != len(current_item_delete_list) - 1:
                            writer_delete.write(str(current_item_delete_list[index]) + ' ')
                        else:
                            writer_delete.write(str(current_item_delete_list[index]) + '\n')

                    line = f.readline()


def random_mask_training_file(epoch, Config):
    np.random.seed(epoch)
    with open(Config.data_path + Config.dataset + '/train.txt', 'r') as f:
        with open(Config.data_path + Config.dataset + '/mask_train.txt', 'w') as writer:
            line = f.readline()
            while line != "" and line != None:
                arr = line.strip().split(' ')
                user, items = arr[0], arr[1:]
                # perform sample strategy
                if len(items) >= Config.few_shot_number:
                    sampled_items = np.random.choice(items, Config.few_shot_number, replace=False)
                else:
                    sampled_items = np.random.choice(items, Config.few_shot_number, replace=True)

                writer.write(str(user) + ' ')
                for item_index in range(len(sampled_items)):
                    if item_index != len(sampled_items) - 1:
                        writer.write(str(sampled_items[item_index]) + ' ')
                    else:
                        writer.write(str(sampled_items[item_index]))
                writer.write('\n')

                line = f.readline()


'''
training user/item embedding reconstruction pretext task

'''


def training_downstream_recommendation_task(model, sess, epoch, config):
    best_precision, best_recall, best_ndcg = 0.0, 0.0, 0.0
    saver = tf.train.Saver(max_to_keep=3)

    users, pos_items, neg_items = data_generator.sample()
    sess.run([model.opt, model.loss, model.mf_loss, model.emb_loss, model.reg_loss],
             feed_dict={model.users: users, model.pos_items: pos_items,
                        model.node_dropout: eval(config.node_dropout),
                        model.mess_dropout: eval(config.mess_dropout),
                        model.neg_items: neg_items})

    '''
    evaluate performance
    '''
    users_to_test = list(data_generator.test_set.keys())
    result = test(sess, model, users_to_test, drop_flag=True)
    if result['ndcg'] > best_ndcg and result['recall'] > best_recall:
        best_ndcg = result['ndcg']
        best_recall = result['recall']
        saver.save(sess, config.checkpoint_path_downstream, global_step=epoch)

    print('Epoch %d, precision is %.4f, recall is %.4f, ndcg is %.4f, mrr is %.4f' % (
        epoch, result['precision'], result['recall'], result['ndcg'], result['mrr']))


def scaled_dot_product_attention(Q, K, V,
                                 causality=False, dropout_rate=0.,
                                 training=True,
                                 scope="scaled_dot_product_attention"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        # dot product
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)
        # Q * K^T  [N, T_q, d_k] * [N, d_k, T_k]

        # scale
        outputs /= d_k ** 0.5

        # key masking
        outputs = mask(outputs, Q, K, type="key")

        # causality or future blinding masking
        if causality:
            outputs = mask(outputs, type="future")

        # current_outputs [N, T_q, T_k]

        # softmax
        outputs = tf.nn.softmax(outputs)  # the sum of each row is equal to 1
        attention = tf.transpose(outputs, [0, 2, 1])  # transpose, the sum of each column is equal to 1
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

        # query masking
        outputs = mask(outputs, Q, K, type="query")  # [N, T_q, T_k]

        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

        # weighted sum (context vectors)
        outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

    return outputs


def mask(inputs, queries=None, keys=None, type=None):
    padding_num = -2 ** 32 + 1  # -4294967295 = -4.2949673e+09
    if type in ("k", "key", "keys"):
        # Generate masks
        masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)  #-1 means the last axis, sum the query
        masks = tf.expand_dims(masks, 1)  # (N, 1, T_k)
        masks = tf.tile(masks, [1, tf.shape(queries)[1], 1])  # (N, T_q, T_k)

        # Apply masks to inputs
        paddings = tf.ones_like(inputs) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)  # (N, T_q, T_k)
        # tf.where(condition, a, b) if condition=True, output = a. if condition=False, output = b
    elif type in ("q", "query", "queries"):
        # Generate masks
        masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
        masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
        masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])  # (N, T_q, T_k)

        # Apply masks to inputs
        outputs = inputs * masks
    elif type in ("f", "future", "right"):
        diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

        paddings = tf.ones_like(masks) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
    else:
        print("Check if you entered type correctly!")

    return outputs


def multihead_attention(queries, keys, values,
                        num_heads=8,
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):
    d_model = queries.get_shape().as_list()[-1]  # embedding_size: d_model
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Linear projections
        Q = tf.layers.dense(queries, d_model, use_bias=False)  # (N, T_q, d_model)
        K = tf.layers.dense(keys, d_model, use_bias=False)  # (N, T_k, d_model)
        V = tf.layers.dense(values, d_model, use_bias=False)  # (N, T_k, d_model)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

        # Attention
        outputs = scaled_dot_product_attention(Q_, K_, V_, causality, dropout_rate, training)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = ln(outputs)

    return outputs


def ff(inputs, num_units, scope="positionwise_feedforward"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

        # Outer layer
        outputs = tf.layers.dense(outputs, num_units[1])

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = ln(outputs)

    return outputs


def ln(inputs, epsilon=1e-8, scope="ln"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        # tf.nn.moments calculate the mean and variance
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      x: float Tensor to perform activation.

    Returns:
      `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

def training_batch(batch_index, model, sess, train_data, batch_size):
    loss = 0.0
    for index in tqdm.tqdm(batch_index):
        batch_input_ids, batch_input_mask, batch_segment_ids, batch_masked_lm_positions, batch_masked_lm_ids, batch_masked_lm_weights, batch_next_sentence_label = gendata_bert.batch_gen(
            train_data, index, batch_size)
        feed_dict = {model.input_ids: batch_input_ids,
                     model.input_mask: batch_input_mask,
                     model.token_type_ids: batch_segment_ids,
                     model.masked_lm_positions: batch_masked_lm_positions,
                     model.masked_lm_ids: batch_masked_lm_ids,
                     model.masked_lm_weights: batch_masked_lm_weights,
                     model.next_sentence_labels: batch_next_sentence_label}
        _, _, batch_loss = sess.run([model.sequence_output, model.train_op, model.total_loss], feed_dict)
        loss += batch_loss
    return loss / len(batch_index)


def training_batch_meta_path(batch_index, model, sess, train_data, batch_size):
    loss = 0.0
    for index in tqdm.tqdm(batch_index):
        batch_input_ids, batch_input_mask, batch_segment_ids, batch_masked_lm_positions, batch_masked_lm_ids, batch_masked_lm_weights, batch_next_sentence_label = gendata_meta_path.batch_gen(
            train_data, index, batch_size)
        feed_dict = {model.input_ids: batch_input_ids,
                     model.input_mask: batch_input_mask,
                     model.token_type_ids: batch_segment_ids,
                     model.masked_lm_positions: batch_masked_lm_positions,
                     model.masked_lm_ids: batch_masked_lm_ids,
                     model.masked_lm_weights: batch_masked_lm_weights,
                     model.next_sentence_labels: batch_next_sentence_label}
        _, _, batch_loss = sess.run([model.sequence_output, model.train_op, model.total_loss], feed_dict)
        loss += batch_loss
    return loss / len(batch_index)


def ranklist_by_heapq_negative_sampling(bert_config, user_pos_test, test_items, rating, Ks):
    '''
    test_items = 100 instances
    '''
    item_score = {}
    for index in range(len(test_items)):
        item_score[index] = float(rating[index])

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)
    # print(K_max_item_score)
    # print(user_pos_test)

    r = []
    for i in K_max_item_score:
        if i + bert_config.num_users in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc


def ranklist_by_heapq(bert_config, user_pos_test, test_items, rating, Ks):
    '''
    test_items = all_items - test_user_support_selected_items  test_items [num_users, num_users+1, ..., num_users+num_items]
    '''
    item_score = {}
    for i in test_items:
        item_score[i - bert_config.num_users] = rating[i - bert_config.num_users]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)
    # print(K_max_item_score)
    # print(user_pos_test)

    r = []
    for i in K_max_item_score:
        if i + bert_config.num_users in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc


def evaluate_negative_sampling(bert_config, model, sess, test_batches, test_user_pos_dict, test_user_ground_truth_dict,
                               negative_sampling_user_dict):
    negative_pool = 100
    split_batch_size = 250000
    hits5, ndcgs5, hits10, ndcgs10, maps, mrrs = [], [], [], [], [], []
    all_input_ids, all_input_mask, all_segment_ids, all_masked_lm_positions, all_masked_lm_ids, all_masked_lm_weights = test_batches
    precision, recall, ndcg, hit_ratio = [], [], [], []

    test_all_instances = len(all_input_ids)
    test_num_users = test_all_instances // split_batch_size + 1
    for user_index in tqdm.tqdm(range(int(test_num_users))):
        i_start = split_batch_size * user_index
        i_end = min(split_batch_size * user_index + split_batch_size, test_all_instances)
        # user = all_masked_lm_ids[user_index * bert_config.num_items][0]
        feed_dict = {model.input_ids: np.array(all_input_ids)[i_start:i_end, :],
                     model.input_mask: np.array(all_input_mask)[i_start:i_end, :],
                     model.segment_ids: np.array(all_segment_ids)[i_start:i_end, :],
                     model.masked_lm_positions: np.array(all_masked_lm_positions)[i_start: i_end, :],
                     model.masked_lm_ids: np.array(all_masked_lm_ids)[i_start: i_end, :],
                     model.masked_lm_weights: np.array(all_masked_lm_weights)[i_start: i_end, :]}

        predict_batch = sess.run(model.next_sentence_log_probs, feed_dict)  # [1000, 2]
        # print(np.array(predict_batch).shape)
        temp_predict_batch = np.array(predict_batch)[:, 1]  # [1000, 1] 取第一列为1的得分
        # print(temp_predict_batch.shape)
        rating = list(np.reshape(temp_predict_batch, (-1)))  # (1000,)

        num_split_rating = len(rating) // negative_pool
        for split_rating_index in range(num_split_rating):
            user = all_masked_lm_ids[user_index * negative_pool * split_rating_index][0]
            u = user
            training_items = test_user_pos_dict[u]  # test support set
            user_pos_test = test_user_ground_truth_dict[u]  # test ground_truth

            sampled_items = set(negative_sampling_user_dict[u])  # id is reindexed
            test_items = list(sampled_items - set(training_items))

            Ks = bert_config.Ks
            r, auc = ranklist_by_heapq_negative_sampling(bert_config, user_pos_test, test_items, rating, Ks)

            precision.append(metrics.precision_at_k(r, Ks[0]))
            recall.append(metrics.recall_at_k(r, Ks[0], len(user_pos_test)))
            ndcg.append(metrics.ndcg_at_k(r, Ks[0]))
            hit_ratio.append(metrics.hit_at_k(r, Ks[0]))
    return np.mean(np.array(precision)), np.mean(np.array(recall)), np.mean(np.array(ndcg)), np.mean(
        np.array(hit_ratio))


def evaluate(bert_config, model, sess, test_batches, test_user_pos_dict, test_user_ground_truth_dict):
    all_input_ids, all_input_mask, all_segment_ids, all_masked_lm_positions, all_masked_lm_ids, all_masked_lm_weights = test_batches
    precision, recall, ndcg, hit_ratio = [], [], [], []
    n_item_batchs = bert_config.num_items // bert_config.batch_size + 1

    test_all_instances = len(all_input_ids)

    test_num_users = test_all_instances / bert_config.num_items

    i_count = 0
    test_item_list = np.reshape(np.array(range(bert_config.num_items)), (-1, 1))
    for user_index in tqdm.tqdm(range(int(test_num_users))):

        predcition = np.zeros(shape=(1, bert_config.num_items))
        user = all_masked_lm_ids[user_index * bert_config.num_items][0]

        for i_batch_id in range(n_item_batchs):
            i_start = i_batch_id * bert_config.batch_size
            i_end = min((i_batch_id + 1) * (bert_config.batch_size), bert_config.num_items)

            item_batch = range(i_start, i_end)

            feed_dict = {model.input_ids: np.array(all_input_ids)[i_start:i_end, :],
                         model.input_mask: np.array(all_input_mask)[i_start:i_end, :],
                         model.segment_ids: np.array(all_segment_ids)[i_start:i_end, :],
                         model.masked_lm_positions: np.array(all_masked_lm_positions)[i_start: i_end, :],
                         model.masked_lm_ids: np.array(all_masked_lm_ids)[i_start: i_end, :],
                         model.masked_lm_weights: np.array(all_masked_lm_weights)[i_start: i_end, :]}

            predict_batch = sess.run(model.next_sentence_log_probs, feed_dict)  # [None, 2]
            # print(np.array(predict_batch).shape)
            temp_predict_batch = np.array(predict_batch)[:, 1]  # [None, 1] 取第一列为1的得分
            # print(temp_predict_batch.shape)
            predict_batch_final = np.reshape(temp_predict_batch, (-1, len(item_batch)))

            predcition[:, i_start: i_end] = predict_batch_final
            i_count += predict_batch_final.shape[1]
        assert i_count == bert_config.num_items
        i_count = 0

        rating = list(np.reshape(predcition, (-1)))  # (num_items,)


        u = user
        training_items = test_user_pos_dict[u]  # test support set
        user_pos_test = test_user_ground_truth_dict[u]  # test ground_truth

        all_items = set(range(bert_config.num_users, bert_config.num_items + bert_config.num_users))  # id is reindexed
        test_items = list(all_items - set(training_items))

        Ks = bert_config.Ks
        r, auc = ranklist_by_heapq(bert_config, user_pos_test, test_items, rating, Ks)

        precision.append(metrics.precision_at_k(r, Ks[0]))
        recall.append(metrics.recall_at_k(r, Ks[0], len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, Ks[0]))
        hit_ratio.append(metrics.hit_at_k(r, Ks[0]))
    return np.mean(np.array(precision)), np.mean(np.array(recall)), np.mean(np.array(ndcg)), np.mean(
        np.array(hit_ratio))


def adj_matrix_passing(config, parameter_config, plain_adj, norm_adj, mean_adj, pre_adj):
    if parameter_config.adj_type == 'plain':
        config['norm_adj'] = plain_adj
        print('use the plain adjacency matrix')
    elif parameter_config.adj_type == 'norm':
        config['norm_adj'] = norm_adj
        print('use the normalized adjacency matrix')
    elif parameter_config.adj_type == 'gcmc':
        config['norm_adj'] = mean_adj
        print('use the gcmc adjacency matrix')
    elif parameter_config.adj_type == 'pre':
        config['norm_adj'] = pre_adj
        print('use the pre adjcency matrix')
    else:
        config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
        print('use the mean adjacency matrix')

    return config


if __name__ == '__main__':

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        parameter_config = Model_Config()
        pretrain_data = load_pretrained_data(parameter_config)
        initial_data = load_initial_data(parameter_config)

        random_mask_training_file(epoch=0, Config=parameter_config)
        data_generator = Data(path=parameter_config.data_path + parameter_config.dataset,
                              batch_size=parameter_config.batch_size,
                              task_name=parameter_config.pretext_task_name)
        config = dict()
        config['n_users'] = data_generator.n_users
        config['n_items'] = data_generator.n_items

        """
        *********************************************************
        Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
        """
        plain_adj, norm_adj, mean_adj, pre_adj = data_generator.get_adj_mat()
        config = adj_matrix_passing(config, parameter_config, plain_adj, norm_adj, mean_adj, pre_adj)

        print('generate contrastive graph...')
        contrastive_generate_training_file(epoch=0, Config=parameter_config)
        contrastive_delete_data_generator = Data(path=parameter_config.data_path + parameter_config.dataset,
                                                 batch_size=parameter_config.batch_size,
                                                 task_name=parameter_config.contrastive_delete)
        contrastive_replace_data_generator = Data(path=parameter_config.data_path + parameter_config.dataset,
                                                  batch_size=parameter_config.batch_size,
                                                  task_name=parameter_config.contrastive_replace)

        config_d = dict()
        config_d['n_users'] = contrastive_delete_data_generator.n_users
        config_d['n_items'] = contrastive_delete_data_generator.n_items
        plain_adj, norm_adj, mean_adj, pre_adj = contrastive_delete_data_generator.get_adj_mat()
        config_d = adj_matrix_passing(config_d, parameter_config, plain_adj, norm_adj, mean_adj, pre_adj)

        config_r = dict()
        config_r['n_users'] = contrastive_replace_data_generator.n_users
        config_r['n_items'] = contrastive_replace_data_generator.n_items
        plain_adj, norm_adj, mean_adj, pre_adj = contrastive_replace_data_generator.get_adj_mat()
        config_r = adj_matrix_passing(config_r, parameter_config, plain_adj, norm_adj, mean_adj, pre_adj)
        print('finish generating contrastive graph...')

        GeneralConv = Recommender(config, config_d, config_r, pretrain_data, initial_data, parameter_config.conv_name,
                                  parameter_config, True)
        print(GeneralConv.n_users, GeneralConv.n_items)
        saver = tf.train.Saver(max_to_keep=3)
        sess.run(tf.global_variables_initializer())

        logging.info("initialized...")
        for epoch in range(0, 45):
            ########################## Contrastive with Transformer ###############################
            print('reconstruction with Transformer')
            training_contrastive_transformer(GeneralConv, sess)
            generate_contrastive_transformer_data(GeneralConv, sess)
            bert_contrastive_data = load_bert_contrastive_data(parameter_config)
            GeneralConv.weights['bert_c_user_embedding'] = bert_contrastive_data['user_embed']
            GeneralConv.weights['bert_c_item_embedding'] = bert_contrastive_data['item_embed']

            ########################## Reconstruction with Transformer ###############################
            print('reconstruction with Transformer')
            training_reconstruction_transformer(GeneralConv, sess)
            generate_transformer_data(GeneralConv, sess)
            bert_reconstruction_data = load_bert_reconstruction_data(parameter_config)
            GeneralConv.weights['bert_r_user_embedding'] = bert_reconstruction_data['user_embed']
            GeneralConv.weights['bert_r_item_embedding'] = bert_reconstruction_data['item_embed']


            # ######################### Task contrastive learning with GNN ############################
            contrastive_generate_training_file(epoch=0, Config=parameter_config)
            contrastive_delete_data_generator = Data(path=parameter_config.data_path + parameter_config.dataset,
                                                     batch_size=parameter_config.batch_size,
                                                     task_name=parameter_config.contrastive_delete)
            contrastive_replace_data_generator = Data(path=parameter_config.data_path + parameter_config.dataset,
                                                      batch_size=parameter_config.batch_size,
                                                      task_name=parameter_config.contrastive_replace)

            config_d = dict()
            config_d['n_users'] = contrastive_delete_data_generator.n_users
            config_d['n_items'] = contrastive_delete_data_generator.n_items
            plain_adj, norm_adj, mean_adj, pre_adj = contrastive_delete_data_generator.get_adj_mat()
            config_d = adj_matrix_passing(config_d, parameter_config, plain_adj, norm_adj, mean_adj, pre_adj)

            config_r = dict()
            config_r['n_users'] = contrastive_replace_data_generator.n_users
            config_r['n_items'] = contrastive_replace_data_generator.n_items
            plain_adj, norm_adj, mean_adj, pre_adj = contrastive_replace_data_generator.get_adj_mat()
            config_r = adj_matrix_passing(config_r, parameter_config, plain_adj, norm_adj, mean_adj, pre_adj)

            GeneralConv.cldel_config = config_d
            GeneralConv.clrep_config = config_r

            print('training user contrastive learning with GNN task ...')
            training_batch_user_contrastive_gnn_task(GeneralConv, sess)
            print('training item contrastive learning with GNN task ...')
            training_batch_item_contrastive_gnn_task(GeneralConv, sess)

            ######################## Task Embedding Reconstruction with GNN  #########################
            random_mask_training_file(0, parameter_config)

            data_generator = Data(path=parameter_config.data_path + parameter_config.dataset,
                                  batch_size=parameter_config.batch_size,
                                  task_name=parameter_config.pretext_task_name)
            config = dict()
            config['n_users'] = data_generator.n_users
            config['n_items'] = data_generator.n_items

            """
            *********************************************************
            Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
            """
            plain_adj, norm_adj, mean_adj, pre_adj = data_generator.get_adj_mat()
            config = adj_matrix_passing(config, parameter_config, plain_adj, norm_adj, mean_adj, pre_adj)

            GeneralConv.data_config = config
            GeneralConv.norm_adj = config['norm_adj']

            print('Embedding Reconstruction with GNN')
            training_user_task(GeneralConv, sess)
            training_item_task(GeneralConv, sess)

            # train meta aggregator
            generate_user_task_data(GeneralConv, sess)
            generate_item_task_data(GeneralConv, sess)
            meta_data = load_meta_data(parameter_config)

            GeneralConv.weights['meta_user_embedding'] = meta_data['user_embed']
            GeneralConv.weights['meta_item_embedding'] = meta_data['item_embed']

            # train final embedding reconstruction task
            # training_batch_item_reconstruct_task(GeneralConv, sess, parameter_config)
            training_batch_user_reconstruct_task(GeneralConv, sess, parameter_config)

            # downstream task
            print('start fine-tuning...')
            data_generator = Data(path=parameter_config.data_path + parameter_config.dataset,
                                  batch_size=parameter_config.batch_size,
                                  task_name='downstream_task')
            config = dict()
            config['n_users'] = data_generator.n_users
            config['n_items'] = data_generator.n_items

            """
            *********************************************************
            Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
            """
            plain_adj, norm_adj, mean_adj, pre_adj = data_generator.get_adj_mat()
            config = adj_matrix_passing(config, parameter_config, plain_adj, norm_adj, mean_adj, pre_adj)

            GeneralConv.data_config = config
            GeneralConv.norm_adj = config['norm_adj']

            training_downstream_recommendation_task(GeneralConv, sess, epoch, parameter_config)

