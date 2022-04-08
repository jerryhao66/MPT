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

# tf.compat.logging.set_verbosity(tf.compat.logging.ERROR)
import os
import argparse
import sys
import threading
import numpy as np
from tensorflow.python.client import device_lib
from utility.helper import *
from utility.batch_test import *
import scipy.sparse as sp
import logging
import tqdm
from scipy.stats import pearsonr
from trainer import *
import gendata as data

import collections
import copy
import json
import math
import re
import six
# import gendata_bert
# import gendata_meta_path
from utility import metrics
import heapq
from utility.Config import Model_Config
from time import time

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import resource_variable_ops

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
cpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'CPU']


class AdamWeightDecayOptimizer(optimizer.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def _prepare(self):
        self.learning_rate_t = ops.convert_to_tensor(
            self.learning_rate, name='learning_rate')
        self.weight_decay_rate_t = ops.convert_to_tensor(
            self.weight_decay_rate, name='weight_decay_rate')
        self.beta_1_t = ops.convert_to_tensor(self.beta_1, name='beta_1')
        self.beta_2_t = ops.convert_to_tensor(self.beta_2, name='beta_2')
        self.epsilon_t = ops.convert_to_tensor(self.epsilon, name='epsilon')

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, 'm', self._name)
            self._zeros_slot(v, 'v', self._name)

    def _apply_dense(self, grad, var):
        learning_rate_t = math_ops.cast(
            self.learning_rate_t, var.dtype.base_dtype)
        beta_1_t = math_ops.cast(self.beta_1_t, var.dtype.base_dtype)
        beta_2_t = math_ops.cast(self.beta_2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self.epsilon_t, var.dtype.base_dtype)
        weight_decay_rate_t = math_ops.cast(
            self.weight_decay_rate_t, var.dtype.base_dtype)

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        # Standard Adam update.
        next_m = (
                tf.multiply(beta_1_t, m) +
                tf.multiply(1.0 - beta_1_t, grad))
        next_v = (
                tf.multiply(beta_2_t, v) + tf.multiply(1.0 - beta_2_t,
                                                       tf.square(grad)))

        update = next_m / (tf.sqrt(next_v) + epsilon_t)

        if self._do_use_weight_decay(var.name):
            update += weight_decay_rate_t * var

        update_with_lr = learning_rate_t * update

        next_param = var - update_with_lr

        return control_flow_ops.group(*[var.assign(next_param),
                                        m.assign(next_m),
                                        v.assign(next_v)])

    def _resource_apply_dense(self, grad, var):
        learning_rate_t = math_ops.cast(
            self.learning_rate_t, var.dtype.base_dtype)
        beta_1_t = math_ops.cast(self.beta_1_t, var.dtype.base_dtype)
        beta_2_t = math_ops.cast(self.beta_2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self.epsilon_t, var.dtype.base_dtype)
        weight_decay_rate_t = math_ops.cast(
            self.weight_decay_rate_t, var.dtype.base_dtype)

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        # Standard Adam update.
        next_m = (
                tf.multiply(beta_1_t, m) +
                tf.multiply(1.0 - beta_1_t, grad))
        next_v = (
                tf.multiply(beta_2_t, v) + tf.multiply(1.0 - beta_2_t,
                                                       tf.square(grad)))

        update = next_m / (tf.sqrt(next_v) + epsilon_t)

        if self._do_use_weight_decay(var.name):
            update += weight_decay_rate_t * var

        update_with_lr = learning_rate_t * update

        next_param = var - update_with_lr

        return control_flow_ops.group(*[var.assign(next_param),
                                        m.assign(next_m),
                                        v.assign(next_v)])

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        learning_rate_t = math_ops.cast(
            self.learning_rate_t, var.dtype.base_dtype)
        beta_1_t = math_ops.cast(self.beta_1_t, var.dtype.base_dtype)
        beta_2_t = math_ops.cast(self.beta_2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self.epsilon_t, var.dtype.base_dtype)
        weight_decay_rate_t = math_ops.cast(
            self.weight_decay_rate_t, var.dtype.base_dtype)

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        m_t = state_ops.assign(m, m * beta_1_t,
                               use_locking=self._use_locking)

        m_scaled_g_values = grad * (1 - beta_1_t)
        with ops.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, m_scaled_g_values)

        v_scaled_g_values = (grad * grad) * (1 - beta_2_t)
        v_t = state_ops.assign(v, v * beta_2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, v_scaled_g_values)

        update = m_t / (math_ops.sqrt(v_t) + epsilon_t)

        if self._do_use_weight_decay(var.name):
            update += weight_decay_rate_t * var

        update_with_lr = learning_rate_t * update

        var_update = state_ops.assign_sub(var,
                                          update_with_lr,
                                          use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
                x, i, v, use_locking=self._use_locking))

    def _resource_scatter_add(self, x, i, v):
        with ops.control_dependencies(
                [resource_variable_ops.resource_scatter_add(
                    x.handle, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        return self._apply_sparse_shared(
            grad, var, indices, self._resource_scatter_add)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True


class BertConfig(object):
    """Configuration for `BertModel`."""

    def __init__(self,
                 vocab_size,
                 hidden_size=32,
                 num_hidden_layers=24,
                 num_attention_heads=12,
                 intermediate_size=4096,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=128,
                 type_vocab_size=2,
                 initializer_range=0.02):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class Recommender(object):
    def __init__(self, data_config, pretrain_data, initial_data, conv_name, Config, is_training, scope=None):
        # gnn parameters
        # self.model_type = 'LightGCN'
        self.Config = Config
        self.data_config = data_config
        self.adj_type = Config.adj_type
        # self.alg_type = Config.alg_type
        self.dataset = Config.dataset
        self.pretrain_data = pretrain_data
        self.initial_data = initial_data
        self.n_users = self.data_config['n_users']
        self.n_items = self.data_config['n_items']
        self.n_fold = 100
        self.norm_adj = self.data_config['norm_adj']
        # self.n_nonzero_elems = self.norm_adj.count_nonzero()
        self.lr = Config.lr
        self.emb_dim = Config.embed_size
        self.batch_size = Config.batch_size
        self.weight_size = eval(Config.layer_size)
        self.weight_size_single = eval(Config.layer_size1)
        self.n_layers = len(self.weight_size)
        self.regs = eval(Config.regs)
        self.decay = self.regs[0]
        # self.log_dir = self.create_model_str()
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




        # bert-based pretext task
        # Config = copy.deepcopy(Config)
        # if not is_training:
        #     Config.hidden_dropout_prob = 0.0
        #     Config.attention_probs_dropout_prob = 0.0

        # bert based pretext task
        # self.bert_based_pretext_task(Config, scope)

        # build convolution model, in order to avoid information leakage, we use additional contextual information
        # self.create_placeholders_convolution(Config)
        # self.weights, self.weights_context = self._init_weights()
        # self.ua_embeddings, self.ia_embeddings = self.GeneralGCN(n_layers=self.n_layers, option='concat_meta')
        # self.ua_embeddings, self.ia_embeddings = self.GeneralGCN(n_layers=self.n_layers, option='concat_bert_mask')
        # self.ua_embeddings, self.ia_embeddings = self.GeneralGCN(n_layers=self.n_layers, option='concat_meta_bert_mask')
        # self.ua_embeddings, self.ia_embeddings = self.GeneralGCN(n_layers=self.n_layers, option='concat_meta_meta_path')
        # self.ua_embeddings, self.ia_embeddings = self.GeneralGCN(n_layers=self.n_layers,
        #                                                          option='concat_meta_path_meta_bert_mask')

        # self.ua_embeddings, self.ia_embeddings = self.GeneralGCN(n_layers=self.n_layers,
        #                                                          option='meta')

        # self.ua_embeddings_c, self.ia_embeddings_c = self.GeneralGCN(n_layers=self.n_layers,
        #                                                          option='concat_contrastive_meta')  # [num_users, 2e], [num_items, 2e]

        # # contrastive pretext task
        # self.contrastive_pretext_task()
        # # self.ua_embeddings, self.ia_embeddings = self.GeneralGCN(n_layers=self.n_layers,
        # #                                                          option='concat_contrastive_meta')  # [num_users, 2e], [num_items, 2e]

        # self.subgraph_mutual_info_task()

        # pretrain trainer parameter
        # self.mask_data = mask_data
        # self.target_user_ebd, self.target_item_ebd = self.pretrain_data['user_embed'], self.pretrain_data['item_embed']
        # restruct user/item embedding through GNN aggregation operation
        # self.restruct_user_ebd, self.restruct_item_ebd = self.ua_embeddings, self.ia_embeddings

        # # add meta embedding, and retrain gnn model
        # self.user_embedding_reconstruct()
        # self.item_embedding_reconstruct()

        # finetuning parameter
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

        self.u_g_embeddings = tf.nn.embedding_lookup( tf.concat([self.user_reconstruct_ebd, self.ua_embeddings], axis=1), self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(tf.concat([self.item_reconstcuct_ebd, self.ia_embeddings], axis=1), self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(tf.concat([self.item_reconstcuct_ebd, self.ia_embeddings], axis=1), self.neg_items)


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



        # self.mf_loss_r,  self.emb_loss_r, self.reg_loss_r = self.create_bpr_loss(self.u_g_embeddings_r,
        #                                                                   self.pos_i_g_embeddings_r,
        #                                                                   self.neg_i_g_embeddings_r,
        #                                                                   self.u_g_embeddings_pre,
        #                                                                   self.pos_i_g_embeddings_pre,
        #                                                                   self.neg_i_g_embeddings_pre)
        #
        # self.loss = self.mf_loss + self.emb_loss + self.mf_loss_r + self.emb_loss_r

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def subgraph_mutual_info_task(self):
        self.ua_embeddings_m, self.ia_embeddings_m = self.GeneralGCN(n_layers=self.n_layers,
                                                                     option='normal')  # [num_users, 3e], [num_items, 3e]

        self.variables_gcn_parameters = tf.trainable_variables()
        self.num_variables_gcn = len(tf.trainable_variables())

        self.loss_mutual_user = self.infer_mutual_info_user_task()
        self.variables_mutual_user_parameters = tf.trainable_variables()[self.num_variables_gcn:]
        self.loss_mutual_item = self.infer_mutual_info_item_task()
        self.num_variables_mutual_item_parameters = tf.trainable_variables()[self.num_variables_gcn + len(
            self.variables_mutual_user_parameters):]

        optimizer = AdamWeightDecayOptimizer(
            learning_rate=self.Config.lr,
            weight_decay_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

        tvars_user = self.variables_gcn_parameters + self.variables_mutual_user_parameters
        grads_user = tf.gradients(self.loss_mutual_user, tvars_user)

        # This is how the model was pre-trained.
        (grads_user, _) = tf.clip_by_global_norm(grads_user, clip_norm=1.0)
        self.train_op_user_m = optimizer.apply_gradients(zip(grads_user, tvars_user))

        tvars_item = self.variables_gcn_parameters + self.num_variables_mutual_item_parameters
        grads_item = tf.gradients(self.loss_mutual_item, tvars_item)
        (grads_item, _) = tf.clip_by_global_norm(grads_item, clip_norm=1.0)

        self.train_op_item_m = optimizer.apply_gradients(zip(grads_item, tvars_item))

    def contrastive_pretext_task(self):
        self.ua_embeddings_c, self.ia_embeddings_c = self.GeneralGCN(n_layers=self.n_layers,
                                                                     option='normal')  # [num_users, e], [num_items, e]
        print(self.ua_embeddings_c.shape)
        print(self.ia_embeddings_c.shape)

        self.ua_embeddings_c_shape = tf.shape(self.ua_embeddings_c)
        # contrastive pretext task
        self.variables_gcn_parameters = tf.trainable_variables()
        self.num_variables_gcn = len(tf.trainable_variables())

        self.contrastive_loss = self.contrastive_coding_linking_user_task()
        self.variables_contrastive_user_parameters = tf.trainable_variables()[self.num_variables_gcn:]

        self.contrastive_loss_item = self.contrastive_coding_linking_item_task()
        self.variables_contrastive_item_parameters = tf.trainable_variables()[self.num_variables_gcn + len(
            self.variables_contrastive_user_parameters):]

        optimizer = AdamWeightDecayOptimizer(
            learning_rate=self.Config.lr,
            weight_decay_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

        tvars_user = self.variables_gcn_parameters + self.variables_contrastive_user_parameters
        grads_user = tf.gradients(self.contrastive_loss, tvars_user)

        # This is how the model was pre-trained.
        (grads_user, _) = tf.clip_by_global_norm(grads_user, clip_norm=1.0)
        self.train_op_user_cc = optimizer.apply_gradients(zip(grads_user, tvars_user))

        tvars_item = self.variables_gcn_parameters + self.variables_contrastive_item_parameters
        grads_item = tf.gradients(self.contrastive_loss_item, tvars_item)
        (grads_item, _) = tf.clip_by_global_norm(grads_item, clip_norm=1.0)

        self.train_op_item_cc = optimizer.apply_gradients(zip(grads_item, tvars_item))

    def bert_based_pretext_task(self, bert_config, scope):
        self.input_ids = tf.placeholder(tf.int32, shape=[None, None])  # [b, n]
        self.input_mask = tf.placeholder(tf.int32, shape=[None, None])  # [b, n]
        self.segment_ids = tf.placeholder(tf.int32, shape=[None, None])  # [b, n]
        self.token_type_ids = self.segment_ids
        self.masked_lm_positions = tf.placeholder(tf.int32, shape=[None, None])  # [b, n]
        self.masked_lm_ids = tf.placeholder(tf.int32, shape=[None, None])  # [b, n]
        self.masked_lm_weights = tf.placeholder(tf.int32, shape=[None, None])
        self.next_sentence_labels = tf.placeholder(tf.int32, shape=[None])
        self.gen_ebd_mask_position = tf.placeholder(tf.int32)

        # batch_size = config.batch_size
        # seq_length = config.seq_length

        with tf.variable_scope(scope, default_name="bert"):
            with tf.variable_scope("embeddings"):
                # Perform embedding lookup on the word ids.

                self.embedding_table = tf.Variable(
                    tf.constant(bert_config.pretrain_data['concat_embed'],
                                shape=[bert_config.vocab_size, bert_config.hidden_size]),
                    trainable=True)
                self.embedding_output = tf.nn.embedding_lookup(self.embedding_table, self.input_ids)  # [b, n, e]

                # Add positional embeddings and token type embeddings, then layer
                # normalize and perform dropout.
                # do not use type id and position embedding
                self.embedding_output = embedding_postprocessor(
                    input_tensor=self.embedding_output,
                    use_token_type=False,
                    token_type_ids=self.token_type_ids,
                    token_type_vocab_size=bert_config.type_vocab_size,
                    token_type_embedding_name="token_type_embeddings",
                    use_position_embeddings=False,
                    position_embedding_name="position_embeddings",
                    initializer_range=bert_config.initializer_range,
                    max_position_embeddings=bert_config.max_position_embeddings,
                    dropout_prob=bert_config.hidden_dropout_prob)

            with tf.variable_scope("encoder"):
                # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
                # mask of shape [batch_size, seq_length, seq_length] which is used
                # for the attention scores.
                attention_mask = create_attention_mask_from_input_mask(bert_config,
                                                                       self.input_ids, self.input_mask)

                # Run the stacked transformer.
                # `sequence_output` shape = [batch_size, seq_length, hidden_size].
                self.all_encoder_layers = transformer_model(
                    input_tensor=self.embedding_output,
                    attention_mask=attention_mask,
                    hidden_size=bert_config.hidden_size,
                    num_hidden_layers=bert_config.num_hidden_layers,
                    num_attention_heads=bert_config.num_attention_heads,
                    intermediate_size=bert_config.intermediate_size,
                    intermediate_act_fn=get_activation(bert_config.hidden_act),
                    hidden_dropout_prob=bert_config.hidden_dropout_prob,
                    attention_probs_dropout_prob=bert_config.attention_probs_dropout_prob,
                    initializer_range=bert_config.initializer_range,
                    do_return_all_layers=True)

            self.sequence_output = self.all_encoder_layers[-1]
            # The "pooler" converts the encoded sequence tensor of shape
            # [batch_size, seq_length, hidden_size] to a tensor of shape
            # [batch_size, hidden_size]. This is necessary for segment-level
            # (or segment-pair-level) classification tasks where we need a fixed
            # dimensional representation of the segment.

            with tf.variable_scope("pooler"):
                # We "pool" the model by simply taking the hidden state corresponding
                # to the first token. We assume that this has been pre-trained
                first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
                self.pooled_output = tf.layers.dense(
                    first_token_tensor,
                    bert_config.hidden_size,
                    activation=tf.tanh,
                    kernel_initializer=create_initializer(bert_config.initializer_range))  # [b, e]

                # self.cls_output = tf.layers.dense(
                #     self.pooled_output,
                #     1,
                #     activation=tf.nn.relu,
                #     kernel_initializer=create_initializer(config.initializer_range))  # [b ,1]

            # test generate embedding
            self.mask_gen_ebd = tf.squeeze(
                self.sequence_output[:, self.gen_ebd_mask_position + 1:self.gen_ebd_mask_position + 2, :], axis=1)

            (masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
                bert_config, self.sequence_output, self.embedding_table,
                self.masked_lm_positions, self.masked_lm_ids, self.masked_lm_weights)

            (next_sentence_loss, next_sentence_example_loss,
             self.next_sentence_log_probs) = get_next_sentence_output(
                bert_config, self.pooled_output, self.next_sentence_labels)

            # self.output_weights = tf.Variable(tf.truncated_normal(shape=[bert_config.hidden_size, 1], mean=0.0, stddev=tf.sqrt(tf.div(2.0, 1 + bert_config.hidden_size))),
            #     name='output_weights', dtype=tf.float32, trainable=True)
            # self.output_bias = tf.get_variable(
            #     "output_bias", shape=[1], initializer=tf.zeros_initializer())

            # self.logits, next_sentence_loss = get_next_sentence_output_new(
            #     bert_config, self.output_weights, self.output_bias, self.pooled_output, self.next_sentence_labels)

            self.total_loss = masked_lm_loss + next_sentence_loss
            optimizer = AdamWeightDecayOptimizer(
                learning_rate=bert_config.lr,
                weight_decay_rate=0.01,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-6,
                exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

            tvars = tf.trainable_variables()
            grads = tf.gradients(self.total_loss, tvars)

            # This is how the model was pre-trained.
            (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def get_pooled_output(self):
        return self.pooled_output

    def get_sequence_output(self):
        """Gets final hidden layer of encoder.

        Returns:
          float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
          to the final hidden of the transformer encoder.
        """
        return self.sequence_output

    def get_all_encoder_layers(self):
        return self.all_encoder_layers

    def get_embedding_output(self):
        """Gets output of the embedding lookup (i.e., input to the transformer).

        Returns:
          float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
          to the output of the embedding layer, after summing the word
          embeddings with the positional embeddings and the token type embeddings,
          then performing layer normalization. This is the input to the transformer.
        """
        return self.embedding_output

    def get_embedding_table(self):
        return self.embedding_table

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

    def _encode_cc(self, input, emb_dim, training=True):
        '''
        input: [b, n, e]
        output: [b, n, e]
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            enc = input
            enc *= emb_dim ** 0.5

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
                    enc = ff(enc, num_units=[self.d_ff, emb_dim])  # [b, q, e]
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

    def contrastive_coding_linking_user_task(self):
        with tf.variable_scope('contrastive_loss_user_task'):
            self.bias = tf.Variable(tf.zeros(self.n_items), name='bias', trainable=True)
            output_bias = tf.get_variable(
                "output_bias_item", shape=[2], initializer=tf.zeros_initializer())
            output_weights = tf.get_variable("output_weights_item", shape=[2, 3 * self.emb_dim],
                                             initializer=tf.truncated_normal_initializer(stddev=0.02))

            pos_item_ebd1 = tf.nn.embedding_lookup(self.ia_embeddings, self.support_item_cc)  # [b, n1, 3e]
            pos_item_ebd2 = tf.nn.embedding_lookup(self.ia_embeddings, self.support_item_pos_cc)  # [b, n2, 3e]
            neg_item_ebd = tf.nn.embedding_lookup(self.ia_embeddings, self.support_item_neg_cc)  # [b, n3, 3e]

            # h1, h2, h3
            pos_item_ebd1 = tf.reduce_mean(self._encode_cc(pos_item_ebd1, 5 * self.emb_dim),
                                           1)  # pass self-attention layer [b, 2e]
            pos_item_ebd2 = tf.reduce_mean(self._encode_cc(pos_item_ebd2, 5 * self.emb_dim), 1)  # [b, e]
            neg_item_ebd = tf.reduce_mean(self._encode_cc(neg_item_ebd, 5 * self.emb_dim), 1)  # [b, e]

            # z1, z2, z3
            pos1_z = tf.contrib.layers.fully_connected(pos_item_ebd1, 32, tf.nn.leaky_relu)
            pos2_z = tf.contrib.layers.fully_connected(pos_item_ebd2, 32, tf.nn.leaky_relu)
            neg_z = tf.contrib.layers.fully_connected(neg_item_ebd, 32, tf.nn.leaky_relu)

            pos_output = Cosine_similarity(pos1_z, pos2_z)
            neg_output = Cosine_similarity(pos1_z, neg_z)

            norm_distance = pos_output - neg_output  # [b, 1]
            one_minus_norm_distance = 1 - norm_distance
            logits = tf.concat([one_minus_norm_distance, norm_distance], 1)  # [b, 2]

            # logits = tf.matmul(norm_distance, output_weights, transpose_b=True)  # [b, 2]
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            b = tf.shape(pos_item_ebd1)[0]
            labels = tf.ones([b], tf.int32)
            one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)

            self.gen_contrastive_user_ebd = pos_item_ebd1  # [b, 2e]

        return loss

    def infer_mutual_info_item_task(self):
        with tf.variable_scope('mutual_info_item_task'):
            output_weights = tf.get_variable("output_weights_item", shape=[2, self.emb_dim],
                                             initializer=tf.truncated_normal_initializer(
                                                 stddev=self.Config.initializer_range))

            output_bias = tf.get_variable(
                "output_bias_item", shape=[2], initializer=tf.zeros_initializer())

            pos_user_ebd = tf.nn.embedding_lookup(self.ua_embeddings_m, self.support_user_pos)  # [b, n, 2e]
            neg_user_ebd = tf.nn.embedding_lookup(self.ua_embeddings_m, self.support_user_neg)  # [b, n. 2e]
            b, n, e = tf.shape(pos_user_ebd)[0], tf.shape(pos_user_ebd)[1], tf.shape(pos_user_ebd)[2]
            trans_pos_user_ebd = self._encode_cc(pos_user_ebd, self.emb_dim)  # [b, n, 2e]
            trans_neg_user_ebd = self._encode_cc(neg_user_ebd, self.emb_dim)  # [b, n, 2e]

            # readout
            pos_user_ebd = tf.reshape(trans_pos_user_ebd, (-1, self.emb_dim))  # [b*n, e]
            neg_user_ebd = tf.reshape(trans_neg_user_ebd, (-1, self.emb_dim))  # [b*n, e]

            norm_pos_user_ebd = tf.contrib.layers.fully_connected(pos_user_ebd, self.emb_dim, tf.nn.relu)  # [b*n, e]
            norm_neg_user_ebd = tf.contrib.layers.fully_connected(neg_user_ebd, self.emb_dim, tf.nn.relu)  # [b*n, e]
            norm_pos_user_ebd = tf.reduce_sum(tf.reshape(norm_pos_user_ebd, (b, n, e)), 1)  # [b, n, e] -> [b, e]
            norm_neg_user_ebd = tf.reduce_sum(tf.reshape(norm_neg_user_ebd, (b, n, e)), 1)  # [b, n, e] -> [b, e]

            self.gen_mutual_item_ebd = norm_pos_user_ebd

            norm_distance = norm_pos_user_ebd - norm_neg_user_ebd  # [b, e]

            logits = tf.matmul(norm_distance, output_weights, transpose_b=True)  # [b, 2]
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            labels = tf.ones([b], tf.int32)
            one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)

        return loss

    def infer_mutual_info_user_task(self):
        with tf.variable_scope("mutual_info_user_task"):
            output_weights = tf.get_variable("output_weights_user", shape=[2, self.emb_dim],
                                             initializer=tf.truncated_normal_initializer(
                                                 stddev=self.Config.initializer_range))

            output_bias = tf.get_variable(
                "output_bias_user", shape=[2], initializer=tf.zeros_initializer())

            pos_item_ebd = tf.nn.embedding_lookup(self.ia_embeddings_m, self.support_item_pos)  # [b, n, e]
            neg_item_ebd = tf.nn.embedding_lookup(self.ia_embeddings_m, self.support_item_neg)  # [b, n, e]
            b, n, e = tf.shape(pos_item_ebd)[0], tf.shape(pos_item_ebd)[1], tf.shape(pos_item_ebd)[2]
            trans_pos_item_ebd = self._encode_cc(pos_item_ebd, self.emb_dim)  # [b, n, e]
            trans_neg_item_ebd = self._encode_cc(neg_item_ebd, self.emb_dim)  # [b, n, e]

            # read_out

            pos_item_ebd = tf.reshape(trans_pos_item_ebd, (-1, self.emb_dim))  # [b*n, e]
            neg_item_ebd = tf.reshape(trans_neg_item_ebd, (-1, self.emb_dim))  # [b*n, e]

            norm_pos_item_ebd = tf.contrib.layers.fully_connected(pos_item_ebd, self.emb_dim, tf.nn.relu)  # [b*n, e]
            norm_neg_item_ebd = tf.contrib.layers.fully_connected(neg_item_ebd, self.emb_dim, tf.nn.relu)  # [b*n, e]
            norm_pos_item_ebd = tf.reduce_sum(tf.reshape(norm_pos_item_ebd, (b, n, e)), 1)  # [b, n, e] -> [b, e]
            norm_neg_item_ebd = tf.reduce_sum(tf.reshape(norm_neg_item_ebd, (b, n, e)), 1)  # [b, n, e] -> [b, e]

            self.gen_mutual_user_ebd = norm_pos_item_ebd

            norm_distance = norm_pos_item_ebd - norm_neg_item_ebd  # [b, 2e]

            logits = tf.matmul(norm_distance, output_weights, transpose_b=True)  # [b, 2]
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            labels = tf.ones([b], tf.int32)
            one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)

            return loss

    def contrastive_coding_linking_item_task(self):
        with tf.variable_scope('contrastive_loss_item_task'):
            self.bias = tf.Variable(tf.zeros(self.n_users), name='bias', trainable=True)
            output_bias = tf.get_variable(
                "output_bias_item", shape=[2], initializer=tf.zeros_initializer())
            output_weights = tf.get_variable("output_weights_item", shape=[2, 3 * self.emb_dim],
                                             initializer=tf.truncated_normal_initializer(stddev=0.02))

            pos_user_ebd1 = tf.nn.embedding_lookup(self.ua_embeddings, self.support_user_cc)  # [b, n, 2e]
            pos_user_ebd2 = tf.nn.embedding_lookup(self.ua_embeddings, self.support_user_pos_cc)  # [b, 2e]
            neg_user_ebd = tf.nn.embedding_lookup(self.ua_embeddings, self.support_user_neg_cc)  # [b, 2e]

            # h1, h2, h3
            pos_user_ebd1 = tf.reduce_mean(self._encode_cc(pos_user_ebd1, 5 * self.emb_dim),
                                           1)  # pass self-attention layer [b, n, 2e]
            pos_user_ebd2 = tf.reduce_mean(self._encode_cc(pos_user_ebd2, 5 * self.emb_dim), 1)
            neg_user_ebd = tf.reduce_mean(self._encode_cc(neg_user_ebd, 5 * self.emb_dim), 1)

            # z1, z2, z3
            pos1_z = tf.contrib.layers.fully_connected(pos_user_ebd1, 32, tf.nn.leaky_relu)
            pos2_z = tf.contrib.layers.fully_connected(pos_user_ebd2, 32, tf.nn.leaky_relu)
            neg_z = tf.contrib.layers.fully_connected(neg_user_ebd, 32, tf.nn.leaky_relu)

            pos_output = Cosine_similarity(pos1_z, pos2_z)
            neg_output = Cosine_similarity(pos1_z, neg_z)

            norm_distance = pos_output - neg_output  # [b, 1]

            one_minus_norm_distance = 1 - norm_distance
            logits = tf.concat([one_minus_norm_distance, norm_distance], 1)  # [b, 2]

            # logits = tf.matmul(norm_distance, output_weights, transpose_b=True)  # [b, 2]
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            b = tf.shape(pos_user_ebd1)[0]
            labels = tf.ones([b], tf.int32)
            one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)

            self.gen_contrastive_item_ebd = pos_user_ebd1

        return loss

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
        # self.support_item_cc = tf.placeholder(tf.int32, shape=[None, None])
        # self.support_item_pos_cc = tf.placeholder(tf.int32, shape=[None, None])
        # self.support_item_neg_cc = tf.placeholder(tf.int32, shape=[None, None])
        #
        # # subgraph mutual user
        # self.support_item_pos = tf.placeholder(tf.int32, shape=[None, None])
        # self.support_item_neg = tf.placeholder(tf.int32, shape=[None, None])
        #
        # # subgraph mutual item
        # self.support_user_pos = tf.placeholder(tf.int32, shape=[None, None])
        # self.support_user_neg = tf.placeholder(tf.int32, shape=[None, None])


        # # contrastive item
        # self.support_user_cc = tf.placeholder(tf.int32, shape=[None, None])
        # self.support_user_pos_cc = tf.placeholder(tf.int32, shape=[None, None])
        # self.support_user_neg_cc = tf.placeholder(tf.int32, shape=[None, None])

    def create_placeholders_convolution(self, Config):

        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        self.node_dropout_flag = Config.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

    def user_embedding_reconstruct(self):
        self.user_reconstruct_ebd_, _ =  self.GeneralGCN(n_layers=4, option='meta')
        self.user_reconstruct_ebd = tf.matmul(self.user_reconstruct_ebd_, self.weights['linear'])

        self.n_layers = len(self.weight_size)

        self.batch_predict_u_ebd = tf.nn.embedding_lookup(self.user_reconstruct_ebd, self.users)
        # self.batch_predict_u_ebd = tf.contrib.layers.fully_connected(self.batch_predict_u_ebd_, 32, tf.nn.leaky_relu)
        self.batch_target_u_ebd = tf.nn.embedding_lookup(self.target_user_ebd, self.users)
        self.batch_loss_u_reconstruct = -tf.reduce_mean(
            Cosine_similarity(self.batch_predict_u_ebd, self.batch_target_u_ebd))
        self.optimizer_u_reconstruct = tf.train.AdagradOptimizer(learning_rate=self.lr,
                                                                 initial_accumulator_value=1e-8).minimize(
            self.batch_loss_u_reconstruct)

    def item_embedding_reconstruct(self):
        _, self.item_reconstcuct_ebd_ = self.GeneralGCN(n_layers=4, option='meta')
        self.item_reconstcuct_ebd = tf.matmul(self.item_reconstcuct_ebd_, self.weights['linear'])

        self.n_layers = len(self.weight_size)

        self.batch_predict_i_ebd = tf.nn.embedding_lookup(self.item_reconstcuct_ebd, self.pos_items)
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

        all_weights['bpr'] = tf.Variable(initializer([self.emb_dim, self.emb_dim]),
                                                    name='linear_item_embedding')
        # all_weights['linear_item'] = tf.Variable(initializer([2 * self.emb_dim, self.emb_dim]),
        #                                             name='linear_item_embedding')

        print('using random initialization')


        self.weight_size_list = [self.emb_dim * 2] + self.weight_size
        self.weight_size_list1 = [self.emb_dim] + self.weight_size_single

        # no contextual weights/bias
        for k in range(self.n_layers):
            all_weights['W_gc_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_mlp_%d' % k)

            all_weights['W_gc_single%d' % k] = tf.Variable(
                initializer([self.weight_size_list1[k], self.weight_size_list1[k + 1]]), name='W_gc_single%d' % k)
            all_weights['b_gc_single%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list1[k + 1]]), name='b_gc_single%d' % k)

            all_weights['W_bi_single%d' % k] = tf.Variable(
                initializer([self.weight_size_list1[k], self.weight_size_list1[k + 1]]), name='W_bi_single%d' % k)
            all_weights['b_bi_single%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list1[k + 1]]), name='b_bi_single%d' % k)

            all_weights['W_mlp_single%d' % k] = tf.Variable(
                initializer([self.weight_size_list1[k], self.weight_size_list1[k + 1]]), name='W_mlp_single%d' % k)
            all_weights['b_mlp_single%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list1[k + 1]]), name='b_mlp_single%d' % k)

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
        # pos_scores = tf.sigmoid(tf.matmul(tf.multiply(users, pos_items)))
        # neg_scores = tf.sigmoid(tf.matmul(tf.multiply(users, neg_items), self.weights['bpr']))
        # self.batch_ratings = tf.matmul(users, pos_items, transpose_a=False, transpose_b=True)
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        # self.batch_ratings = tf.matmul(users, pos_items, transpose_a=False, transpose_b=True)

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
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        if option == 'normal':
            ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        # elif option == 'meta_user':
        #     ego_embeddings = tf.concat(
        #         [self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        # elif option == 'meta_item':
        #     ego_embeddings = tf.concat(
        #         [self.weights_context['half_user_embedding'], self.weights_context['mask_item_embedding']], axis=0)

        # general gnn meta embedding
        elif option == 'meta':

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


def load_contrastive_data():
    contrastive_data = {}
    contrastive_data['user_embed'] = np.load('./Data/movielens_dataset/contrastive/contrastive_user_ebd.npy')
    contrastive_data['item_embed'] = np.load('./Data/movielens_dataset/contrastive/contrastive_item_ebd.npy')
    return contrastive_data



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
    # test_users, test_pos_items, test_neg_items = data_generator.sample_test()
    # data = sess.run([model.loss, model.mf_loss, model.emb_loss],
    #                      feed_dict={model.users: test_users, model.pos_items: test_pos_items,
    #                                 model.neg_items: test_neg_items,
    #                                 model.node_dropout: eval(args.node_dropout),
    #                                 model.mess_dropout: eval(args.mess_dropout)})

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


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    '''
    tf.gather的用法 从params的axis维根据indices的参数值获取切片

    examples
    import tensorflow as tf
    temp = tf.range(0, 10) * 10 + tf.constant(1, shape=[10])
    temp1 = tf.gather(temp, [4, 5, 9])

    temp: [1 11 21 31 41 51 61 71 81 91]
    temp1: [41 51 91 ]

    '''
    return output_tensor


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
    '''
    (masked_lm_loss,
       masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
           bert_config, model.get_sequence_output(), model.get_embedding_table(),
           masked_lm_positions, masked_lm_ids, masked_lm_weights)

    '''
    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=get_activation(bert_config.hidden_act),
                kernel_initializer=create_initializer(
                    bert_config.initializer_range))
            input_tensor = layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "output_bias",
            shape=[bert_config.vocab_size],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        label_weights = tf.cast(label_weights, tf.float32)
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
    """Get loss and log probs for the next sentence prediction."""

    # Simple binary classification. Note that 0 is "next sentence" and 1 is
    # "random sentence". This weight matrix is not used after pre-training.
    with tf.variable_scope("cls/seq_relationship"):
        output_weights = tf.get_variable(
            "output_weights",
            shape=[2, bert_config.hidden_size],
            initializer=create_initializer(bert_config.initializer_range))
        output_bias = tf.get_variable(
            "output_bias", shape=[2], initializer=tf.zeros_initializer())

        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        labels = tf.reshape(labels, [-1])
        one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)  # [b, 2]
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, per_example_loss, log_probs)


def get_next_sentence_output_new(bert_config, output_weights, output_bias, input_tensor, labels):
    labels = tf.expand_dims(labels, -1)  # [b] -> [b, 1]

    logits = tf.matmul(input_tensor, output_weights)  # [b, 1]
    loss = tf.losses.log_loss(labels, logits) + 1e-5 * tf.reduce_sum(tf.square(output_weights))

    return logits, loss


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


def get_activation(activation_string):
    """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

    Args:
      activation_string: String name of the activation function.

    Returns:
      A Python function corresponding to the activation function. If
      `activation_string` is None, empty, or "linear", this will return None.
      If `activation_string` is not a string, it will return `activation_string`.

    Raises:
      ValueError: The `activation_string` does not correspond to a known
        activation.
    """

    # We assume that anything that"s not a string is already an activation
    # function, so we just return it.
    if not isinstance(activation_string, six.string_types):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output


def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
    """Runs layer normalization followed by dropout."""
    output_tensor = layer_norm(input_tensor, name)
    output_tensor = dropout(output_tensor, dropout_prob)
    return output_tensor


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):
    """Looks up words embeddings for id tensor.

    Args:
      input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
        ids.
      vocab_size: int. Size of the embedding vocabulary.
      embedding_size: int. Width of the word embeddings.
      initializer_range: float. Embedding initialization range.
      word_embedding_name: string. Name of the embedding table.
      use_one_hot_embeddings: bool. If True, use one-hot method for word
        embeddings. If False, use `tf.gather()`.

    Returns:
      float Tensor of shape [batch_size, seq_length, embedding_size].
    """
    # This function assumes that the input is of shape [batch_size, seq_length,
    # num_inputs].
    #
    # If the input is a 2D tensor of shape [batch_size, seq_length], we
    # reshape to [batch_size, seq_length, 1].
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])

    embedding_table = tf.get_variable(
        name=word_embedding_name,
        shape=[vocab_size, embedding_size],
        initializer=create_initializer(initializer_range))

    flat_input_ids = tf.reshape(input_ids, [-1])
    if use_one_hot_embeddings:
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
        output = tf.matmul(one_hot_input_ids, embedding_table)
    else:
        output = tf.gather(embedding_table, flat_input_ids)

    input_shape = get_shape_list(input_ids)

    output = tf.reshape(output,
                        input_shape[0:-1] + [input_shape[-1] * embedding_size])
    return (output, embedding_table)


def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
    """Performs various post-processing on a word embedding tensor.

    Args:
      input_tensor: float Tensor of shape [batch_size, seq_length,
        embedding_size].
      use_token_type: bool. Whether to add embeddings for `token_type_ids`.
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
        Must be specified if `use_token_type` is True.
      token_type_vocab_size
      : int. The vocabulary size of `token_type_ids`.
      token_type_embedding_name: string. The name of the embedding table variable
        for token type ids.
      use_position_embeddings: bool. Whether to add position embeddings for the
        position of each token in the sequence.
      position_embedding_name: string. The name of the embedding table variable
        for positional embeddings.
      initializer_range: float. Range of the weight initialization.
      max_position_embeddings: int. Maximum sequence length that might ever be
        used with this model. This can be longer than the sequence length of
        input_tensor, but cannot be shorter.
      dropout_prob: float. Dropout probability applied to the final output tensor.

    Returns:
      float tensor with same shape as `input_tensor`.

    Raises:
      ValueError: One of the tensor shapes or input values is invalid.
    """
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    output = input_tensor

    if use_token_type:
        if token_type_ids is None:
            raise ValueError("`token_type_ids` must be specified if"
                             "`use_token_type` is True.")
        token_type_table = tf.get_variable(
            name=token_type_embedding_name,
            shape=[token_type_vocab_size, width],
            initializer=create_initializer(initializer_range))
        # This vocab will be small so we always do one-hot here, since it is always
        # faster for a small vocabulary.
        flat_token_type_ids = tf.reshape(token_type_ids, [-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
        token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
        token_type_embeddings = tf.reshape(token_type_embeddings,
                                           [batch_size, seq_length, width])
        output += token_type_embeddings

    if use_position_embeddings:
        assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
        with tf.control_dependencies([assert_op]):
            full_position_embeddings = tf.get_variable(
                name=position_embedding_name,
                shape=[max_position_embeddings, width],
                initializer=create_initializer(initializer_range))
            # Since the position embedding table is a learned variable, we create it
            # using a (long) sequence length `max_position_embeddings`. The actual
            # sequence length might be shorter than this, for faster training of
            # tasks that do not have long sequences.
            #
            # So `full_position_embeddings` is effectively an embedding table
            # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
            # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
            # perform a slice.
            position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                           [seq_length, -1])
            num_dims = len(output.shape.as_list())

            # Only the last two dimensions are relevant (`seq_length` and `width`), so
            # we broadcast among the first dimensions, which is typically just
            # the batch size.
            position_broadcast_shape = []
            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend([seq_length, width])
            position_embeddings = tf.reshape(position_embeddings,
                                             position_broadcast_shape)
            output += position_embeddings

    output = layer_norm_and_dropout(output, dropout_prob)
    return output


def create_attention_mask_from_input_mask(config, from_tensor, to_mask):
    """Create 3D attention mask from a 2D tensor mask.

    Args:
      from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
      to_mask: int32 Tensor of shape [batch_size, to_seq_length].

    Returns:
      float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    # from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    # batch_size = from_shape[0]
    # from_seq_length = from_shape[1]

    # batch_size = config.batch_size
    from_seq_length = config.seq_length

    batch_size = tf.shape(from_tensor)[0]

    # to_shape = get_shape_list(to_mask, expected_rank=2)
    # to_seq_length = to_shape[1]
    to_seq_length = config.seq_length

    to_mask = tf.cast(
        tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones(
        shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask

    return mask


def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
    """Performs multi-headed attention from `from_tensor` to `to_tensor`.

    This is an implementation of multi-headed attention based on "Attention
    is all you Need". If `from_tensor` and `to_tensor` are the same, then
    this is self-attention. Each timestep in `from_tensor` attends to the
    corresponding sequence in `to_tensor`, and returns a fixed-with vector.

    This function first projects `from_tensor` into a "query" tensor and
    `to_tensor` into "key" and "value" tensors. These are (effectively) a list
    of tensors of length `num_attention_heads`, where each tensor is of shape
    [batch_size, seq_length, size_per_head].

    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor and returned.

    In practice, the multi-headed attention are done with transposes and
    reshapes rather than actual separate tensors.

    Args:
      from_tensor: float Tensor of shape [batch_size, from_seq_length,
        from_width].
      to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
      attention_mask: (optional) int32 Tensor of shape [batch_size,
        from_seq_length, to_seq_length]. The values should be 1 or 0. The
        attention scores will effectively be set to -infinity for any positions in
        the mask that are 0, and will be unchanged for positions that are 1.
      num_attention_heads: int. Number of attention heads.
      size_per_head: int. Size of each attention head.
      query_act: (optional) Activation function for the query transform.
      key_act: (optional) Activation function for the key transform.
      value_act: (optional) Activation function for the value transform.
      attention_probs_dropout_prob: (optional) float. Dropout probability of the
        attention probabilities.
      initializer_range: float. Range of the weight initializer.
      do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
        * from_seq_length, num_attention_heads * size_per_head]. If False, the
        output will be of shape [batch_size, from_seq_length, num_attention_heads
        * size_per_head].
      batch_size: (Optional) int. If the input is 2D, this might be the batch size
        of the 3D version of the `from_tensor` and `to_tensor`.
      from_seq_length: (Optional) If the input is 2D, this might be the seq length
        of the 3D version of the `from_tensor`.
      to_seq_length: (Optional) If the input is 2D, this might be the seq length
        of the 3D version of the `to_tensor`.

    Returns:
      float Tensor of shape [batch_size, from_seq_length,
        num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
        true, this will be of shape [batch_size * from_seq_length,
        num_attention_heads * size_per_head]).

    Raises:
      ValueError: Any of the arguments or tensor shapes are invalid.
    """

    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                             seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

    if len(from_shape) != len(to_shape):
        raise ValueError(
            "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`

    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)

    # `query_layer` = [B*F, N*H]
    query_layer = tf.layers.dense(
        from_tensor_2d,
        num_attention_heads * size_per_head,
        activation=query_act,
        name="query",
        kernel_initializer=create_initializer(initializer_range))

    # `key_layer` = [B*T, N*H]
    key_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=key_act,
        name="key",
        kernel_initializer=create_initializer(initializer_range))

    # `value_layer` = [B*T, N*H]
    value_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=value_act,
        name="value",
        kernel_initializer=create_initializer(initializer_range))

    # `query_layer` = [B, N, F, H]
    query_layer = transpose_for_scores(query_layer, batch_size,
                                       num_attention_heads, from_seq_length,
                                       size_per_head)

    # `key_layer` = [B, N, T, H]
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                     to_seq_length, size_per_head)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(size_per_head)))

    if attention_mask is not None:
        # `attention_mask` = [B, 1, F, T]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_scores += adder

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

    # `value_layer` = [B, T, N, H]
    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, num_attention_heads, size_per_head])

    # `value_layer` = [B, N, T, H]
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    # `context_layer` = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)

    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    if do_return_2d_tensor:
        # `context_layer` = [B*F, N*H]
        context_layer = tf.reshape(
            context_layer,
            [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        # `context_layer` = [B, F, N*H]
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, num_attention_heads * size_per_head])

    return context_layer


def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
    """Multi-headed, multi-layer Transformer from "Attention is All You Need".

    This is almost an exact implementation of the original Transformer encoder.

    See the original paper:
    https://arxiv.org/abs/1706.03762

    Also see:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

    Args:
      input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
      attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
        seq_length], with 1 for positions that can be attended to and 0 in
        positions that should not be.
      hidden_size: int. Hidden size of the Transformer.
      num_hidden_layers: int. Number of layers (blocks) in the Transformer.
      num_attention_heads: int. Number of attention heads in the Transformer.
      intermediate_size: int. The size of the "intermediate" (a.k.a., feed
        forward) layer.
      intermediate_act_fn: function. The non-linear activation function to apply
        to the output of the intermediate/feed-forward layer.
      hidden_dropout_prob: float. Dropout probability for the hidden layers.
      attention_probs_dropout_prob: float. Dropout probability of the attention
        probabilities.
      initializer_range: float. Range of the initializer (stddev of truncated
        normal).
      do_return_all_layers: Whether to also return all layers or just the final
        layer.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size], the final
      hidden layer of the Transformer.

    Raises:
      ValueError: A Tensor shape or parameter is invalid.
    """
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (hidden_size, num_attention_heads))

    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]

    # The Transformer performs sum residuals on all layers so the input needs
    # to be the same as the hidden size.
    if input_width != hidden_size:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                         (input_width, hidden_size))

    # We keep the representation as a 2D tensor to avoid re-shaping it back and
    # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
    # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
    # help the optimizer.
    prev_output = reshape_to_matrix(input_tensor)

    all_layer_outputs = []
    for layer_idx in range(num_hidden_layers):
        with tf.variable_scope("layer_%d" % layer_idx):
            layer_input = prev_output

            with tf.variable_scope("attention"):
                attention_heads = []
                with tf.variable_scope("self"):
                    attention_head = attention_layer(
                        from_tensor=layer_input,
                        to_tensor=layer_input,
                        attention_mask=attention_mask,
                        num_attention_heads=num_attention_heads,
                        size_per_head=attention_head_size,
                        attention_probs_dropout_prob=attention_probs_dropout_prob,
                        initializer_range=initializer_range,
                        do_return_2d_tensor=True,
                        batch_size=batch_size,
                        from_seq_length=seq_length,
                        to_seq_length=seq_length)
                    attention_heads.append(attention_head)

                attention_output = None
                if len(attention_heads) == 1:
                    attention_output = attention_heads[0]
                else:
                    # In the case where we have other sequences, we just concatenate
                    # them to the self-attention head before the projection.
                    attention_output = tf.concat(attention_heads, axis=-1)

                # Run a linear projection of `hidden_size` then add a residual
                # with `layer_input`.
                with tf.variable_scope("output"):
                    attention_output = tf.layers.dense(
                        attention_output,
                        hidden_size,
                        kernel_initializer=create_initializer(initializer_range))
                    attention_output = dropout(attention_output, hidden_dropout_prob)
                    attention_output = layer_norm(attention_output + layer_input)

            # The activation is only applied to the "intermediate" hidden layer.
            with tf.variable_scope("intermediate"):
                intermediate_output = tf.layers.dense(
                    attention_output,
                    intermediate_size,
                    activation=intermediate_act_fn,
                    kernel_initializer=create_initializer(initializer_range))

            # Down-project back to `hidden_size` then add the residual.
            with tf.variable_scope("output"):
                layer_output = tf.layers.dense(
                    intermediate_output,
                    hidden_size,
                    kernel_initializer=create_initializer(initializer_range))
                layer_output = dropout(layer_output, hidden_dropout_prob)
                layer_output = layer_norm(layer_output + attention_output)
                prev_output = layer_output
                all_layer_outputs.append(layer_output)

    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output)
        return final_outputs
    else:
        final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.

    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                         (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
    """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.

    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def from_dict(json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = BertConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
        config.__dict__[key] = value
    return config


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
        # print(rating)

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


'''
training meta aggregator  wsdm21 code 
'''


################### bert mask task generate embedding #######

def generate_bert_mask_task_data(model, sess, test_batches0, test_batches1, test_batches2, test_batches3,
                                 parameter_config):
    test_num_batch0 = np.array(test_batches0[0]).shape[0] // parameter_config.batch_size + 1
    test_num_batch1 = np.array(test_batches1[0]).shape[0] // parameter_config.batch_size + 1
    test_num_batch2 = np.array(test_batches2[0]).shape[0] // parameter_config.batch_size + 1
    test_num_batch3 = np.array(test_batches3[0]).shape[0] // parameter_config.batch_size + 1
    batch_test_index0, batch_test_index1, batch_test_index2, batch_test_index3 = range(test_num_batch0), range(
        test_num_batch1), range(test_num_batch2), range(test_num_batch3)

    print('mask position 0 embedding generate...')
    for index in tqdm.tqdm(batch_test_index0):
        batch_input_ids, batch_input_mask, batch_segment_ids, batch_masked_lm_positions, batch_masked_lm_ids, batch_masked_lm_weights, batch_next_sentence_label = gendata_bert.batch_gen(
            test_batches0, index, parameter_config.batch_size)
        feed_dict = {model.input_ids: batch_input_ids,
                     model.input_mask: batch_input_mask,
                     model.token_type_ids: batch_segment_ids,
                     model.masked_lm_positions: batch_masked_lm_positions,
                     model.masked_lm_ids: batch_masked_lm_ids,
                     model.masked_lm_weights: batch_masked_lm_weights,
                     model.next_sentence_labels: batch_next_sentence_label,
                     model.gen_ebd_mask_position: 0}
        if index == 0:
            gen_ebd0 = sess.run(model.mask_gen_ebd, feed_dict)
            gen_ebd0 = np.array(gen_ebd0)
        else:
            temp = sess.run(model.mask_gen_ebd, feed_dict)
            temp = np.array(temp)
            gen_ebd0 = np.concatenate([gen_ebd0, temp], 0)

    print('mask position 1 embedding generate...')
    for index in tqdm.tqdm(batch_test_index1):
        batch_input_ids, batch_input_mask, batch_segment_ids, batch_masked_lm_positions, batch_masked_lm_ids, batch_masked_lm_weights, batch_next_sentence_label = gendata_bert.batch_gen(
            test_batches1, index, parameter_config.batch_size)
        feed_dict = {model.input_ids: batch_input_ids,
                     model.input_mask: batch_input_mask,
                     model.token_type_ids: batch_segment_ids,
                     model.masked_lm_positions: batch_masked_lm_positions,
                     model.masked_lm_ids: batch_masked_lm_ids,
                     model.masked_lm_weights: batch_masked_lm_weights,
                     model.next_sentence_labels: batch_next_sentence_label,
                     model.gen_ebd_mask_position: 1}
        if index == 0:
            gen_ebd1 = sess.run(model.mask_gen_ebd, feed_dict)
            gen_ebd1 = np.array(gen_ebd1)
        else:
            temp = sess.run(model.mask_gen_ebd, feed_dict)
            temp = np.array(temp)
            gen_ebd1 = np.concatenate([gen_ebd1, temp], 0)

    print('mask position 2 embedding generate...')
    for index in tqdm.tqdm(batch_test_index2):
        batch_input_ids, batch_input_mask, batch_segment_ids, batch_masked_lm_positions, batch_masked_lm_ids, batch_masked_lm_weights, batch_next_sentence_label = gendata_bert.batch_gen(
            test_batches2, index, parameter_config.batch_size)
        feed_dict = {model.input_ids: batch_input_ids,
                     model.input_mask: batch_input_mask,
                     model.token_type_ids: batch_segment_ids,
                     model.masked_lm_positions: batch_masked_lm_positions,
                     model.masked_lm_ids: batch_masked_lm_ids,
                     model.masked_lm_weights: batch_masked_lm_weights,
                     model.next_sentence_labels: batch_next_sentence_label,
                     model.gen_ebd_mask_position: 2}
        if index == 0:
            gen_ebd2 = sess.run(model.mask_gen_ebd, feed_dict)
            gen_ebd2 = np.array(gen_ebd2)
        else:
            temp = sess.run(model.mask_gen_ebd, feed_dict)
            temp = np.array(temp)
            gen_ebd2 = np.concatenate([gen_ebd2, temp], 0)

    print('mask position 3 embedding generate...')
    for index in tqdm.tqdm(batch_test_index3):
        batch_input_ids, batch_input_mask, batch_segment_ids, batch_masked_lm_positions, batch_masked_lm_ids, batch_masked_lm_weights, batch_next_sentence_label = gendata_bert.batch_gen(
            test_batches3, index, parameter_config.batch_size)
        feed_dict = {model.input_ids: batch_input_ids,
                     model.input_mask: batch_input_mask,
                     model.token_type_ids: batch_segment_ids,
                     model.masked_lm_positions: batch_masked_lm_positions,
                     model.masked_lm_ids: batch_masked_lm_ids,
                     model.masked_lm_weights: batch_masked_lm_weights,
                     model.next_sentence_labels: batch_next_sentence_label,
                     model.gen_ebd_mask_position: 3}
        if index == 0:
            gen_ebd3 = sess.run(model.mask_gen_ebd, feed_dict)
            gen_ebd3 = np.array(gen_ebd3)
        else:
            temp = sess.run(model.mask_gen_ebd, feed_dict)
            temp = np.array(temp)
            gen_ebd3 = np.concatenate([gen_ebd3, temp], 0)

    return gen_ebd0, gen_ebd1, gen_ebd2, gen_ebd3


def split_and_save_bert_mask_task_data(config, ebd0, ebd1, ebd2, ebd3):
    user0 = ebd0[0:config.num_users, :]
    item0 = ebd0[config.num_users:, :]

    user1 = ebd1[0:config.num_users, :]
    item1 = ebd1[config.num_users:, :]

    user2 = ebd2[0:config.num_users, :]
    item2 = ebd2[config.num_users:, :]

    user3 = ebd3[0:config.num_users, :]
    item3 = ebd3[config.num_users:, :]

    np.save('bert_mask_user0.npy', user0)
    np.save('bert_mask_item0.npy', item0)

    np.save('bert_mask_user1.npy', user1)
    np.save('bert_mask_item1.npy', item1)

    np.save('bert_mask_user2.npy', user2)
    np.save('bert_mask_item2.npy', item2)

    np.save('bert_mask_user3.npy', user3)
    np.save('bert_mask_item3.npy', item3)

    user = (user0 + user1 + user2 + user3) / 4
    item = (item0 + item1 + item2 + item3) / 4
    return user, item


################################################################


def generate_meta_path_mask_task_data(model, sess, test_batches0, test_batches1, test_batches2, test_batches3,
                                      test_batches4, test_batches5, test_batches6, test_batches7, parameter_config):
    test_num_batch0 = np.array(test_batches0[0]).shape[0] // parameter_config.batch_size + 1
    test_num_batch1 = np.array(test_batches1[0]).shape[0] // parameter_config.batch_size + 1
    test_num_batch2 = np.array(test_batches2[0]).shape[0] // parameter_config.batch_size + 1
    test_num_batch3 = np.array(test_batches3[0]).shape[0] // parameter_config.batch_size + 1
    test_num_batch4 = np.array(test_batches4[0]).shape[0] // parameter_config.batch_size + 1
    test_num_batch5 = np.array(test_batches5[0]).shape[0] // parameter_config.batch_size + 1
    test_num_batch6 = np.array(test_batches6[0]).shape[0] // parameter_config.batch_size + 1
    test_num_batch7 = np.array(test_batches7[0]).shape[0] // parameter_config.batch_size + 1

    batch_test_index0, batch_test_index1, batch_test_index2, batch_test_index3 = range(test_num_batch0), range(
        test_num_batch1), range(test_num_batch2), range(test_num_batch3)

    batch_test_index4, batch_test_index5, batch_test_index6, batch_test_index7 = range(test_num_batch4), range(
        test_num_batch5), range(test_num_batch6), range(test_num_batch7)

    print('mask position 0 embedding generate...')
    for index in tqdm.tqdm(batch_test_index0):
        batch_input_ids, batch_input_mask, batch_segment_ids, batch_masked_lm_positions, batch_masked_lm_ids, batch_masked_lm_weights, batch_next_sentence_label = gendata_meta_path.batch_gen(
            test_batches0, index, parameter_config.batch_size)
        feed_dict = {model.input_ids: batch_input_ids,
                     model.input_mask: batch_input_mask,
                     model.token_type_ids: batch_segment_ids,
                     model.masked_lm_positions: batch_masked_lm_positions,
                     model.masked_lm_ids: batch_masked_lm_ids,
                     model.masked_lm_weights: batch_masked_lm_weights,
                     model.next_sentence_labels: batch_next_sentence_label,
                     model.gen_ebd_mask_position: 0}
        if index == 0:
            gen_ebd0 = sess.run(model.mask_gen_ebd, feed_dict)
            gen_ebd0 = np.array(gen_ebd0)
        else:
            temp = sess.run(model.mask_gen_ebd, feed_dict)
            temp = np.array(temp)
            gen_ebd0 = np.concatenate([gen_ebd0, temp], 0)

    print('mask position 1 embedding generate...')
    for index in tqdm.tqdm(batch_test_index1):
        batch_input_ids, batch_input_mask, batch_segment_ids, batch_masked_lm_positions, batch_masked_lm_ids, batch_masked_lm_weights, batch_next_sentence_label = gendata_meta_path.batch_gen(
            test_batches1, index, parameter_config.batch_size)
        feed_dict = {model.input_ids: batch_input_ids,
                     model.input_mask: batch_input_mask,
                     model.token_type_ids: batch_segment_ids,
                     model.masked_lm_positions: batch_masked_lm_positions,
                     model.masked_lm_ids: batch_masked_lm_ids,
                     model.masked_lm_weights: batch_masked_lm_weights,
                     model.next_sentence_labels: batch_next_sentence_label,
                     model.gen_ebd_mask_position: 1}
        if index == 0:
            gen_ebd1 = sess.run(model.mask_gen_ebd, feed_dict)
            gen_ebd1 = np.array(gen_ebd1)
        else:
            temp = sess.run(model.mask_gen_ebd, feed_dict)
            temp = np.array(temp)
            gen_ebd1 = np.concatenate([gen_ebd1, temp], 0)

    print('mask position 2 embedding generate...')
    for index in tqdm.tqdm(batch_test_index2):
        batch_input_ids, batch_input_mask, batch_segment_ids, batch_masked_lm_positions, batch_masked_lm_ids, batch_masked_lm_weights, batch_next_sentence_label = gendata_meta_path.batch_gen(
            test_batches2, index, parameter_config.batch_size)
        feed_dict = {model.input_ids: batch_input_ids,
                     model.input_mask: batch_input_mask,
                     model.token_type_ids: batch_segment_ids,
                     model.masked_lm_positions: batch_masked_lm_positions,
                     model.masked_lm_ids: batch_masked_lm_ids,
                     model.masked_lm_weights: batch_masked_lm_weights,
                     model.next_sentence_labels: batch_next_sentence_label,
                     model.gen_ebd_mask_position: 2}
        if index == 0:
            gen_ebd2 = sess.run(model.mask_gen_ebd, feed_dict)
            gen_ebd2 = np.array(gen_ebd2)
        else:
            temp = sess.run(model.mask_gen_ebd, feed_dict)
            temp = np.array(temp)
            gen_ebd2 = np.concatenate([gen_ebd2, temp], 0)

    print('mask position 3 embedding generate...')
    for index in tqdm.tqdm(batch_test_index3):
        batch_input_ids, batch_input_mask, batch_segment_ids, batch_masked_lm_positions, batch_masked_lm_ids, batch_masked_lm_weights, batch_next_sentence_label = gendata_meta_path.batch_gen(
            test_batches3, index, parameter_config.batch_size)
        feed_dict = {model.input_ids: batch_input_ids,
                     model.input_mask: batch_input_mask,
                     model.token_type_ids: batch_segment_ids,
                     model.masked_lm_positions: batch_masked_lm_positions,
                     model.masked_lm_ids: batch_masked_lm_ids,
                     model.masked_lm_weights: batch_masked_lm_weights,
                     model.next_sentence_labels: batch_next_sentence_label,
                     model.gen_ebd_mask_position: 3}
        if index == 0:
            gen_ebd3 = sess.run(model.mask_gen_ebd, feed_dict)
            gen_ebd3 = np.array(gen_ebd3)
        else:
            temp = sess.run(model.mask_gen_ebd, feed_dict)
            temp = np.array(temp)
            gen_ebd3 = np.concatenate([gen_ebd3, temp], 0)

    print('mask position 4 embedding generate...')
    for index in tqdm.tqdm(batch_test_index4):
        batch_input_ids, batch_input_mask, batch_segment_ids, batch_masked_lm_positions, batch_masked_lm_ids, batch_masked_lm_weights, batch_next_sentence_label = gendata_meta_path.batch_gen(
            test_batches4, index, parameter_config.batch_size)
        feed_dict = {model.input_ids: batch_input_ids,
                     model.input_mask: batch_input_mask,
                     model.token_type_ids: batch_segment_ids,
                     model.masked_lm_positions: batch_masked_lm_positions,
                     model.masked_lm_ids: batch_masked_lm_ids,
                     model.masked_lm_weights: batch_masked_lm_weights,
                     model.next_sentence_labels: batch_next_sentence_label,
                     model.gen_ebd_mask_position: 4}
        if index == 0:
            gen_ebd4 = sess.run(model.mask_gen_ebd, feed_dict)
            gen_ebd4 = np.array(gen_ebd4)
        else:
            temp = sess.run(model.mask_gen_ebd, feed_dict)
            temp = np.array(temp)
            gen_ebd4 = np.concatenate([gen_ebd4, temp], 0)

    print('mask position 5 embedding generate...')
    for index in tqdm.tqdm(batch_test_index5):
        batch_input_ids, batch_input_mask, batch_segment_ids, batch_masked_lm_positions, batch_masked_lm_ids, batch_masked_lm_weights, batch_next_sentence_label = gendata_meta_path.batch_gen(
            test_batches5, index, parameter_config.batch_size)
        feed_dict = {model.input_ids: batch_input_ids,
                     model.input_mask: batch_input_mask,
                     model.token_type_ids: batch_segment_ids,
                     model.masked_lm_positions: batch_masked_lm_positions,
                     model.masked_lm_ids: batch_masked_lm_ids,
                     model.masked_lm_weights: batch_masked_lm_weights,
                     model.next_sentence_labels: batch_next_sentence_label,
                     model.gen_ebd_mask_position: 5}
        if index == 0:
            gen_ebd5 = sess.run(model.mask_gen_ebd, feed_dict)
            gen_ebd5 = np.array(gen_ebd5)
        else:
            temp = sess.run(model.mask_gen_ebd, feed_dict)
            temp = np.array(temp)
            gen_ebd5 = np.concatenate([gen_ebd5, temp], 0)

    print('mask position 6 embedding generate...')
    for index in tqdm.tqdm(batch_test_index6):
        batch_input_ids, batch_input_mask, batch_segment_ids, batch_masked_lm_positions, batch_masked_lm_ids, batch_masked_lm_weights, batch_next_sentence_label = gendata_meta_path.batch_gen(
            test_batches6, index, parameter_config.batch_size)
        feed_dict = {model.input_ids: batch_input_ids,
                     model.input_mask: batch_input_mask,
                     model.token_type_ids: batch_segment_ids,
                     model.masked_lm_positions: batch_masked_lm_positions,
                     model.masked_lm_ids: batch_masked_lm_ids,
                     model.masked_lm_weights: batch_masked_lm_weights,
                     model.next_sentence_labels: batch_next_sentence_label,
                     model.gen_ebd_mask_position: 6}
        if index == 0:
            gen_ebd6 = sess.run(model.mask_gen_ebd, feed_dict)
            gen_ebd6 = np.array(gen_ebd6)
        else:
            temp = sess.run(model.mask_gen_ebd, feed_dict)
            temp = np.array(temp)
            gen_ebd6 = np.concatenate([gen_ebd6, temp], 0)

    print('mask position 7 embedding generate...')
    for index in tqdm.tqdm(batch_test_index7):
        batch_input_ids, batch_input_mask, batch_segment_ids, batch_masked_lm_positions, batch_masked_lm_ids, batch_masked_lm_weights, batch_next_sentence_label = gendata_meta_path.batch_gen(
            test_batches7, index, parameter_config.batch_size)
        feed_dict = {model.input_ids: batch_input_ids,
                     model.input_mask: batch_input_mask,
                     model.token_type_ids: batch_segment_ids,
                     model.masked_lm_positions: batch_masked_lm_positions,
                     model.masked_lm_ids: batch_masked_lm_ids,
                     model.masked_lm_weights: batch_masked_lm_weights,
                     model.next_sentence_labels: batch_next_sentence_label,
                     model.gen_ebd_mask_position: 7}
        if index == 0:
            gen_ebd7 = sess.run(model.mask_gen_ebd, feed_dict)
            gen_ebd7 = np.array(gen_ebd7)
        else:
            temp = sess.run(model.mask_gen_ebd, feed_dict)
            temp = np.array(temp)
            gen_ebd7 = np.concatenate([gen_ebd7, temp], 0)

    return gen_ebd0, gen_ebd1, gen_ebd2, gen_ebd3, gen_ebd4, gen_ebd5, gen_ebd6, gen_ebd7


def split_and_save_meta_path_mask_task_data(config, ebd0, ebd1, ebd2, ebd3, ebd4, ebd5, ebd6, ebd7):
    user0 = ebd0[0:config.num_users, :]
    item0 = ebd0[config.num_users:, :]

    user1 = ebd1[0:config.num_users, :]
    item1 = ebd1[config.num_users:, :]

    user2 = ebd2[0:config.num_users, :]
    item2 = ebd2[config.num_users:, :]

    user3 = ebd3[0:config.num_users, :]
    item3 = ebd3[config.num_users:, :]

    user4 = ebd4[0:config.num_users, :]
    item4 = ebd4[config.num_users:, :]

    user5 = ebd5[0:config.num_users, :]
    item5 = ebd5[config.num_users:, :]

    user6 = ebd6[0:config.num_users, :]
    item6 = ebd6[config.num_users:, :]

    user7 = ebd7[0:config.num_users, :]
    item7 = ebd7[config.num_users:, :]

    np.save('bert_mask_user0.npy', user0)
    np.save('bert_mask_item0.npy', item0)

    np.save('bert_mask_user1.npy', user1)
    np.save('bert_mask_item1.npy', item1)

    np.save('bert_mask_user2.npy', user2)
    np.save('bert_mask_item2.npy', item2)

    np.save('bert_mask_user3.npy', user3)
    np.save('bert_mask_item3.npy', item3)

    np.save('bert_mask_user4.npy', user4)
    np.save('bert_mask_item4.npy', item4)

    np.save('bert_mask_user5.npy', user5)
    np.save('bert_mask_item5.npy', item5)

    np.save('bert_mask_user6.npy', user6)
    np.save('bert_mask_item6.npy', item6)

    np.save('bert_mask_user7.npy', user7)
    np.save('bert_mask_item7.npy', item7)

    user = (user0 + user1 + user2 + user3 + user4 + user5 + user6 + user7) / 8
    item = (item0 + item1 + item2 + item3 + item4 + item5 + item6 + item7) / 8
    return user, item

def get_cos_similar(v1: list, v2: list):
    num = float(np.dot(v1, v2))  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0


def dynamic_mask_training_file(epoch, Config):
    meta_user_embedding = np.load(Config.meta_user_ebd_path)
    meta_item_embedding = np.load(Config.meta_item_ebd_path)

    current_user_embedding = np.load(Config.current_user_ebd_path)
    current_item_embedding = np.load(Config.current_item_ebd_path)

    concat_user_embedding = np.concatenate((meta_user_embedding, current_user_embedding), axis=1)
    concat_item_embedding = np.concatenate((meta_item_embedding, current_item_embedding), axis=1)



    np.random.seed(epoch)
    with open(Config.data_path + Config.dataset + '/train.txt', 'r') as f:
        with open('exp.txt', 'w') as writer:
        # with open(Config.data_path + Config.dataset + '/mask_train.txt', 'w') as writer:
            line = f.readline()
            while line != "" and line != None:
                arr = line.strip().split(' ')
                user, items = int(arr[0]), arr[1:]
                # print(user)
                current_cosine = []
                current_item_id = []
                current_sample_neighbor = []
                user_embedding = concat_user_embedding[user, :]

                # print(user_embedding)
                for item in items:
                    item = int(item)
                    current_item_id.append(item)
                    item_embedding = concat_item_embedding[item]
                    cos = get_cos_similar(user_embedding, item_embedding)
                    # print(cos)
                    current_cosine.append(cos)

                # index = current_cosine.index(max(current_cosine))
                index = heapq.nlargest(3, range(len(current_cosine)), current_cosine.__getitem__)
                # print(current_cosine)
                #
                # print(index)

                for per_index in index:
                    current_sample_neighbor.append(current_item_id[per_index])
                # print(current_item_id)
                # print(current_sample_neighbor)

                # # perform sample strategy
                # if len(items) >= Config.few_shot_number:
                #     sampled_items = np.random.choice(items, Config.few_shot_number, replace=False)
                # else:
                #     sampled_items = np.random.choice(items, Config.few_shot_number, replace=True)

                writer.write(str(user) + ' ')
                for item_index in range(len(current_sample_neighbor)):
                    if item_index != len(current_sample_neighbor) - 1:
                        writer.write(str(current_sample_neighbor[item_index]) + ' ')
                    else:
                        writer.write(str(current_sample_neighbor[item_index]))
                writer.write('\n')

                line = f.readline()

def load_current_embedding(model, sess, config):
    current_user_embedding = np.array(sess.run(model.ua_embeddings))
    current_item_embedding = np.array(sess.run(model.ia_embeddings))
    np.save(config.current_user_ebd_path, current_user_embedding)
    np.save(config.current_item_ebd_path, current_item_embedding)



if __name__ == '__main__':

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:

        # pseudo code
        '''
        for epoch in Epoch:
            random mask some users/items, prepare adj matrix etc
            for subepoch in SubEpoch:
                perform GNN aggregation operation

        '''

        '''
        global parameters
        '''
        parameter_config = Model_Config()
        pretrain_data = load_pretrained_data(parameter_config)
        initial_data = load_initial_data(parameter_config)

        random_mask_training_file(epoch=0, Config=parameter_config)
        begin_time = time()
        data_generator = Data(path=parameter_config.data_path + parameter_config.dataset,
                              batch_size=parameter_config.batch_size,
                              task_name=parameter_config.pretext_task_name)
        print('time continue is %.4f' % float(time() - begin_time))
        config = dict()
        config['n_users'] = data_generator.n_users
        config['n_items'] = data_generator.n_items

        """
        *********************************************************
        Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
        """
        plain_adj, norm_adj, mean_adj, pre_adj = data_generator.get_adj_mat()
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

        GeneralConv = Recommender(config, pretrain_data, initial_data, parameter_config.conv_name, parameter_config, True)
        print(GeneralConv.n_users, GeneralConv.n_items)
        saver = tf.train.Saver(max_to_keep=3)
        sess.run(tf.global_variables_initializer())

        logging.info("initialized pretrainer...")
        for epoch in range(1, 45):
            print('epoch is %d' % epoch)
            ################################ step 1 ###################################
            # step 1: train meta learner function f, g, and generate meta embedding
            train_begin = time()
            if epoch ==0:
                random_mask_training_file(0, parameter_config)
            else:
                dynamic_mask_training_file(epoch, parameter_config)
            train_end = time() - train_begin
            print('train time is % .4f' % train_end)

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

            GeneralConv.data_config = config
            GeneralConv.norm_adj = config['norm_adj']

            print('Embedding Reconstruction with GNN')
            training_user_task(GeneralConv, sess)
            training_item_task(GeneralConv, sess)

            # train meta aggregator
            generate_user_task_data(GeneralConv, sess)
            generate_item_task_data(GeneralConv, sess)
            # concatenate_user_item_ebd()
            meta_data = load_meta_data(parameter_config)

            GeneralConv.weights['meta_user_embedding'] = meta_data['user_embed']
            GeneralConv.weights['meta_item_embedding'] = meta_data['item_embed']

            # train final embedding reconstruction task
            # training_batch_item_reconstruct_task(GeneralConv, sess, parameter_config)
            training_batch_user_reconstruct_task(GeneralConv, sess, parameter_config)


            print('start fine-tuning...')

            # if epoch  == 0:

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

            GeneralConv.data_config = config
            GeneralConv.norm_adj = config['norm_adj']

            training_downstream_recommendation_task(GeneralConv, sess, epoch, parameter_config)

            load_current_embedding(GeneralConv, sess, parameter_config)

        # end step 1####################################################################

        ############# step 5 node-level mutual information ############

        print('training user mutual info task ...')
        training_batch_user_mutual_info_task(GeneralConv, sess)
        print('training item mutual info task ...')
        training_batch_item_mutual_info_task(GeneralConv, sess)

        # generate mutual embedding
        generate_mutual_user_task_data(GeneralConv, sess)
        generate_mutual_item_task_data(GeneralConv, sess)
        concatenate_mutual_user_item_ebd()
        mutual_data = load_mutual_data()
        GeneralConv.mutual_user_ebd = mutual_data['user_embed']
        GeneralConv.mutual_item_ebd = mutual_data['item_embed']
        ############ end step 5 ##############################

        ############## step 2  subgraph-level contrastive ############
        print('training user contrastive info task ...')
        training_batch_user_contrastive_coding_task(GeneralConv, sess)
        print('training item contrastive info task ...')
        training_batch_item_contrastive_coding_task(GeneralConv, sess)

        # generate contrastive embedding
        generate_contrastive_user_task_data(GeneralConv, sess)
        generate_contrastive_item_task_data(GeneralConv, sess)
        concatenate_contrastive_user_item_ebd()
        contrastive_data = load_contrastive_data()
        GeneralConv.contrastive_user_ebd = contrastive_data['user_embed']
        GeneralConv.contrastive_item_ebd = contrastive_data['item_embed']
        #############  end step 2 #####################################

        ######################## step 3  meta path pretext task ###################################
        # first perform bert mask, next sentence prediction task

        gendata_meta_path.main()
        ####################################################################################################
        #################           perform meta path pertraining task   ###################################
        batches_meta_path = gendata_meta_path.generate_batch_meta_path(parameter_config)
        num_batch_meta_path = np.array(batches_meta_path[0]).shape[0] // parameter_config.batch_size
        print('num batch for meta path is %d' % num_batch_meta_path)
        batch_index_meta_path = range(num_batch_meta_path)
        print('start training meta path pretraining task...')
        loss = training_batch_meta_path(batch_index_meta_path, GeneralConv, sess, batches_meta_path,
                                        parameter_config.batch_size)

        for mask_index in range(parameter_config.walk_length_meta_path):
            count = -1
            for count, line in enumerate(open('./meta_path_all_%d.csv' % mask_index, 'r')):
                pass
                count += 1

            flag = os.path.exists('./meta_path_all_%d.csv' % mask_index)
            if flag == False or count != parameter_config.vocab_size + 1:
                print('meta path does not exist or is not complete..., run random walk')
                gendata_meta_path.main()
                break
            else:
                print('meta path %d already exists...' % mask_index)
        print('deal with meta path random walk file... place [mask] symbol and write target id')
        gendata_meta_path.deal_context_file('./meta_path_all_0.csv', './meta_path_all_1.csv',
                                            './meta_path_all_2.csv', './meta_path_all_3.csv',
                                            './meta_path_all_4.csv', './meta_path_all_5.csv',
                                            './meta_path_all_6.csv', './meta_path_all_7.csv',
                                            './d_meta_path_all_0.csv', './d_meta_path_all_1.csv',
                                            './d_meta_path_all_2.csv', './d_meta_path_all_3.csv',
                                            './d_meta_path_all_4.csv', './d_meta_path_all_5.csv',
                                            './d_meta_path_all_6.csv', './d_meta_path_all_7.csv')

        print('load context dict...')
        context_meta_path_dict = gendata_meta_path.read_dealt_context_file('./d_meta_path_all_0.csv',
                                                                           './d_meta_path_all_1.csv',
                                                                           './d_meta_path_all_2.csv',
                                                                           './d_meta_path_all_3.csv',
                                                                           './d_meta_path_all_4.csv',
                                                                           './d_meta_path_all_5.csv',
                                                                           './d_meta_path_all_6.csv',
                                                                           './d_meta_path_all_7.csv', )

        instances_test0, instances_test1, instances_test2, instances_test3, instances_test4, instances_test5, instances_test6, instances_test7 = gendata_meta_path.load_placeholder_data_test(
            parameter_config, context_meta_path_dict, 1)

        test_batches0 = gendata_meta_path.generate_batch(parameter_config, instances_test0)
        test_batches1 = gendata_meta_path.generate_batch(parameter_config, instances_test1)
        test_batches2 = gendata_meta_path.generate_batch(parameter_config, instances_test2)
        test_batches3 = gendata_meta_path.generate_batch(parameter_config, instances_test3)
        test_batches4 = gendata_meta_path.generate_batch(parameter_config, instances_test4)
        test_batches5 = gendata_meta_path.generate_batch(parameter_config, instances_test5)
        test_batches6 = gendata_meta_path.generate_batch(parameter_config, instances_test6)
        test_batches7 = gendata_meta_path.generate_batch(parameter_config, instances_test7)

        all_ebd0, all_ebd1, all_ebd2, all_ebd3, all_ebd4, all_ebd5, all_ebd6, all_ebd7 = generate_meta_path_mask_task_data(
            GeneralConv, sess, test_batches0, test_batches1,
            test_batches2, test_batches3, test_batches4, test_batches5, test_batches6, test_batches7,
            parameter_config)
        print(all_ebd0.shape)
        print(all_ebd1.shape)
        print(all_ebd2.shape)
        print(all_ebd3.shape)
        print(all_ebd0.shape)
        print(all_ebd4.shape)
        print(all_ebd5.shape)
        print(all_ebd6.shape)
        print(all_ebd7.shape)

        meta_path_mask_user, mata_path_mask_item = split_and_save_meta_path_mask_task_data(parameter_config,
                                                                                           all_ebd0, all_ebd1,
                                                                                           all_ebd2, all_ebd3,
                                                                                           all_ebd4, all_ebd5,
                                                                                           all_ebd6, all_ebd7)
        GeneralConv.meta_path_user_ebd = meta_path_mask_user
        GeneralConv.meta_path_item_ebd = mata_path_mask_item

        #################### end step 3 ##########################################################################

        ################### step 4 bert mask task and next sentence prediction task ###########################

        for mask_index in range(parameter_config.walk_length):
            # count = -1
            # for count, line in enumerate(open('./recommendation_all_%d.csv' % mask_index, 'r')):
            #     pass
            #     count += 1

            flag = os.path.exists('./recommendation_all_%d.csv' % mask_index)

            if flag == False or count != parameter_config.vocab_size + 1:
                print('bert mask meta path does not exist or is not complete..., run random walk')
                gendata_bert.main(args=parameter_config, write_test=True)
                break
            else:
                print('bert mask meta path %d already exists...' % mask_index)

        print('deal with bert mask random walk file... place [mask] symbol and write target id')
        gendata_bert.deal_context_file('./recommendation_all_0.csv', './recommendation_all_1.csv',
                                       './recommendation_all_2.csv',
                                       './recommendation_all_3.csv',
                                       './d_recommendation_all_0.csv', './d_recommendation_all_1.csv',
                                       './d_recommendation_all_2.csv', './d_recommendation_all_3.csv')

        # print('sort all users and items context...')
        # gendata_bert.sort_all_users_items_file('./recommendation_all_0.csv',
        #                                        './recommendation_all_1.csv',
        #                                        './recommendation_all_2.csv',
        #                                        './recommendation_all_3.csv',
        #                                        './sorted_all1_.csv', './sorted_all_2.csv',
        #                                        './sorted_all_3.csv', './sorted_all_4.csv')

        print('load context dict...')
        context_dict = gendata_bert.read_dealt_context_file('./d_recommendation_all_0.csv',
                                                            './d_recommendation_all_1.csv',
                                                            './d_recommendation_all_2.csv',
                                                            './d_recommendation_all_3.csv')

        # context_dict_test = gendata_bert.read_dealt_context_file_test('./recommendation_all_0.csv',
        #                                                               './recommendation_all_1.csv',
        #                                                               './recommendation_all_2.csv',
        #                                                               './recommendation_all_3.csv', )

        # # training instances
        print('prepare training data...')
        user_input, item_input, labels = gendata_bert.get_train_instances(args=parameter_config,
                                                                          random_seed=1,
                                                                          train_adjlist='./Data/movielens_dataset/bert_mask/reindex_train.txt',
                                                                          negative_num=1)
        instances = gendata_bert.load_placeholder_data(parameter_config, context_dict, user_input, item_input,
                                                       labels,
                                                       1)

        instances_test0, instances_test1, instances_test2, instances_test3 = gendata_bert.load_placeholder_data_test(
            parameter_config, context_dict, user_input, item_input, labels,
            1)
        # for (inst_index, instance) in enumerate(instances):
        #     print(instance)
        batches = gendata_bert.generate_batch(parameter_config, instances)
        num_batch = np.array(batches[0]).shape[0] // parameter_config.batch_size
        print('num_batch is %d' % num_batch)
        batch_index = range(num_batch)
        print('start training...')
        loss = training_batch(batch_index, GeneralConv, sess, batches, parameter_config.batch_size)

        test_batches0 = gendata_bert.generate_batch(parameter_config, instances_test0)
        test_batches1 = gendata_bert.generate_batch(parameter_config, instances_test1)
        test_batches2 = gendata_bert.generate_batch(parameter_config, instances_test2)
        test_batches3 = gendata_bert.generate_batch(parameter_config, instances_test3)

        all_ebd0, all_ebd1, all_ebd2, all_ebd3 = generate_bert_mask_task_data(GeneralConv, sess, test_batches0,
                                                                              test_batches1,
                                                                              test_batches2, test_batches3,
                                                                              parameter_config)
        print(all_ebd0.shape)
        print(all_ebd1.shape)
        print(all_ebd2.shape)
        print(all_ebd3.shape)

        bert_mask_user, bert_mask_item = split_and_save_bert_mask_task_data(parameter_config, all_ebd0,
                                                                            all_ebd1, all_ebd2, all_ebd3)
        GeneralConv.bert_mask_user_ebd = bert_mask_user
        GeneralConv.bert_mask_item_ebd = bert_mask_item

        # test instances
        # test
        # (1) ###############################################################################################################
        # all ranking... the instances is extremely huge, and runing this costs lots of time
        # print('prepare test data...')
        # test_user_input, test_user_pos_dict, test_user_ground_truth_dict = gendata.get_test_instances(
        #     bert_config, train_adjlist='./reindex_train.txt', test_adjlist='./reindex_test.txt')
        # print('start preprare test instances...')
        # test_instances = gendata.load_test_placeholder_data(bert_config, context_dict, test_user_input)
        # test_batches = gendata.generate_test_batch(bert_config, test_instances)

        # (2) ################################################################################################################
        # positive examples plus negative examples = 100
        print('prepare test data...')
        test_user_input, test_user_pos_dict, test_user_ground_truth_dict, negative_sampling_user_dict = gendata_bert.get_test_instances_negative_sampling(
            args=parameter_config, train_adjlist='./Data/movielens_dataset/bert_mask/reindex_train.txt',
            test_adjlist='./Data/movielens_dataset/bert_mask/reindex_test.txt')
        print('start preprare test instances...')
        test_instances = gendata_bert.load_test_placeholder_data_negative_sampling(parameter_config,
                                                                                   context_dict,
                                                                                   test_user_input,
                                                                                   negative_sampling_user_dict)
        print(len(test_instances))
        test_batches = gendata_bert.generate_test_batch(parameter_config, test_instances)

        # precision, recall, ndcg, hr = evaluate(bert_config, model, sess, test_batches, test_user_pos_dict,
        #                                        test_user_ground_truth_dict)

        # 1+99 evaluate cost lots of time
        # precision, recall, ndcg, hr = evaluate_negative_sampling(parameter_config, GeneralConv, sess,
        #                                                          test_batches,
        #                                                          test_user_pos_dict,
        #                                                          test_user_ground_truth_dict,
        #                                                          negative_sampling_user_dict)
        # print('Iteration %d: precision = %.4f, recall = %.4f, ndcg = %.4f, hr = %.4f, training loss = %.4f'
        #       % (epoch, precision, recall, ndcg, hr, loss))

        bert_mask_user, bert_mask_item = split_and_save_bert_mask_task_data(parameter_config, all_ebd0,
                                                                            all_ebd1, all_ebd2, all_ebd3)
        GeneralConv.bert_mask_user_ebd = bert_mask_user
        GeneralConv.bert_mask_item_ebd = bert_mask_item

        # 1+99 evaluate cost lots of time
        # precision, recall, ndcg, hr = evaluate_negative_sampling(parameter_config, GeneralConv, sess,
        #                                                          test_batches,
        #                                                          test_user_pos_dict,
        #                                                          test_user_ground_truth_dict,
        #                                                          negative_sampling_user_dict)
        # print('Iteration %d: precision = %.4f, recall = %.4f, ndcg = %.4f, hr = %.4f, training loss = %.4f'
        #       % (epoch, precision, recall, ndcg, hr, loss))

        ############################  end of step 4 ###########################################################

        for epoch in range(1, 45):
            random_mask_training_file(epoch, parameter_config)

            data_generator = Data(path=parameter_config.data_path + parameter_config.dataset,
                                  batch_size=parameter_config.batch_size,
                                  pretext_task_name='downstream_task')
            config = dict()
            config['n_users'] = data_generator.n_users
            config['n_items'] = data_generator.n_items

            """
            *********************************************************
            Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
            """
            plain_adj, norm_adj, mean_adj, pre_adj = data_generator.get_adj_mat()
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

            GeneralConv.data_config = config
            GeneralConv.norm_adj = config['norm_adj']

            generate_user_task_data(GeneralConv, sess)
            generate_item_task_data(GeneralConv, sess)
            # concatenate_user_item_ebd()
            meta_data = load_meta_data(parameter_config)

            GeneralConv.weights['meta_user_embedding'] = meta_data['user_embed']
            GeneralConv.weights['meta_item_embedding'] = meta_data['item_embed']

            training_batch_item_reconstruct_task(GeneralConv, sess, parameter_config)
            training_batch_user_reconstruct_task(GeneralConv, sess, parameter_config)





            '''
            fine-tuning recommendation task,
            first load original dataset, and get the adj matrix (inculded in the config dict),
            then we pass the config pretrain_data into the finetuning recommender class
            '''
            data_generator = Data(path=parameter_config.data_path + parameter_config.dataset,
                                  batch_size=parameter_config.batch_size,
                                  pretext_task_name='downstream_recommendation_task')

            config = dict()
            config['n_users'] = data_generator.n_users
            config['n_items'] = data_generator.n_items

            """
            *********************************************************
            Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
            """
            plain_adj, norm_adj, mean_adj, pre_adj = data_generator.get_adj_mat()
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

            # saver.restore(sess, tf.train.get_checkpoint_state(
            #     os.path.dirname(parameter_config.checkpoint_path_downstream + 'checkpoint')).model_checkpoint_path)
            # print('training downstream task with restored checkpoint...')


