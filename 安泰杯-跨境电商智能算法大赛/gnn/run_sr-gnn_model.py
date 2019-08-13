# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:46:52 2019

@author: sutianpei
"""

from __future__ import division
import numpy as np
import pandas as pd
import copy
import random
import joblib
import argparse
import datetime
import tensorflow as tf
import math
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print ('start model training.')
def data_masks(all_usr_pois, item_tail):
    us_lens = np.array([len(upois) for upois in all_usr_pois])
    len_max = max(us_lens)
    us_pois = [np.append(upois, item_tail * (len_max - le)) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [np.append([1] * le, [0] * (len_max - le)) for le in us_lens]
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

class Data():
    def __init__(self, data, sub_graph=False, sparse=False, shuffle=False):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.sub_graph = sub_graph
        self.sparse = sparse

    #剩余不足batch size的数据归入最后一个batch
    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length-batch_size, self.length)
        return slices

    def get_slice(self, index):
        if 1:
            items, n_node, A_in, A_out, alias_inputs = [], [], [], [], []
            for u_input in self.inputs[index]:
                n_node.append(len(np.unique(u_input)))
            max_n_node = np.max(n_node)
            for u_input in self.inputs[index]:
                node = np.unique(u_input)
                items.append(node.tolist() + (max_n_node - len(node)) * [0])
                u_A = np.zeros((max_n_node, max_n_node))
                for i in np.arange(len(u_input) - 1):
                    if u_input[i + 1] == 0:
                        break
                    u = np.where(node == u_input[i])[0][0]
                    v = np.where(node == u_input[i + 1])[0][0]
                    u_A[u][v] = 1
                u_sum_in = np.sum(u_A, 0)
                u_sum_in[np.where(u_sum_in == 0)] = 1
                u_A_in = np.divide(u_A, u_sum_in)
                u_sum_out = np.sum(u_A, 1)
                u_sum_out[np.where(u_sum_out == 0)] = 1
                u_A_out = np.divide(u_A.transpose(), u_sum_out)

                A_in.append(u_A_in)
                A_out.append(u_A_out)
                alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
            return A_in, A_out, alias_inputs, items, self.mask[index], self.targets[index]
        else:
            return self.inputs[index], self.mask[index], self.targets[index]

class Model(object):
    def __init__(self, hidden_size=100, out_size=100, batch_size=100, nonhybrid=True):
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.batch_size = batch_size
        self.mask = tf.placeholder(dtype=tf.float32)
        self.alias = tf.placeholder(dtype=tf.int32)  # 给给每个输入重新
        self.item = tf.placeholder(dtype=tf.int32)   # 重新编号的序列构成的矩阵
        self.tar = tf.placeholder(dtype=tf.int32)
        self.nonhybrid = nonhybrid
        self.stdv = 1.0 / math.sqrt(self.hidden_size)

        self.nasr_w1 = tf.get_variable('nasr_w1', [self.out_size, self.out_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_w2 = tf.get_variable('nasr_w2', [self.out_size, self.out_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_v = tf.get_variable('nasrv', [1, self.out_size], dtype=tf.float32,
                                      initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_b = tf.get_variable('nasr_b', [self.out_size], dtype=tf.float32, initializer=tf.zeros_initializer())

    def forward(self, re_embedding, train=True):
        rm = tf.reduce_sum(self.mask, 1)
        last_id = tf.gather_nd(self.alias, tf.stack([tf.range(self.batch_size), tf.to_int32(rm)-1], axis=1))
        last_h = tf.gather_nd(re_embedding, tf.stack([tf.range(self.batch_size), last_id], axis=1))
        seq_h = tf.stack([tf.nn.embedding_lookup(re_embedding[i], self.alias[i]) for i in range(self.batch_size)],
                         axis=0)                                                           #batch_size*T*d
        last = tf.matmul(last_h, self.nasr_w1)
        seq = tf.matmul(tf.reshape(seq_h, [-1, self.out_size]), self.nasr_w2)
        last = tf.reshape(last, [self.batch_size, 1, -1])
        m = tf.nn.sigmoid(last + tf.reshape(seq, [self.batch_size, -1, self.out_size]) + self.nasr_b)
        coef = tf.matmul(tf.reshape(m, [-1, self.out_size]), self.nasr_v, transpose_b=True) * tf.reshape(
            self.mask, [-1, 1])
        b = self.embedding[1:]
        if not self.nonhybrid:
            ma = tf.concat([tf.reduce_sum(tf.reshape(coef, [self.batch_size, -1, 1]) * seq_h, 1),
                            tf.reshape(last, [-1, self.out_size])], -1)
            self.B = tf.get_variable('B', [2 * self.out_size, self.out_size],
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
            y1 = tf.matmul(ma, self.B)
            logits = tf.matmul(y1, b, transpose_b=True)
        else:
            ma = tf.reduce_sum(tf.reshape(coef, [self.batch_size, -1, 1]) * seq_h, 1)
            logits = tf.matmul(ma, b, transpose_b=True)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.tar - 1, logits=logits))
        self.vars = tf.trainable_variables()
        if train:
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.vars if v.name not
                               in ['bias', 'gamma', 'b', 'g', 'beta']]) * self.L2
            loss = loss + lossL2
        return loss, logits

    def run(self, fetches, tar, item, adj_in, adj_out, alias, mask):
        return self.sess.run(fetches, feed_dict={self.tar: tar, self.item: item, self.adj_in: adj_in,
                                                 self.adj_out: adj_out, self.alias: alias, self.mask: mask})


class GGNN(Model):
    def __init__(self,hidden_size=100, out_size=100, batch_size=300, n_node=None,
                 lr=None, l2=None, step=1, decay=None, lr_dc=0.1, nonhybrid=False):
        super(GGNN,self).__init__(hidden_size, out_size, batch_size, nonhybrid)
        self.embedding = tf.get_variable(shape=[n_node, hidden_size], name='embedding', dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.adj_in = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
        self.adj_out = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
        self.n_node = n_node
        self.L2 = l2
        self.step = step
        self.nonhybrid = nonhybrid
        self.W_in = tf.get_variable('W_in', shape=[self.out_size, self.out_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_in = tf.get_variable('b_in', [self.out_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.W_out = tf.get_variable('W_out', [self.out_size, self.out_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_out = tf.get_variable('b_out', [self.out_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        with tf.variable_scope('ggnn_model', reuse=None):
            self.loss_train, _ = self.forward(self.ggnn())
        with tf.variable_scope('ggnn_model', reuse=True):
            self.loss_test, self.score_test = self.forward(self.ggnn(), train=False)
        self.global_step = tf.Variable(0)
        self.learning_rate = tf.train.exponential_decay(lr, global_step=self.global_step, decay_steps=decay,
                                                        decay_rate=lr_dc, staircase=True)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_train, global_step=self.global_step)
        # gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def ggnn(self):
        fin_state = tf.nn.embedding_lookup(self.embedding, self.item)
        cell = tf.nn.rnn_cell.GRUCell(self.out_size)
        with tf.variable_scope('gru'):
            for i in range(self.step):
                fin_state = tf.reshape(fin_state, [self.batch_size, -1, self.out_size])
                fin_state_in = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),
                                                    self.W_in) + self.b_in, [self.batch_size, -1, self.out_size])
                fin_state_out = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),
                                                     self.W_out) + self.b_out, [self.batch_size, -1, self.out_size])
                av = tf.concat([tf.matmul(self.adj_in, fin_state_in),
                                tf.matmul(self.adj_out, fin_state_out)], axis=-1)
                state_output, fin_state = \
                    tf.nn.dynamic_rnn(cell, tf.expand_dims(tf.reshape(av, [-1, 2*self.out_size]), axis=1),
                                      initial_state=tf.reshape(fin_state, [-1, self.out_size]))
        return tf.reshape(fin_state, [self.batch_size, -1, self.out_size])

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
# parser.add_argument('--validation', action='store_true', help='validation')
# parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
# parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
# parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
# parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
# parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
# parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
# parser.add_argument('--nonhybrid', action='store_true', help='global preference')
# parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
# parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
# opt = parser.parse_args()

para = {}
para['validation'] = 'store_true'
para['epoch'] = 1
para['batchSize'] = 256
para['hiddenSize'] = 100
para['validation'] = 'store_true'
para['l2'] = 1e-5
para['lr'] = 0.001
para['step'] = 1
para['nonhybrid'] = 'store_true'
para['lr_dc'] = 0.1
para['lr_dc_step'] = 3
n_node = np.loadtxt('sr-gnn_tmp.txt', dtype=int)

train_label = joblib.load(open('train_label', 'rb'))
train_seq = joblib.load(open('train_seq', 'rb'))
item_corr_dict = joblib.load(open('item_corr_dict', 'rb'))
test_seq = joblib.load(open('test_seq', 'rb'))

item_corr_dict2 = {j:i for i, j in item_corr_dict.items()}

print ('processing data.')
train_data = Data((train_seq, train_label), shuffle=True)
test_data = Data((test_seq, train_label), shuffle=False)
model = GGNN(hidden_size=para['hiddenSize'], out_size=para['hiddenSize'], 
             batch_size=para['batchSize'], n_node=n_node, lr=para['lr'], 
             l2=para['l2'],  step=para['step'],
             decay=para['lr_dc_step'] * len(train_data.inputs) / para['batchSize'],
             lr_dc=para['lr_dc'], nonhybrid=para['nonhybrid'])

best_result = [0, 0]
best_epoch = [0, 0]
for epoch in range(para['epoch']):
    print('epoch: ', epoch)
    slices = train_data.generate_batch(model.batch_size)
    fetches = [model.opt, model.loss_train, model.global_step]
    print('epoch: %s start training'%epoch)
    loss_ = []
    for i, j in zip(slices, np.arange(len(slices))):
        adj_in, adj_out, alias, item, mask, targets = train_data.get_slice(i)
        _, loss, _ = model.run(fetches, targets, item, adj_in, adj_out, alias,  mask)
        loss_.append(loss)
    loss = np.mean(loss_)
slices = test_data.generate_batch(model.batch_size)
print('start predicting: ')
hit, mrr, test_loss_ = [], [],[]
ans = pd.DataFrame(np.zeros((0, 30), dtype=int))
for i, j in zip(slices, np.arange(len(slices))):
    adj_in, adj_out, alias, item, mask, targets = test_data.get_slice(i)
    scores, test_loss = model.run([model.score_test, model.loss_test], targets, item, adj_in, adj_out, alias,  mask)
    test_loss_.append(test_loss)
    index = np.argsort(scores, 1)[:, -30:]
    ans = pd.concat([ans, pd.DataFrame(index)], axis=0).reset_index(drop=True)
ans2 = ans.applymap(lambda x: item_corr_dict2[x+1])
ans2.to_csv('ans.csv', index=None, header=None)
#     #print (aaaa)
#     for score, target in zip(index, targets):
#         hit.append(np.isin(target - 1, score))
#         if len(np.where(score == target - 1)[0]) == 0:
#             mrr.append(0)
#         else:
#             mrr.append(1 / (20-np.where(score == target - 1)[0][0]))
# hit = np.mean(hit)*100
# mrr = np.mean(mrr)*100
# test_loss = np.mean(test_loss_)
# if hit >= best_result[0]:
#     best_result[0] = hit
#     best_epoch[0] = epoch
# if mrr >= best_result[1]:
#     best_result[1] = mrr
#     best_epoch[1]=epoch
# print('train_loss:\t%.4f\ttest_loss:\t%4f\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'%
#       (loss, test_loss, best_result[0], best_result[1], best_epoch[0], best_epoch[1]))

