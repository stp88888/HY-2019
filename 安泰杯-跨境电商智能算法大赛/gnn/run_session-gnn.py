# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 13:15:29 2019

@author: sutianpei
"""


import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
import joblib
import copy
import time

#是否初始化
inital = 0
#item最少出现次数，少于此数目抛弃
item_least_num = 2
#drop_user只有在inital为1时才有效
drop_user = 1
#一旦修改了drop_user_limit，必须重新初始化
drop_user_limit = 1000

time1 = time.time()

print ('starting data processing.')
if inital:
    print ('initializing.')
    print ('para: {drop_user_limit: %s}'%drop_user_limit)
    data_train = pd.read_csv('Antai_AE_round1_train_20190626.csv')
    data_test = pd.read_csv('Antai_AE_round1_test_20190626.csv')
    data_item = pd.read_csv('Antai_AE_round1_item_attr_20190626.csv')
    data_all = pd.concat([data_train, data_test], axis=0)

    if drop_user == 1:
    #剔除购买商品数量超过一定数量的用户
        user_drop = set(data_all[['buyer_admin_id','item_id']]
                        .groupby(['buyer_admin_id'])
                        .count()
                        .reset_index(drop=False)
                        .query('item_id > @drop_user_limit')
                        .buyer_admin_id)

    item_user = defaultdict(int)
    for _, tmp in data_all.iterrows():
        if tmp['buyer_admin_id'] not in user_drop:
            item_user[tmp['item_id']] += 1
    pickle.dump(item_user, open('item_user.pkl', 'wb'))

    # tra = (data_train[['buyer_admin_id', 'item_id', 'irank']]
    #     .sort_values('irank')[['buyer_admin_id', 'item_id']]
    #     .groupby('buyer_admin_id'))
    # tra = {i['buyer_admin_id'].iloc[0]: list(i['item_id']) for _, i in tra}
    # tes = (data_test[['buyer_admin_id', 'item_id', 'irank']]
    #     .sort_values('irank')[['buyer_admin_id', 'item_id']]
    #     .groupby('buyer_admin_id'))
    # tes = {i['buyer_admin_id'].iloc[0]: list(i['item_id']) for _, i in tes}
    data = (data_all[['buyer_admin_id', 'item_id', 'irank']]
        .sort_values('irank')[['buyer_admin_id', 'item_id']]
        .groupby('buyer_admin_id'))
    data = {i['buyer_admin_id'].iloc[0]: list(i['item_id']) for _, i in data}

    # pickle.dump(tra, open('tra.pkl', 'wb'))
    # pickle.dump(tes, open('tes.pkl', 'wb'))
    pickle.dump(data, open('data.pkl', 'wb'))
else:
    print ('loading previous data.')
    #tra = pickle.load(open('tra.pkl', 'rb'))
    #tes = pickle.load(open('tes.pkl', 'rb'))
    data = pickle.load(open('data.pkl', 'rb'))
    item_user = pickle.load(open('item_user.pkl', 'rb'))

#删除item数量少于指定值的item，若session经过处理后长度小于2则抛弃
data_tmp = copy.deepcopy(data)
for i, j in data.items():
    tmp = list(filter(lambda x: item_user[x] >= item_least_num, j))
    if len(tmp) < item_least_num:
        del data_tmp[i]
    else:
        data_tmp[i] = tmp
data = copy.deepcopy(data_tmp)
del data_tmp

def TransformDataToSeq(df_dict, item_user):
    item_dict = {}
    ids = []
    seqs = []
    ctr = 1
    for sess_id, seq in df_dict.items():
        output = []
        for i in seq:
            if i in item_dict:
                output += [item_dict[i]]
            else:
                item_dict[i] = ctr
                output += [ctr]
                ctr += 1
            if len(output) < 2:
                continue
        ids += [sess_id]
        seqs += [output]
    print (ctr)
    with open("sr-gnn_tmp.txt", "w") as fout:
        fout.write(str(ctr) + "\n")
    return ids, seqs, item_dict
    
data_ids, data_seq, item_correspond_dict = TransformDataToSeq(data, item_user)
del data

def ProcessSeq(seqs):
    output_seqs = []
    label = []
    for seq in seqs:
        for i in range(1, len(seq)):
            tar = seq[-i]
            label += [tar]
            output_seqs += [np.array(seq[:-i], dtype=int)]
    return output_seqs, label

data_seq, data_label = ProcessSeq(data_seq)

data_seq = np.array(data_seq)
#pickle.dump((data_seq, data_label), open('train_gnn.pkl', 'wb'))
#pickle.dump(np.array(data_seq), open('train_seq.pkl', 'wb'))
#pickle.dump(data_label, open('train_label.pkl', 'wb'))
#pickle.dump(item_correspond_dict, open('item_correspond_dict.pkl', 'wb'))
joblib.dump(np.array(data_seq), open('train_seq', 'wb'), compress=3)
joblib.dump(np.array(data_label), open('train_label', 'wb'), compress=3)
joblib.dump(item_correspond_dict, open('item_corr_dict', 'wb'), compress=3)

time2 = time.time()
print ('data preprocessing complete, time cost:', time2-time1)

'''
###
print ('start model training.')
def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
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
    def __init__(self, data, sub_graph=False, method='ggnn', sparse=False, shuffle=False):
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
        self.method = method

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
            if self.method == 'ggnn':
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
            elif self.method == 'gat':
                A_in = []
                A_out = []
                for u_input in self.inputs[index]:
                    node = np.unique(u_input)
                    items.append(node.tolist() + (max_n_node - len(node)) * [0])
                    u_A = np.eye(max_n_node)
                    for i in np.arange(len(u_input) - 1):
                        if u_input[i + 1] == 0:
                            break
                        u = np.where(node == u_input[i])[0][0]
                        v = np.where(node == u_input[i + 1])[0][0]
                        u_A[u][v] = 1
                    A_in.append(-1e9 * (1 - u_A))
                    A_out.append(-1e9 * (1 - u_A.transpose()))
                    alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
                return A_in, A_out, alias_inputs, items, self.mask[index], self.targets[index]

        else:
            return self.inputs[index], self.mask[index], self.targets[index]
'''