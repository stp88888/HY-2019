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
    tes = (data_test[['buyer_admin_id', 'item_id', 'irank']]
        .sort_values('irank')[['buyer_admin_id', 'item_id']]
        .groupby('buyer_admin_id'))
    tes = {i['buyer_admin_id'].iloc[0]: list(i['item_id']) for _, i in tes}
    data = (data_all[['buyer_admin_id', 'item_id', 'irank']]
        .sort_values('irank')[['buyer_admin_id', 'item_id']]
        .groupby('buyer_admin_id'))
    data = {i['buyer_admin_id'].iloc[0]: list(i['item_id']) for _, i in data}

    # pickle.dump(tra, open('tra.pkl', 'wb'))
    pickle.dump(tes, open('tes.pkl', 'wb'))
    pickle.dump(data, open('data.pkl', 'wb'))

    del data_all, data_train, data_test
else:
    print ('loading previous data.')
    #tra = pickle.load(open('tra.pkl', 'rb'))
    tes = pickle.load(open('tes.pkl', 'rb'))
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
print ('train data preprocessing complete, time cost:', time2-time1)


def TransformDataToSeq_Predict(df_dict, item_user, item_dict):
    ids = []
    seqs = []
    for sess_id, seq in df_dict.items():
        output = []
        for i in seq:
            if i in item_dict:
                output += [item_dict[i]]
        if len(output) < 2:
            continue
        ids += [sess_id]
        seqs += [output]
    return ids, seqs

test_ids, test_seq = TransformDataToSeq_Predict(tes, item_user, item_correspond_dict)
joblib.dump(np.array(test_seq), open('test_seq', 'wb'), compress=3)

time3 = time.time()
print ('test data preprocessing complete, time cost:', time3-time2)