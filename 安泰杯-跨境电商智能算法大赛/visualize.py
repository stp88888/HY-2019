# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 12:48:17 2019

@author: STP
"""


import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import pickle
import time
#余弦相似度
def cossim(x, y):
    return len(set(x).intersection(set(y))) / (len(set(x)) * len(set(y)))**0.5

def dd():
    return defaultdict(int)

time1 = time.time()

neighbour_num = 5
output_offline_test = 0
offline = 0
#random_state_num = 1

data_train = pd.read_csv('Antai_AE_round1_train_20190626.csv')
data_test = pd.read_csv('Antai_AE_round1_test_20190626.csv')
data_item = pd.read_csv('Antai_AE_round1_item_attr_20190626.csv')

data_all = pd.concat([data_train, data_test], axis=0)

#画出每天购买的图(按照irank划分),结果显示呈现反比例
day_buy_num = data_all[['irank', 'item_id']].groupby(['irank']).count()
day_buy_num_plot = day_buy_num.iloc[:100, :].plot()

#画出所有用户购买次数的图,结果显示反比例(TODO:剔除过10000的用户,不正常用户)
user_buy_num = data_all[['buyer_admin_id','item_id']].groupby(['buyer_admin_id']).count()
user_buy_num['buy_item_num'] = 1
user_buy_num = user_buy_num.groupby(['item_id']).sum()
user_buy_num.iloc[:100, :].plot()

#画出所有商品被购买次数的图,结果显示反比例
item_buy_num = data_all[['buyer_admin_id','item_id']].groupby(['item_id']).count()
item_buy_num['buy_user_num'] = 1
item_buy_num = item_buy_num.groupby(['buyer_admin_id']).sum()
item_buy_num.iloc[:100, :].plot()

#读取数据
user_cart_train = pickle.load(open('user_cart_train.pkl', 'rb'))
item_user_dict = pickle.load(open('item_user_dict.pkl', 'rb'))
item_cate_dict = pickle.load(open('item_cate_dict.pkl', 'rb'))
item_store_dict = pickle.load(open('item_store_dict.pkl', 'rb'))

#大概查看有重复购买记录的用户数量和比例(不准确)
k = 0
for i in user_cart_train.keys():
    if max(user_cart_train[i].values()) >= 1:
        k += 1
print ('total user numbers:', len(user_cart_train.keys()))
print ('有重复购买记录的用户数量和比例', k, k / len(user_cart_train.keys()))



