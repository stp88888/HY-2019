# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:38:31 2019

@author: sutianpei
"""


import pandas as pd
import numpy as np
import pickle
from collections import defaultdict

def dd():
    return defaultdict(int)

sub = pd.read_csv('sub.csv', header=None)
predict = pd.read_csv('username.csv', header=None)

user_cart_train = pickle.load(open('user_cart_train.pkl', 'rb'))
item_user_dict = pickle.load(open('item_user_dict.pkl', 'rb'))
item_cate_dict = pickle.load(open('item_cate_dict.pkl', 'rb'))
item_store_dict = pickle.load(open('item_store_dict.pkl', 'rb'))
item_info = pickle.load(open('item_info.pkl', 'rb'))

print ('start')
final = pd.DataFrame(np.zeros((len(sub), 31), dtype = int))
final.iloc[:, 0] = sub.iloc[:, 0]
for i in range(len(sub)):
    each_sub = list(sub.iloc[i].values)[1:][::-1]
    each_pred = list(predict.iloc[i].values)[1:][::-1]

    tmp = []
    while each_sub:
        j = each_sub.pop()
        if j != -1:
            tmp.append(j)
    #each_sub = tmp[::-1]
    
    each_temp = tmp
    while len(each_temp) < 30:
        if not each_pred:
            each_temp.append(5595070)
            continue
        j = each_pred.pop()
        if j not in each_temp:
            each_temp.append(j)
    final.iloc[i, 1:] = each_temp


    #前5个位用户历史购买的最后五个
    # each_temp = tmp[:5]

    # each_sub = each_sub[5:]
    # while each_sub:
    #     tmp.append(each_sub.pop())
    #     tmp.append(each_pred.pop())
    
    # while len(tmp) < 30:
    #     tmp.append(each_pred.pop())
    # final.iloc[i, 1:] = tmp[:30]
final.to_csv('final.csv', index=None, header=None)
        

