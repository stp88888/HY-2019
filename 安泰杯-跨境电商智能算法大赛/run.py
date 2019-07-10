# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 21:00:59 2019

@author: STP
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter

neighbour_num = 5

data_train = pd.read_csv('Antai_AE_round1_train_20190626.csv')
data_test = pd.read_csv('Antai_AE_round1_test_20190626.csv')
data_item = pd.read_csv('Antai_AE_round1_item_attr_20190626.csv')

del data_train['buyer_country_id'],data_test['buyer_country_id']

'''
#translate time
data_train['create_order_time'] = pd.to_datetime(data_train['create_order_time'])
data_test['create_order_time'] = pd.to_datetime(data_test['create_order_time'])

#transform item price
data_item['item_price'] = np.log(data_item['item_price'])
'''

user_cart_train = defaultdict(lambda : [])
item_user_dict = defaultdict(lambda : [])
for _, each_train in data_train.iterrows():
    user_cart_train[each_train['buyer_admin_id']].append(each_train['item_id'])
    item_user_dict[each_train['item_id']].append(each_train['buyer_admin_id'])

user_cart_test = defaultdict(lambda : [])
for _, each_test in data_test.iterrows():
    user_cart_test[each_test['buyer_admin_id']].append(each_test['item_id'])

same_high_score_userid = []
predict = []
for user_id, item_list in user_cart_test.items():
    neighbour_num_temp = neighbour_num
    user_neighbour_all = []
    for each_item in item_list:
        user_neighbour_all.extend(item_user_dict[each_item])
    user_neighbour_all = set(user_neighbour_all)
    
    #calculate similarity
    neighbour_score = []
    for each_neightbour in user_neighbour_all:
        score = len(set(user_cart_train[each_neightbour]).intersection(set(item_list))) / (len(set(user_cart_train[each_neightbour])) 
                + len(set(item_list)))
        neighbour_score.append([each_neightbour, score])
    neighbour_score.sort(key=lambda x:x[1], reverse=True)
    
    #if having two or more same high score
    while 1:
        if len(neighbour_score) > neighbour_num_temp and neighbour_score[neighbour_num_temp-1][1] == neighbour_score[neighbour_num_temp][1]:
            same_high_score_userid.append(user_id)
            neighbour_num_temp += 1
        else:
            break
    
    #get N nearest neightbour
    if len(neighbour_score) > neighbour_num_temp:
        neighbour = neighbour_score[:neighbour_num_temp]
    
    #get N nearest neightbour's items
    neighbour_item = []
    for each_neightbour in neighbour:
        neighbour_item.extend(user_cart_train[each_neightbour[0]])
    
    ans = [user_id]
    ans_temp = Counter(neighbour_item).most_common(30)
    for i in ans_temp:
        ans.append(i[0])
    predict.append(ans)
    
f = open('username.csv', 'w+')
for i in predict:
    f.write(','.join(map(str, i))+'\n')
f.close()
