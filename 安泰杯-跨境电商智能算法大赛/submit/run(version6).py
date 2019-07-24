# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 21:00:59 2019

@author: STP
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import pickle
import time
import math
#jarcard相似度
def cossim(x, y):
    return len(set(x).intersection(set(y))) / (len(set(x)) * len(set(y)))**0.5

def calc_score_sim(user_id, user_cart_train, calculated_neighbour_sim):
    score_temp = 0
    for i in user_cart_train[user_id].keys():
        score_temp += user_cart_train[user_id][i]**2
    calculated_neighbour_sim[user_id] = score_temp**0.5
    return calculated_neighbour_sim

time1 = time.time()

neighbour_num = 50
output_offline_test = 0
offline = 0
#random_state_num = 1

data_train = pd.read_csv('Antai_AE_round1_train_20190626.csv')
data_test = pd.read_csv('Antai_AE_round1_test_20190626.csv')
data_item = pd.read_csv('Antai_AE_round1_item_attr_20190626.csv')

data_all = pd.concat([data_train, data_test], axis=0)

#剔除购买商品数量超过一定数量的用户
#user_buy_num = data_all[['buyer_admin_id','item_id']].groupby(['buyer_admin_id']).count()
#user_buy_num = user_buy_num.reset_index(drop=False)
#user_drop_list = user_buy_num[user_buy_num['item_id'] > 50]['buyer_admin_id'].values

time2 = time.time()
print ('step1 complete', time2 - time1)

'''
#translate time
data_train['create_order_time'] = pd.to_datetime(data_train['create_order_time'])
data_test['create_order_time'] = pd.to_datetime(data_test['create_order_time'])

#transform item price
data_item['item_price'] = np.log(data_item['item_price'])
'''

user_cart_train = defaultdict(lambda: defaultdict(lambda : 0))
item_user_dict = defaultdict(lambda : [])
for _, each_train in data_all.iterrows():
    if each_train['buyer_admin_id'] not in user_drop_list:
        user_cart_train[each_train['buyer_admin_id']][each_train['item_id']] += math.exp(0.1 * (1 - each_train['irank']))
        item_user_dict[each_train['item_id']].append(each_train['buyer_admin_id'])

time3 = time.time()
print ('step2 complete', time3 - time2)

'''
#output ans
pickle.dump(offline_ans, open('offline_ans.pickle', 'wb'))
'''

same_high_score_userid = []
predict = []
calculated_neighbour_sim = defaultdict(lambda: 0)
for user_id in data_test.buyer_admin_id.drop_duplicates():
    item_list = user_cart_train[user_id].keys()
    neighbour_num_temp = neighbour_num
    user_neighbour_all = []
    for each_item in item_list:
        user_neighbour_all.extend(item_user_dict[each_item])
    user_neighbour_all = set(user_neighbour_all)
    
    #计算用户的模
    if user_id not in calculated_neighbour_sim.keys():
        score_temp = 0
        for i in item_list:
            score_temp += user_cart_train[user_id][i]**2
        calculated_neighbour_sim[user_id] = score_temp**0.5
        #calculated_neighbour_sim = calc_score_sim(user_id, user_cart_train, calculated_neighbour_sim)

    #calculate similarity, formula: 2 * (two user's intersection) / (one user's set + other user's set)
    #could improve by: (two user's intersection) / (two user's aggregate)
    #could improve by: (some user's intersection may be bought more than once)
    neighbour_score = []
    for each_neightbour in user_neighbour_all:
        #计算邻居的模
        if each_neightbour not in calculated_neighbour_sim.keys():
            score_temp = 0
            for i in user_cart_train[each_neightbour].keys():
                score_temp += user_cart_train[each_neightbour][i]**2
            calculated_neighbour_sim[each_neightbour] = score_temp**0.5
            #calculated_neighbour_sim = calc_score_sim(each_neightbour, user_cart_train, calculated_neighbour_sim)

        #get intersection of user and neighbour
        item_intersection = set(item_list).intersection(user_cart_train[each_neightbour].keys())

        #calc score
        score = 0
        for i in item_intersection:
            score += user_cart_train[user_id][i] * user_cart_train[each_neightbour][i] / math.log(1 + len(item_user_dict[i]))
        score /= (calculated_neighbour_sim[user_id] * calculated_neighbour_sim[each_neightbour])
        #score = 2 * len(set(user_cart_train[each_neightbour]).intersection(set(item_list))) / (len(set(user_cart_train[each_neightbour]))
        #        + len(set(item_list)))
        #use cossim to calc similarity
        #score = cossim(user_cart_train[each_neightbour], item_list)
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
        #neighbour_score[0] is user_id, drop it
        neighbour = neighbour_score[:neighbour_num_temp]
    
    #get N nearest neightbour's items
    neighbour_item = defaultdict(lambda: 0)
    for each_neighbour in neighbour:
        for neightbour_item in user_cart_train[each_neighbour[0]].keys():
            neighbour_item[neightbour_item] += user_cart_train[each_neighbour[0]][neightbour_item]
    
    #get most possible 30 items
    neighbour_item_sort = sorted(neighbour_item.items(), key=lambda x: x[1], reverse=True)
    ans = [user_id]
    for i in neighbour_item_sort[:30]:
        ans.append(i[0])
    # ans_temp = Counter(neighbour_item).most_common(30)
    # for i in ans_temp:
    #     ans.append(i[0])
    predict.append(ans)


#get hot items
item_sales_number = []
for i in item_user_dict.keys():
    item_sales_number.append([i, len(item_user_dict[i])])
item_sales_number = pd.DataFrame(item_sales_number)
item_sales_number.columns = ['item_id', 'sales_num']
item_sales_number = item_sales_number.sort_values('sales_num', ascending=False).iloc[:30, :]
item_hot_list = item_sales_number['item_id'].values


predict = pd.DataFrame(predict)
predict_null = predict.T.isna().any()
predict = predict.fillna(-1)
for i, j in enumerate(predict_null):
    if j:
        l = 0
        for k in range(30):
            if predict.iloc[i, k + 1] == -1:
                predict.iloc[i, k + 1] = item_hot_list[l]
                l += 1
predict = predict.astype(int)
predict.columns = ['k'+str(i) for i in range(predict.shape[1])]
predict.rename(columns={'k0' : 'user_id'}, inplace=True)
predict.to_csv('username.csv', index=None, header=None)

time4 = time.time()
print ('step3 complete', time4 - time3)

'''
f = open('username_offline.csv', 'w+')
for i in predict:
    f.write(','.join(map(str, i))+'\n')
f.close()
'''