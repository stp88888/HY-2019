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
import xgboost as xgb
#相似度
def cossim(x, y):
    return len(set(x).intersection(set(y))) / (len(set(x)) * len(set(y)))**0.5

def calc_score_sim(user_id, user_cart_train, calculated_neighbour_sim):
    score_temp = 0
    for i in user_cart_train[user_id].keys():
        score_temp += user_cart_train[user_id][i]**2
    calculated_neighbour_sim[user_id] = score_temp**0.5
    return calculated_neighbour_sim

def dd():
    return defaultdict(int)

time1 = time.time()

neighbour_num = 5
output_offline_test = 0
offline = 0
#random_state_num = 1
drop_user = 1
drop_user_limit = 50
#if xgb_switch on, drop_user must be 1
xgb_switch = 0
#if first_run off, drop_user is useless
first_run = 0

data_train = pd.read_csv('Antai_AE_round1_train_20190626.csv')
data_test = pd.read_csv('Antai_AE_round1_test_20190626.csv')
data_item = pd.read_csv('Antai_AE_round1_item_attr_20190626.csv')

data_all = pd.concat([data_train, data_test], axis=0)

user_drop_list = []
if drop_user == 1:
    #剔除购买商品数量超过一定数量的用户
    user_buy_num = data_all[['buyer_admin_id','item_id']].groupby(['buyer_admin_id']).count()
    user_buy_num = user_buy_num.reset_index(drop=False)
    user_drop_list = user_buy_num[user_buy_num['item_id'] > drop_user_limit]['buyer_admin_id'].values
    user_drop_list_set = set(user_drop_list)

time2 = time.time()
print ('step1 complete', time2 - time1)

'''
#translate time
data_train['create_order_time'] = pd.to_datetime(data_train['create_order_time'])
data_test['create_order_time'] = pd.to_datetime(data_test['create_order_time'])

#transform item price
data_item['item_price'] = np.log(data_item['item_price'])
'''

if first_run == 1:
    user_cart_train = defaultdict(lambda: defaultdict(lambda : 0))
    item_user_dict = defaultdict(lambda : [])
    for _, each_train in data_all.iterrows():
        if each_train['buyer_admin_id'] not in user_drop_list_set:
            user_cart_train[each_train['buyer_admin_id']][each_train['item_id']] += math.exp(0.1 * (1 - each_train['irank']))
            item_user_dict[each_train['item_id']].append(each_train['buyer_admin_id'])
else:
    user_cart_train = pickle.load(open('user_cart_train.pkl', 'rb'))
    item_user_dict = pickle.load(open('item_user_dict.pkl', 'rb'))
    item_cate_dict = pickle.load(open('item_cate_dict.pkl', 'rb'))
    item_store_dict = pickle.load(open('item_store_dict.pkl', 'rb'))

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

time4 = time.time()
print ('step3 complete', time4 - time3)

'''
#xgb
#构建负样本
time4 = time.time()
item_all_list_set = set(data_item['item_id'].values)
neg = []
union_zero = 0
each_data_drop_index = []
for each_data in data_all.iterrows():
    each_data2 = each_data[1]
    if each_data2['buyer_admin_id'] in user_drop_set:
        each_data_drop_index.append(each_data[0])
        continue

    if each_data[0] % 100000 == 0:
        print (each_data[0], time.time() - time4)

    if each_data2['item_id'] not in item_all_list_set:
        continue

    each_item_data = data_item[data_item['item_id'] == each_data2['item_id']]
    each_item_cate = each_item_data['cate_id'].values[0]
    each_item_store = each_item_data['store_id'].values[0]
    union = set(item_cate_dict[each_item_cate]).intersection(set(item_store_dict[each_item_store]))
    #优先从cate和store的交集中取一个作为负样本，其次从store中取，再其次从cate取，最后从所有item中取
    #取的负样本不能在用户的已购买item里
    for i in user_cart_train[each_data2['buyer_admin_id']].keys():
        if i in union:
            union.remove(i)
    
    cate_set = set(item_cate_dict[each_item_cate])
    for i in user_cart_train[each_data2['buyer_admin_id']].keys():
        if i in cate_list:
            cate_list.remove(i)
    
    store_set = set(item_store_dict[each_item_store])
    for i in user_cart_train[each_data2['buyer_admin_id']].keys():
        if i in store_list:
            store_list.remove(i)

    if union:
        neg_sample = np.random.choice(union)
        neg.append([each_data2['buyer_admin_id'], neg_sample])
    elif store_set:
        union_zero += 1
        neg_sample = np.random.choice(store_set)
        neg.append([each_data2['buyer_admin_id'], neg_sample])
    elif cate_set:
        union_zero += 1
        neg_sample = np.random.choice(cate_set)
        neg.append([each_data2['buyer_admin_id'], neg_sample])
    else:
        neg_sample = np.random.choice(item_user_dict.keys())
        while 1:
            if neg_sample in user_cart_train[each_data2['buyer_admin_id']].keys():
                neg_sample = np.random.choice(item_user_dict.keys())
            else:
                break
        neg.append([each_data2['buyer_admin_id'], neg_sample])

neg = pd.DataFrame(neg)
neg.columns = ['buyer_admin_id', 'item_id']
neg.to_csv('neg.csv', index=None, header=None)

time5 = time.time()
print ('step4 complete', time5 - time4)
'''
'''
#get hot items
item_sales_number = []
for i in item_user_dict.keys():
    item_sales_number.append([i, len(item_user_dict[i])])
item_sales_number = pd.DataFrame(item_sales_number)
item_sales_number.columns = ['item_id', 'sales_num']
item_sales_number = item_sales_number.sort_values('sales_num', ascending=False).iloc[:30, :]
item_hot_list = item_sales_number['item_id'].values
'''


#重复填充用户已购买过的item
predict = pd.DataFrame(predict)
predict_null = predict.T.isna().any()
predict = predict.fillna(-1)
for i, j in enumerate(predict_null):
    if j:
        l = 0
        item_hot_list = [i[0] for i in sorted(user_cart_train[predict.iloc[i, 0]].items(), key=lambda x:x[1], reverse = True)] * 30
        for k in range(30):
            if predict.iloc[i, k + 1] == -1:
                predict.iloc[i, k + 1] = item_hot_list[l]
                l += 1
predict = predict.astype(int)
predict.columns = ['k'+str(i) for i in range(predict.shape[1])]
predict.rename(columns={'k0' : 'user_id'}, inplace=True)
predict.to_csv('username.csv', index=None, header=None)



'''
f = open('username_offline.csv', 'w+')
for i in predict:
    f.write(','.join(map(str, i))+'\n')
f.close()
'''