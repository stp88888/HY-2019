# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 18:15:54 2019

@author: STP
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import pickle
#余弦相似度
def cossim(x, y):
    return len(set(x).intersection(set(y))) / (len(set(x)) * len(set(y)))**0.5

neighbour_num = 5
output_offline_test = 1

data_train = pd.read_csv('Antai_AE_round1_train_20190626.csv')
data_test = pd.read_csv('Antai_AE_round1_test_20190626.csv')
data_item = pd.read_csv('Antai_AE_round1_item_attr_20190626.csv')

#seperate xx and yy from train data
data_xx = data_train[data_train['buyer_country_id'] == 'xx']
data_yy = data_train[data_train['buyer_country_id'] == 'yy']
data_yy = data_yy.sample(frac=1)
data_yy_train = data_yy.iloc[:int(0.5*len(data_yy)), :]
data_yy_test = data_yy.iloc[int(0.5*len(data_yy)):, :]

#create offline train and test data
data_train_offline = pd.concat([data_xx, data_yy_train], axis=0)
data_test_offline = data_yy_test

if output_offline_test:
    data_test_offline.to_csv('data_test_offline.csv', index=None)

del data_train_offline['buyer_country_id'],data_test_offline['buyer_country_id']

'''
#translate time
data_train['create_order_time'] = pd.to_datetime(data_train['create_order_time'])
data_test['create_order_time'] = pd.to_datetime(data_test['create_order_time'])

#transform item price
data_item['item_price'] = np.log(data_item['item_price'])
'''

user_cart_train = defaultdict(lambda : [])
item_user_dict = defaultdict(lambda : [])
for _, each_train in data_train_offline.iterrows():
    user_cart_train[each_train['buyer_admin_id']].append(each_train['item_id'])
    item_user_dict[each_train['item_id']].append(each_train['buyer_admin_id'])

user_cart_test = defaultdict(lambda : [])
offline_ans = defaultdict(lambda : [])
for _, each_test in data_test_offline.iterrows():
    user_cart_test[each_test['buyer_admin_id']].append(each_test['item_id'])
    if offline_ans[each_test['buyer_admin_id']]:
        if offline_ans[each_test['buyer_admin_id']][1] > each_test['irank']:
            offline_ans[each_test['buyer_admin_id']] = [each_test['item_id'], each_test['irank']]
    else:
        offline_ans[each_test['buyer_admin_id']] = [each_test['item_id'], each_test['irank']]

#output ans
f = open('./offline_ans.pickle','w') 
pickle.dump(offline_ans, f, 0)
f.close()

#drop latest bought item
for i in offline_ans.keys():
    user_cart_test[i].remove(offline_ans[i][0])

same_high_score_userid = []
predict = []
for user_id, item_list in user_cart_test.items():
    neighbour_num_temp = neighbour_num
    user_neighbour_all = []
    for each_item in item_list:
        user_neighbour_all.extend(item_user_dict[each_item])
    user_neighbour_all = set(user_neighbour_all)
    
    #calculate similarity, formula: 2 * (two user's intersection) / (one user's set + other user's set)
    #could improve by: (two user's intersection) / (two user's aggregate)
    #could improve by: (some user's intersection may be bought more than once)
    neighbour_score = []
    for each_neightbour in user_neighbour_all:
        score = 2 * len(set(user_cart_train[each_neightbour]).intersection(set(item_list))) / (len(set(user_cart_train[each_neightbour])) 
                + len(set(item_list)))
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
        neighbour = neighbour_score[:neighbour_num_temp]
    
    #get N nearest neightbour's items
    neighbour_item = []
    for each_neightbour in neighbour:
        neighbour_item.extend(user_cart_train[each_neightbour[0]])
    
    #get most possible 30 items
    ans = [user_id]
    ans_temp = Counter(neighbour_item).most_common(30)
    for i in ans_temp:
        ans.append(i[0])
    predict.append(ans)
    
predict = pd.DataFrame(predict)
predict.columns = ['k'+str(i) for i in range(predict.shape[1])]
predict.rename(columns={'k0' : 'user_id'}, inplace=True)
predict.to_csv('username_offline.csv', index=None)    

'''
f = open('username_offline.csv', 'w+')
for i in predict:
    f.write(','.join(map(str, i))+'\n')
f.close()
'''