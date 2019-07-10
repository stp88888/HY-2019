# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 15:20:52 2019

@author: 03912303
"""

import pandas as pd
import numpy as np
import pickle

f = open('./offline_ans.pickle','r')
ans = pickle.load(f)
f.close()

data = pd.read_csv('username_offline.csv')

score = 0
for each_user in ans.keys():
    user_list = data[data['user_id'] == each_user].iloc[:, 1:].to_list()
    if ans[each_user] in user_list:
        score += 1 / (1 + user_list.index(ans[each_user]))
        
print ('final score:', score)
