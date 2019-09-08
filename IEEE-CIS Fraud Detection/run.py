# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 23:27:51 2019

@author: STP
"""

import pandas as pd
import numpy as np
import gc

#通过数据类型转换减少内存占用，极为实用
#由于dataframe默认使用float64或int64存储数据，因此大量浪费了内存
#当col中存在np.nan或小数时，默认使用float存储，因为int无法保存np.nan
#判断col默认数据类型，将int64降至int8，float64降至float16
def reduce_mem_usage(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'] 
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    return df

train_df = pd.read_csv('train_transaction.csv', index_col='TransactionID')
test_df = pd.read_csv('test_transaction.csv', index_col='TransactionID')
train_identity = pd.read_csv('train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv('test_identity.csv', index_col='TransactionID')
train_df2 = train_df.merge(train_identity, how='left')
train_df = train_df.merge(train_identity, how='left', left_index=True, right_index=True)

#train_df = reduce_mem_usage(train_df)