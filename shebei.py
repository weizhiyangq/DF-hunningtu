# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 21:52:09 2019

@author: YWZQ
"""

import pandas as pd 
import numpy as np
from tqdm import *
import math 
import matplotlib.pyplot as plt
import pickle
import operator

import os
import sys
from datetime import *
import lightgbm as lgb
import xgboost as xgb

import time
from sklearn.metrics import roc_auc_score,f1_score
from sklearn.model_selection import StratifiedKFold,KFold

'''   
train_uid = os.listdir('../data/data_train/')
test_uid = os.listdir('../data/data_test/')
shebeid=[]
for i in tqdm(train_uid):
    df = pd.read_csv('../data/data_train/'+i)
    shebei=df['设备类型'].drop_duplicates().values.tolist()
    shebeid.extend(shebei)
for i in tqdm(test_uid):
    df = pd.read_csv('../data/data_test/'+i)
    shebei=df['设备类型'].drop_duplicates().values.tolist()
    shebeid.extend(shebei)
shebei_set=set(shebeid)
print(shebei_set)
#{'ZVe44', 'ZVa78', 'ZV573', 'ZVa9c', 'ZV252', 'ZVfd4', 'ZV63d'}
'''


train_uid = os.listdir('../data/data_train/')
test_uid = os.listdir('../data/data_test/')
all_id=train_uid+test_uid
df_shebei=pd.DataFrame()
df_shebei['uid']=all_id
shebei_list=['ZVe44', 'ZVa78', 'ZV573', 'ZVa9c', 'ZV252', 'ZVfd4', 'ZV63d']
for i in shebei_list:
    df_shebei[i]=0
    
for i in tqdm(train_uid):
    df = pd.read_csv('../data/data_train/'+i)
    shebei=df['设备类型'].drop_duplicates().values.tolist()
    df_shebei.loc[df_shebei['uid']==i,shebei]=1

for i in tqdm(test_uid):
    df = pd.read_csv('../data/data_test/'+i)
    shebei=df['设备类型'].drop_duplicates().values.tolist()
    df_shebei.loc[df_shebei['uid']==i,shebei]=1
    
df_shebei.to_csv('../data/shebei_id.csv',index=None,sep='\t')
 

print(df_shebei)

print(all_id)







