# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 19:45:24 2019

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


def save_variable(v,filename):
    f=open(filename,'wb')
    pickle.dump(v,f)
    f.close()
    return filename

def load_variavle(filename):
    f=open(filename,'rb')
    r=pickle.load(f)
    f.close()
    return r

def auc(y,pred):
    return roc_auc_score(y, pred)

def f1(y,pred):
    return f1_score(y, pred,average='macro')
def create_feature(df):
    create_fe = list()
    col = list()

    create_fe.append(len(df))
    create_fe.append(len(df.drop_duplicates()))
    col.append('data_len')
    col.append('data_drop_dup_len')
    for i in df.columns:
        if i!='设备类型':
            create_fe.append(len(df[i].unique()))
            create_fe.append(df[i].max())
            create_fe.append(df[i].min())
#             create_fe.append(df[i].max()-df[i].min())    
            
            
            create_fe.append(df[i].sum())
            create_fe.append(df[i].mean())
            
            create_fe.append(df[i].std())
            create_fe.append(df[i].var())
            
#             create_fe.append(df[i].std()/df[i].mean())  
#             create_fe.append(df[i].skew())
            
            col.append(i+'_unique_len')
            col.append(i+'_max')
            col.append(i+'_min')
#             col.append(i+'max_min_sub')
            
            col.append(i+'_sum')
            col.append(i+'_mean')
            
            col.append(i+'_std')
            col.append(i+'_var')
            
#             col.append(i+'std_mean_sub')
#             col.append(i+'_skew')
        else:
            col.append(i+'num')
            shebeiid=df[i].drop_duplicates()
            create_fe.append(len(shebeiid))
#            create_fe.append(df[i].max())
#            col.append(i+'_')
    return create_fe,col


train_uid = os.listdir('../data/data_train/')
test_uid = os.listdir('../data/data_test/')


def get_data():
#    shebei = {'ZV41153':0, 'ZV55eec':1, 'ZV75a42':2, 
#          'ZV7e8e3':3, 'ZV90b78':4, 'ZVc1d93':5, 'ZVe0672':6}
    
#    try:
#        train_all_fe =load_variavle('../data/train_fe_v3.pkl')
#        test_all_fe =load_variavle('../data/test_fe_v3.pkl')
#    except:
        # trin feature
    train_all_fe = list()
    for i in tqdm(train_uid):
        df = pd.read_csv('../data/data_train/'+i)
#            df['设备类型'] = df['设备类型'].map(shebei)
        df,col  = create_feature(df)
        train_all_fe.append(df)
    train_all_fe = pd.DataFrame(train_all_fe,columns=col)
    save_variable(train_all_fe,'../data/train_fe_v3.pkl')
    # test feature
    test_all_fe = list()
    for i in tqdm(test_uid):
        df = pd.read_csv('../data/data_test/'+i)
#            df['设备类型'] = df['设备类型'].map(shebei)
        df,col = create_feature(df)
        test_all_fe.append(df)
    test_all_fe = pd.DataFrame(test_all_fe,columns=col)
    save_variable(test_all_fe,'../data/test_fe_v3.pkl')
    return train_all_fe,test_all_fe


train_all_fe,test_all_fe = get_data()
train_all_fe['uid'] = train_uid
test_all_fe['uid'] = test_uid
train_all_fe.to_csv('../data/train_no_shebei_less.csv',index=None,sep='\t')
test_all_fe.to_csv('../data/test_no_shebei_less.csv',index=None,sep='\t')
print(train_all_fe.columns)


'''
f = open('train_fe_v3.pkl',encoding='gbk')
inf = pickle.load(f)
print (inf)
'''


'''
df_train=pd.read_pickle("train_fe_v3.pkl")
df_test=pd.read_pickle("test_fe_v3.pkl")
print(len(df_train))
print(len(df_train))
#print(df_train)
#print(df_test)
#clas=pd.value_counts(df_train['活塞工作时长_unique_len'])
clas=df_train['活塞工作时长_unique_len'].drop_duplicates().tolist()
print(clas)
'''