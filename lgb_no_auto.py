

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:18:03 2019

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


train=pd.read_csv('x_all_label.csv',sep='\t',encoding='gbk')
test=pd.read_csv('x_test_all.csv',sep='\t',encoding='gbk')

#feature_no_import=['ZVe44', 'ZVa9c', '活塞工作时长_std', 'ZV63d', 'data_drop_dup_len', 'ZVfd4', 'ZV252', '低压开关_max', '高压开关_max', '活塞工作时长_unique_len', '正泵_max', '反泵_unique_len', '低压开关_unique_len', '低压开关_min', '设备类型num', '反泵_std', '反泵_mean', '低压开关_mean', '反泵_min', '反泵_max', '正泵_mean', '正泵_std', '高压开关_min', '低压开关_std', '正泵_min', '正泵_unique_len', '搅拌超压信号_std', '搅拌超压信号_mean', '搅拌超压信号_sum', '高压开关_unique_len', '搅拌超压信号_max', '搅拌超压信号_unique_len', '高压开关_std', '高压开关_mean', '搅拌超压信号_min']
#train=train.drop(feature_no_import,axis=1)
#test=test.drop(feature_no_import,axis=1)

#都是只有一个值，且label为0和1的值一样，所以删掉这些无用特征，lgb中也显示特征重要性为0
#train=train.drop(['搅拌超压信号_max','搅拌超压信号_min','搅拌超压信号_sum','搅拌超压信号_mean','搅拌超压信号_unique_len'],axis=1)
#test=test.drop(['搅拌超压信号_max','搅拌超压信号_min','搅拌超压信号_sum','搅拌超压信号_mean','搅拌超压信号_unique_len'],axis=1)

#train=train.drop('设备类型num',axis=1)
#test=test.drop('设备类型num',axis=1)

#删掉活塞这两个，线上线下都下降一点点
#train=train.drop(['活塞工作时长_min','活塞工作时长_mean'],axis=1)
#test=test.drop(['活塞工作时长_min','活塞工作时长_mean'],axis=1)


#train['has_ZV63d']=(train['ZV63d']==1).astype('int')
#test['has_ZV63d']=(test['ZV63d']==1).astype('int')

#增加这个会下降
#train['活塞工作时长_min_less_2.65']=(train['活塞工作时长_min']<2.65).astype('int')
#train['活塞工作时长_mean_less_2.66']=(train['活塞工作时长_mean']<2.66).astype('int')
#test['活塞工作时长_min_less_2.65']=(test['活塞工作时长_min']<2.65).astype('int')
#test['活塞工作时长_mean_less_2.66']=(test['活塞工作时长_mean']<2.66).astype('int')


y_train=train['label']
X_train=train.drop(['uid','label'],axis=1)


test_userid = test.pop('uid')

print(X_train.columns.tolist())
test_columns=test.columns.tolist()
print(test_columns)

#train_len=len(X_train)
#test_len=len(test)
print(X_train.shape)
print(test.shape)

#train_test=pd.concat([X_train,test])
#print(train_test.shape)
#for i in train_test.columns.tolist():
#    train_test[i]=train_test[i]/(train_test[i].max()+0.0001)

#X_train=train_test.iloc[:train_len,:]
#print(X_train.shape)
#test=train_test.iloc[train_len:,:]
#    

#test=(test - test.mean()) / (test.std())
#X_train=(X_train - X_train.mean()) / (X_train.std())
#df_norm2 = df.apply(lambda x: (x - np.mean(x)) / (np.std(x))) 也可以标准化

test=test.values
X=X_train.values
y=y_train.values


K = 5
skf = StratifiedKFold(n_splits = K, shuffle = True ,random_state=16)


import lightgbm as lgb
from sklearn.metrics import roc_auc_score

xx_cv = []
xx_pre = []
xx_beat = {}

import operator

lgb_pred_te=np.array([0]*len(test))
for k,(train_in,test_in) in enumerate(skf.split(X,y)):
    X_train,X_test,y_train,y_test = X[train_in],X[test_in],y[train_in],y[test_in]

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # specify your configurations as a dict
    params = {
        
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': 31,
        'learning_rate': 0.04,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
        
    }

    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=40000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=50,
                    verbose_eval=500)

    print('Start predicting...')
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    test_predict=gbm.predict(test, num_iteration=gbm.best_iteration)
    xx_pre.append(list(test_predict))

#    print(test_predict)
#    print(xx_pre)

pre=[(xx_pre[0][i]+xx_pre[1][i]+xx_pre[2][i]+xx_pre[3][i]+xx_pre[4][i])/5 for i in range(len(xx_pre[0]))]
        



res = pd.DataFrame()
res['sample_file_name'] = list(test_userid.values)
res['label'] = pre
res = res.sort_values('label',ascending=False).reset_index(drop = True)

#预测 01记得加回来
#res.loc[res.index<24710,'label'] = 1
#res.loc[res.index>=24710,'label'] = 0
#res['label'] = res['label'].astype(int)


#time_date = time.strftime('%Y-%m-%d',time.localtime(time.time()))

submit=pd.read_csv('submit_example.csv')
submit=submit[['sample_file_name']]
submit=submit.merge(res,on='sample_file_name',how='left')

#submit.to_csv(r'res_learate_25_24710_fuxian_01.csv',index=None)
submit.to_csv(r'res_learate_4_24710_fuxian_continue.csv',index=None)
#print(submit)
lgb_feature_importance=pd.DataFrame({'column':test_columns,'importance':gbm.feature_importance()})
lgb_feature_importance=lgb_feature_importance.sort_values('importance',ascending=False).reset_index(drop=True)
lgb_feature_importance.to_csv('lgb_feature_importance.csv',sep='\t')
print(lgb_feature_importance)
print('info')
print('线下成绩约',np.mean(xx_cv))
print('预测的正样本大约',submit[submit['label']==1].shape[0] / submit.shape[0])


