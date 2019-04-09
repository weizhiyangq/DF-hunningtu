

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
from sklearn.model_selection import StratifiedKFold,KFold,RepeatedKFold
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LogisticRegression


train=pd.read_csv('x_all_label.csv',sep='\t',encoding='gbk')
test=pd.read_csv('x_test_all.csv',sep='\t',encoding='gbk')



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
oof_lgb = np.zeros(len(train))
predictions_lgb = np.zeros(len(test))

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
    
    oof_lgb[test_in] = gbm.predict(X[test_in], num_iteration=gbm.best_iteration)
    predictions_lgb += gbm.predict(test, num_iteration=gbm.best_iteration) / skf.n_splits


print('lgb predict over')

        

oof_xgb = np.zeros(len(train))
predictions_xgb = np.zeros(len(test))
for k,(train_in,test_in) in enumerate(skf.split(X,y)):
    X_train,X_test,y_train,y_test = X[train_in],X[test_in],y[train_in],y[test_in]
   
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_eval = xgb.DMatrix(X_test, label=y_test)
    

 
    xgb_params = {'eta': 0.008, 
              'max_depth': 5,
              'subsample': 0.8, 
              'colsample_bytree': 0.8, 
              'objective':'binary:logistic',
              'eval_metric':'auc',
              'silent': True, 
              'nthread': 4,
              }



    # train
    watchlist = [(xgb_train,'train'),(xgb_eval,'val')] 
    xgb_model = xgb.train(xgb_params,
                    xgb_train,
                    evals=watchlist,
                    num_boost_round=40000,
#                    valid_sets=xgb_eval,
                    early_stopping_rounds=50,
                    verbose_eval=100)


    
    
    print('Start predicting...')
#    y_pred = xgb_model.predict(xgb_eval, ntree_limit=xgb_model.best_ntree_limit)
    oof_xgb[test_in] = xgb_model.predict(xgb.DMatrix(X[test_in]), ntree_limit=xgb_model.best_ntree_limit)
    predictions_xgb += xgb_model.predict(xgb.DMatrix(test), ntree_limit=xgb_model.best_ntree_limit) / skf.n_splits
print('xgb predict over')

# 将lgb和xgb的结果进行stacking
train_stack = np.vstack([oof_lgb,oof_xgb]).transpose()
test_stack = np.vstack([predictions_lgb, predictions_xgb]).transpose()

folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4590)
oof_stack = np.zeros(train_stack.shape[0])
predictions = np.zeros(test_stack.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack,y_train)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], y_train.iloc[trn_idx].values
    val_data, val_y = train_stack[val_idx], y_train.iloc[val_idx].values
    

    clf_3 = LogisticRegression()
    clf_3.fit(trn_data, trn_y)
    
    oof_stack[val_idx] = clf_3.predict(val_data)
    predictions += clf_3.predict(test_stack) / 10




res = pd.DataFrame()
res['sample_file_name'] = list(test_userid.values)
res['label'] = predictions

submit=pd.read_csv('submit_example.csv')
submit=submit[['sample_file_name']]
submit=submit.merge(res,on='sample_file_name',how='left')

submit.to_csv('logistic_lianxu.csv',sep='\t')


submit = submit.sort_values('label',ascending=False).reset_index(drop = True)


res.loc[res.index<24710,'label'] = 1
res.loc[res.index>=24710,'label'] = 0
res['label'] = res['label'].astype(int)


submit.to_csv(r'logistic_0_1.csv',index=None)
#print(submit)
lgb_feature_importance=pd.DataFrame({'column':test_columns,'importance':gbm.feature_importance()})
lgb_feature_importance=lgb_feature_importance.sort_values('importance',ascending=False).reset_index(drop=True)
lgb_feature_importance.to_csv('lgb_feature_importance.csv',sep='\t')
#print(lgb_feature_importance)
print('info')
print('线下成绩约',np.mean(xx_cv))
print('预测的正样本大约',submit[submit['label']==1].shape[0] / submit.shape[0])


