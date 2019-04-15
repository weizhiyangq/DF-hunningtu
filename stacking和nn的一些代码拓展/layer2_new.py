# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 12:13:10 2019

@author: Administrator
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
from sklearn.svm import SVC


oof_lgb_train=pd.read_csv('oof_lgb_train.csv',sep='\t')
oof_lgb_test=pd.read_csv('oof_lgb_test.csv',sep='\t')

oof_svc_train=pd.read_csv('oof_svc_train.csv',sep='\t')
oof_svc_test=pd.read_csv('oof_svc_test.csv',sep='\t')

test_userid=oof_lgb_test['uid']

oof_xgb_train=pd.read_csv('oof_xgb_train.csv',sep='\t')
oof_xgb_test=pd.read_csv('oof_xgb_test.csv',sep='\t')

oof_nb_train=pd.read_csv('oof_nb_train.csv',sep='\t')
oof_nb_test=pd.read_csv('oof_nbc_test.csv',sep='\t')

oof_logistic_train=pd.read_csv('oof_logistic_train.csv',sep='\t')
oof_logistic_test=pd.read_csv('oof_logistic_test.csv',sep='\t')



print(oof_lgb_train)
print(oof_lgb_test)


folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4590)

oof_lgb=oof_lgb_train['lgb_fea'].tolist()
oof_xgb=oof_xgb_train['xgb_fea'].tolist()
oof_svc=oof_svc_train['svc_fea'].tolist()
oof_nb=oof_nb_train['nb_fea'].tolist()
oof_logistic=oof_logistic_train['logistic_fea'].tolist()

predictions_lgb=oof_lgb_test['lgb_fea'].tolist()
predictions_xgb=oof_xgb_test['xgb_fea'].tolist()
predictions_svc=oof_svc_test['svc_fea'].tolist()
predictions_nb=oof_nb_test['nb_fea'].tolist()
predictions_logistic=oof_logistic_test['logistic_fea'].tolist()



label=oof_lgb_train['label']


# 将lgb和xgb的结果进行stacking
train_stack = np.vstack([oof_lgb,oof_xgb,oof_svc]).transpose()
test_stack = np.vstack([predictions_lgb, predictions_xgb,predictions_svc]).transpose()

folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4590)
oof_stack = np.zeros(train_stack.shape[0])
#predictions = np.zeros([test_stack.shape[0],2])
predictions = np.zeros(test_stack.shape[0])

print(train_stack)

print('begin logistic:')
for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack,label)):
    print("fold {} begin:".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], label.iloc[trn_idx].values
    val_data, val_y = train_stack[val_idx], label.iloc[val_idx].values
    

    clf_3 = LogisticRegression(verbose=1)
#    clf_3 = SVC(probability=True)
    clf_3.fit(trn_data, trn_y)    
    oof_stack[val_idx] = clf_3.predict(val_data)
    
    predictions += clf_3.predict_proba(test_stack)[:,1] / 10

print(predictions[:200])


res = pd.DataFrame()
res['sample_file_name'] = list(test_userid.values)
res['label'] = predictions

#若要得到 01值，记得加这几句
#res = res.sort_values('label',ascending=False).reset_index(drop = True)
#res.loc[res.index<24770,'label'] = 1
#res.loc[res.index>=24770,'label'] = 0
#res['label'] = res['label'].astype(int)


#time_date = time.strftime('%Y-%m-%d',time.localtime(time.time()))

submit=pd.read_csv('submit_example.csv')
submit=submit[['sample_file_name']]
submit=submit.merge(res,on='sample_file_name',how='left')
print(submit)
submit.to_csv(r'fuxian_continue.csv',index=None)
