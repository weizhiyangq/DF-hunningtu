# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 15:37:10 2019

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
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



train=pd.read_csv('x_all_label.csv',sep='\t',encoding='gbk')
test=pd.read_csv('x_test_all.csv',sep='\t',encoding='gbk')




label=train['label']

train_uid=train['uid']

X_train=train.drop(['uid','label'],axis=1)


test_userid = test.pop('uid')

#print(X_train.columns.tolist())
test_columns=test.columns.tolist()
#print(test_columns)

#train_len=len(X_train)
#test_len=len(test)
print(X_train.shape)
print(test.shape)


#print(X_train.describe())

X_train.fillna(0,inplace=True)
print(test.isnull().any().astype('int').sum())

test=test.values
X=X_train.values
y=label.values


K = 5
skf = StratifiedKFold(n_splits = K, shuffle = True ,random_state=16)



from sklearn.metrics import roc_auc_score

xx_cv = []
xx_pre = []
xx_beat = {}


oof_logistic = np.zeros(len(train))
predictions_logistic = np.zeros(len(test))

oof_svc = np.zeros(len(train))
predictions_svc = np.zeros(len(test))

oof_nb = np.zeros(len(train))
predictions_nb = np.zeros(len(test))


lgb_pred_te=np.array([0]*len(test))
print('begin logistic predict:')
for k,(train_in,test_in) in enumerate(skf.split(X,y)):
    print("fold {} begin:\n".format(k))
    X_train,X_test,y_train,y_test = X[train_in],X[test_in],y[train_in],y[test_in]
#    clf = LogisticRegression(verbose=1)
#    clf=GaussianNB()
    clf = SVC(probability=True)
    
    clf.fit(X_train,y_train)
#    oof_nb[test_in] = clf.predict_proba(X_test)[:,1]
#    predictions_nb += clf.predict_proba(test)[:,1]/skf.n_splits
    

#    oof_logistic[test_in] = clf.predict_proba(X_test)[:,1]
#    predictions_logistic += clf.predict_proba(test)[:,1] / skf.n_splits
    oof_svc[test_in] = clf.predict_proba(X_test)[:,1]
    predictions_svc += clf.predict_proba(test)[:,1] / skf.n_splits
    

#df_oof_nb_train=pd.DataFrame()
#df_oof_nb_train['uid']=train_uid
#df_oof_nb_train['nb_fea']=oof_nb
#df_oof_nb_train['label']=label
#df_oof_nb_train.to_csv('oof_nb_train.csv',index=None,sep='\t')
#
#df_oof_nb_test=pd.DataFrame()
#df_oof_nb_test['uid']=test_userid
#df_oof_nb_test['nb_fea']=predictions_nb
#df_oof_nb_test.to_csv('oof_nbc_test.csv',index=None,sep='\t')

#df_oof_logistic_train=pd.DataFrame()
#df_oof_logistic_train['uid']=train_uid
#df_oof_logistic_train['logistic_fea']=oof_logistic
#df_oof_logistic_train['label']=label
#df_oof_logistic_train.to_csv('oof_logistic_median_mad_train.csv',index=None,sep='\t')
#
#df_oof_xgb_test=pd.DataFrame()
#df_oof_xgb_test['uid']=test_userid
#df_oof_xgb_test['logistic_fea']=predictions_logistic
#df_oof_xgb_test.to_csv('oof_logistic_median_mad_test.csv',index=None,sep='\t')
 
    
df_oof_svc_train=pd.DataFrame()
df_oof_svc_train['uid']=train_uid
df_oof_svc_train['svc_fea']=oof_svc
df_oof_svc_train['label']=label
df_oof_svc_train.to_csv('oof_svc_median_mad_train.csv',index=None,sep='\t')

df_oof_svc_test=pd.DataFrame()
df_oof_svc_test['uid']=test_userid
df_oof_svc_test['svc_fea']=predictions_svc
df_oof_svc_test.to_csv('oof_svc_median_mad_test.csv',index=None,sep='\t')

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    