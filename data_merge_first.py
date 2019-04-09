# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:55:46 2019

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


X_train_no_shebei=pd.read_csv('train_no_shebei.csv',encoding='gbk',sep='\t')
X_train_shebei=pd.read_csv('shebei_id.csv',sep='\t')
#X_train_median_mad = pd.read_csv('train_feaadd_median_mad.csv',encoding='gbk',sep='\t')

X_train=X_train_no_shebei.merge(X_train_shebei,on=['uid'],how='left')
#X_train = X_train.merge(X_train_median_mad,on=['uid'],how='left')

X_train_label=pd.read_csv('train_labels.csv')
X_train_label.columns=['uid','label']
X_train=X_train.merge(X_train_label,on='uid',how='left')
print(X_train)
X_train.to_csv('x_all_label.csv',sep='\t',index=None)
'''

'''
X_test_no_shebei=pd.read_csv('test_no_shebei.csv',encoding='gbk',sep='\t')
X_test_shebei=pd.read_csv('shebei_id.csv',sep='\t')
#X_test_median_mad = pd.read_csv('test_feaadd_median_mad.csv',encoding='gbk',sep='\t')

X_test=X_test_no_shebei.merge(X_test_shebei,on='uid',how='left')
#X_test=X_test.merge(X_test_median_mad,on=['uid'],how='left')

print(X_test)
X_test.to_csv('x_test_all.csv',sep='\t',index=None)







