# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 19:27:59 2019

@author: YWZQ
"""

import pandas as pd
lgb_feature_import=pd.read_csv('lgb_feature_importance.csv',encoding='gbk',sep='\t')
lgb_noimortan_feat=lgb_feature_import.loc[lgb_feature_import['importance']<120,:]
print(lgb_noimortan_feat['column'].tolist())