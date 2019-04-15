# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:48:29 2019

@author: Administrator
"""

import pandas as pd
import keras
from sklearn.model_selection import StratifiedKFold,KFold
from tqdm import *
import numpy as np

train=pd.read_csv('x_all_label.csv',sep='\t',encoding='gbk')
test=pd.read_csv('x_test_all.csv',sep='\t',encoding='gbk')
#print(train)

label=train['label']

train_uid=train['uid']

X_train=train.drop(['uid','label'],axis=1)


test_userid = test.pop('uid')

#print(X_train.columns.tolist())
test_columns=test.columns.tolist()
#print(test_columns)

#train_len=len(X_train)
#test_len=len(test)


#print(X_train.shape)
#print(test.shape)


#print(X_train.describe())

X_train.fillna(0,inplace=True)
print(test.isnull().any().astype('int').sum())

##奇怪，加了这个损失为nan准确率为0？？？？
#test=(test - test.mean()) / (test.std())
#X_train=(X_train - X_train.mean()) / (X_train.std())


test=test.values
X=X_train.values
y=label.values

X=np.array(X)
y=np.array(y)
y_len=len(y)
y=y.reshape(y_len,1)
K = 5
skf = StratifiedKFold(n_splits = K, shuffle = True ,random_state=16)
X=X.astype('float32')
print(X)
print(y)
print(X.shape)
print(y.shape)

######搭建神经网络#####
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D

model=Sequential()
model.add(Dense(units=800,input_dim=88,kernel_initializer='normal',activation='tanh'))
model.add(Dropout(0.5))

model.add(Dense(units=400,kernel_initializer='normal',activation='tanh'))
model.add(Dropout(0.5))

model.add(Dense(units=100,kernel_initializer='normal',activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(units=1,kernel_initializer='normal',activation='sigmoid'))
print(model.summary())
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history=model.fit(x=X,y=y,validation_split=0.3,epochs=80,batch_size=500,verbose=2)

import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history[train])
    plt.plot(train_history[validation])
    plt.title('train history')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'])
    plt.show()
show_train_history(train_history,'acc','val_acc')