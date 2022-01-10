#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 21:51:11 2021

@author: elisson
"""

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

def category_columns(X):
    head = X.columns
    label_columns = []
    one_hot_columns = []
    for i in range(len(X.dtypes)):
        if(X.dtypes[i] == object):
            if(len(X[head[i]].unique()) <= 2):   
                label_columns.append(head[i])
            else:
                one_hot_columns.append(head[i])   
                
    return label_columns, one_hot_columns

def encode(X):
    label_columns, one_hot_columns = category_columns(X)        
    
    if(label_columns):
        le = LabelEncoder()
        print('Label Columns: ', label_columns)
        X[label_columns] = X[label_columns].apply(le.fit_transform)
    
    if(one_hot_columns):
        print('On Hot Columns: ', one_hot_columns)
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), one_hot_columns)],
                               remainder='passthrough')
        X = ct.fit_transform(X)    
                
    return X