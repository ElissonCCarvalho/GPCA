#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 17:14:25 2021

@author: elisson
"""

import pandas as pd
import tensorflow as tf

from model import create_model
from results import show_results
from training import grid_search
from preprocessing import encode

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.wrappers.scikit_learn import KerasClassifier

def preprocessing(X, y):
    #X = encode(X)
    sc = StandardScaler()
    X = sc.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    return x_train, x_test, y_train, y_test
    
def training(x_train, x_test, y_train, y_test):
    classifier = KerasClassifier(build_fn=create_model, verbose=2)
    grid = grid_search(classifier, x_train)
    
    return grid.fit(x_train, y_train, validation_data=(x_test, y_test))
    
if __name__ == '__main__':
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    
    data = 'dataset'
    
    dataset = pd.read_csv('../../{}.csv'.format(data)).dropna()
            
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    
    x_train, x_test, y_train, y_test = preprocessing(X, y)
    
    grid_result = training(x_train, x_test, y_train, y_test)
    show_results(grid_result, data)
        
    
    
    
    
    