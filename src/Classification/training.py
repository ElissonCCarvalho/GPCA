#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 21:46:11 2021

@author: elisson
"""

from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import GridSearchCV

def parameters(x_train):
    params = {'optimizer'       : ['adam'],
               'batch_size'     : [8],
               'epochs'         : [1000],
               'hidden_layers'  : [7],
               'units'          : [145],
               'input_dim'      : [len(x_train[0])]
              }
 
    return params

def custom_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (tn / (tn + fp))

def custom_sensitivity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (tp / (tp + fn))

def evaluation_metrics():
    metrics = {'accuracy'      :   'accuracy',
               'roc_auc'       :   'roc_auc',
               'f1'            :   'f1',
               'sensitivity'   :   make_scorer(custom_sensitivity),
               'specificity'   :   make_scorer(custom_specificity)}
    
    return metrics
    
def grid_search(classifier, x_train):
    param_grid = parameters(x_train)
    metrics = evaluation_metrics()    
        
    grid_search = GridSearchCV(estimator = classifier,
                           verbose = 3,
                           param_grid = param_grid,
                           n_jobs = None,
                           scoring = metrics,
                           refit = 'accuracy', 
                           cv = 5)
    
    return grid_search
   
