#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 21:44:57 2021

@author: elisson
"""

import json
import time
import numpy as np

import matplotlib.pyplot as plt


def cv_results(grid_result):
    cv_results = '===================================================================================\n\t\t\t   CROSS VALIDATION RESULTS\n'            
    cv_results += '===================================================================================\n'
    
    cv_results += '\n-----------------------------------------------------------------------------------'
    cv_results += '\nBEST PARAMS: ' + json.dumps(grid_result.best_params_).replace('{', '').replace('}', '').replace('"', '')
    cv_results += '\n-----------------------------------------------------------------------------------'
    
    for i in range(len(grid_result.cv_results_['params'])):                
        cv_results += '\n-----------------------------------------------------------------------------------'
        cv_results += '\nPARAMS: ' + json.dumps(grid_result.cv_results_['params'][i]).replace('{', '').replace('}', '').replace('"', '')
        cv_results += '\n-----------------------------------------------------------------------------------'
        cv_results += '\nFOLD\tACCURACY\tROC_AUC\t\tF1 SCORE\tSENSITIVITY\tSPECIFICITY'
        cv_results += '\n-----------------------------------------------------------------------------------\n'
        
        for j in range(grid_result.n_splits_):
            cv_results += str(j) + '\t'
            cv_results += '{:.2f} %  \t'.format(grid_result.cv_results_['split{}_test_accuracy'.format(j)][i]*100)
            cv_results += '{:.2f} %  \t'.format(grid_result.cv_results_['split{}_test_roc_auc'.format(j)][i]*100)
            cv_results += '{:.2f} %  \t'.format(grid_result.cv_results_['split{}_test_f1'.format(j)][i]*100)
            cv_results += '{:.2f} %  \t'.format(grid_result.cv_results_['split{}_test_sensitivity'.format(j)][i]*100)
            cv_results += '{:.2f} %  \n'.format(grid_result.cv_results_['split{}_test_specificity'.format(j)][i]*100)            
        
        cv_results += '-----------------------------------------------------------------------------------\n'
        cv_results += 'MEAN\t'
        cv_results += '{:.2f} %  \t'.format(grid_result.cv_results_['mean_test_accuracy'][i]*100)  
        cv_results += '{:.2f} %  \t'.format(grid_result.cv_results_['mean_test_roc_auc'][i]*100)  
        cv_results += '{:.2f} %  \t'.format(grid_result.cv_results_['mean_test_f1'][i]*100)  
        cv_results += '{:.2f} %  \t'.format(grid_result.cv_results_['mean_test_sensitivity'][i]*100)  
        cv_results += '{:.2f} %'.format(grid_result.cv_results_['mean_test_specificity'][i]*100) 
        
        cv_results += '\n-----------------------------------------------------------------------------------\n'
        cv_results += 'STD\t'
        cv_results += '{:.2f}\t\t'.format(grid_result.cv_results_['std_test_accuracy'][i]*100)  
        cv_results += '{:.2f}\t\t'.format(grid_result.cv_results_['std_test_roc_auc'][i]*100)  
        cv_results += '{:.2f}\t\t'.format(grid_result.cv_results_['std_test_f1'][i]*100)  
        cv_results += '{:.2f}\t\t'.format(grid_result.cv_results_['std_test_sensitivity'][i]*100)  
        cv_results += '{:.2f}'.format(grid_result.cv_results_['std_test_specificity'][i]*100) 
        cv_results += '\n-----------------------------------------------------------------------------------\n'
    
    return cv_results

def plot_acc_loss(grid_result, name):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    history = grid_result.best_estimator_.model.history.history
    
    ax.plot(np.arange(len(history['val_loss'])), history['val_loss'], label='Validation loss')
    ax.plot(np.arange(len(history['val_accuracy'])), history['val_accuracy'], label='Validation accuracy')
    
    ax.plot(np.arange(len(history['loss'])), history['loss'], label='Loss')
    ax.plot(np.arange(len(history['accuracy'])), history['accuracy'], label='Accuracy')
        
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    ax.set_xlabel('Epochs', fontsize = 12.0)
    ax.set_ylabel('Loss/Accuracy', fontsize = 12.0)
    
    ax.set_yticks(np.arange(0, max(ax.get_yticks()) + 1))
    ax.grid(True)
    
    fig.tight_layout(pad=0.4)
        
    fig.savefig(name, dpi=500)

def show_results(grid_result, data):
    result = cv_results(grid_result)
    print(result)
    
    name = '../../results/Files/{}/{}.txt'.format(data, time.time())
    
    #print(grid_result.best_estimator_.model.history.history)
    
    file = open(name, 'a')
    file.write(result)
    
    plot_acc_loss(grid_result, name.replace('Files', 'Graphics').replace('txt', 'png'))
        
    

        
