#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:43:01 2020

@author: Alan
"""

#load libraries, import functions and import parameters (nested import in the line below)
from MI_helpers import *

#default parameters
if len(sys.argv) != 3:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age') #target
    sys.argv.append('B') #id_set
    
#read parameters from command
target = sys.argv[1]
id_set = sys.argv[2]

#Load the predictions
PREDICTIONS={}
for fold in folds:
    Predictions = pd.read_csv(path_store +'PREDICTIONS_withoutEnsembles_' + target + '_' + fold + '_' + id_set + '.csv')

#print trainin history
list_versions = [col_name.replace('pred_', '') for col_name in PREDICTIONS['test'].columns.values if 'pred_' in col_name]
for version in list_versions:
    for outer_fold in outer_folds:
        plot_logger(path_store=path_store, version=version + '_' + outer_fold, display_learning_rate=True)
