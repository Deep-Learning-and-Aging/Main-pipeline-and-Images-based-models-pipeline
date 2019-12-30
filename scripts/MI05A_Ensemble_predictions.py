#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 17:17:38 2019

@author: Alan
"""

#load libraries, import functions and import parameters (nested import in the line below)
from MI_helpers import *
    
#default parameters
if len(sys.argv) != 3:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Sex') #target
    sys.argv.append('val') #inner fold

#read parameters from command
target = sys.argv[1]
fold = sys.argv[2]

#which ensemble models to create?
#best models altogether no discrimination
#best models from a specific organ
#

#load performances and predictions
for id_set in id_sets:
    #debug


id_set = 'B'
Performances = pd.read_csv(path_store + 'PERFORMANCES_ranked_' + target + '_' + fold + '.csv')
#Predictions = pd.read_csv(path_store + 'PREDICTIONS_' + target + '_' + fold + '_' + id_set + '.csv')
#below is temp for debug, keep the one abvoe TODO
Predictions = pd.read_csv(path_store + 'Predictions_' + target + '_' + fold + '_' + id_set + '.csv')

main_metric_name = dict_metrics_names[dict_prediction_types[target]]
main_metric_function = dict_metrics[main_metric_name]['sklearn']
Ensemble_cols = []
perf_ensemble = Performances[main_metric_name][0]
#build performance model with everything allowed
for i, version in enumerate(Performances['version']):
    #Ensemble_cols.append('Pred_' + version)
    #TODO remove below keep above
    Ensemble_cols.append('Pred_' + version.replace('_str.csv',''))
    Ensemble_predictions = Predictions[Ensemble_cols].mean(axis=1)
    #evaluate the ensemble predictions
    perf_ensemble = 1
    if perf_ensemble > 
    
