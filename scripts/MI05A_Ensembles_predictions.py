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
    sys.argv.append('Age') #target
    sys.argv.append('B') #id_set

#read parameters from command
target = sys.argv[1]
id_set = sys.argv[2]

#set other parameters
parameters = {'target':target, 'organ':'*', 'field_id':'*', 'view':'*', 'transformation':'*', 'architecture':'*', 'optimizer':'*', 'learning_rate':'*', 'weight_decay':'*', 'dropout_rate':'*'}
version = parameters_to_version(parameters)
main_metric_name = dict_main_metrics_names[target]
init_perf = -np.Inf if main_metrics_modes[main_metric_name] == 'max' else np.Inf
Performances = pd.read_csv(path_store + 'PERFORMANCES_tuned_ranked_' + target + '_val_' + id_set + '.csv').set_index('version', drop=False)
Performances['field_id'] = Performances['field_id'].astype(str)

#compute the new prediction for each inner fold, and save them in the Predictions dataframes
#prepare variables
PREDICTIONS = {}
for fold in folds:
    PREDICTIONS[fold] = pd.read_csv(path_store + 'PREDICTIONS_tuned_' + target + '_' + fold + '_' + id_set + '.csv')

#Generate all the ensemble models
list_ensemble_levels = ['transformation', 'view', 'field_id', 'organ'] #list in reverse order for the recursive call of the ensemble building algo purpose
recursive_ensemble_builder(PREDICTIONS, target, main_metric_name, id_set, Performances, parameters, version, list_ensemble_levels)

#save ensemble predictions
for fold in folds:
    PREDICTIONS[fold].to_csv(path_store + 'PREDICTIONS_withEnsembles_' + target + '_' + fold + '_' + id_set + '.csv', index=False)

print('Done.')
sys.exit(0)






