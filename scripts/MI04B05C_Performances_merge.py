#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 16:02:42 2019

@author: Alan
"""

#load libraries, import functions and import parameters (nested import in the line below)
from MI_helpers import *
    
#default parameters
if len(sys.argv) != 5:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age') #target
    sys.argv.append('test') #inner_fold
    sys.argv.append('B') #id_set
    sys.argv.append('True') #ensemble_models. if True, only compute the performances for the ensemble models. Otherwise ignore them. Set False for MI04B and True for MI05B.

#read parameters from command
target = sys.argv[1]
fold = sys.argv[2]
id_set = sys.argv[3]
ensemble_models = convert_string_to_boolean(sys.argv[4])

#options
debug_mode = True
regenerate_performances = True
save_performances = True
names_metrics = dict_metrics_names[dict_prediction_types[target]]

#list the models that need to be merged
list_models = glob.glob(path_store + 'Performances_' + target + '_*_' + fold + '_' + id_set + '_str.csv')

#get rid of ensemble models
if ensemble_models:
    list_models = [model for model in list_models if ('*' in model | '?' in model | ',' in model) ]
else:
    list_models = [model for model in list_models if not ('*' in model | '?' in model | ',' in model) ]

#Fill the summary performances dataframe row by row
Performances_ranked = fill_summary_performances_matrix(list_models, target, fold, id_set, ensemble_models, save_performances)

#print the results
print('Performances_ranked: ')
print(Performances_ranked[['version', dict_main_metrics_names[target] + '_all']])

#exit
print('Done.')
sys.exit(0)

