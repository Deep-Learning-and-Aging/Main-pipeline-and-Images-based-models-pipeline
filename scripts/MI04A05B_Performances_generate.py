#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 15:40:44 2019

@author: Alan
"""

#load libraries, import functions and import parameters (nested import in the line below)
from MI_helpers import *

#options
#debunk mode: exclude train set
debug_mode = True
#regenerate performances if already exists
regenerate_performances = True
#save performances
save_performances = True

#set parameters for debug mode
if debug_mode:
    n_bootstrap_iterations = 10

##default parameters
#if len(sys.argv) != 11:
#    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
#    sys.argv = ['']
#    sys.argv.append('Age') #target
#    sys.argv.append('PhysicalActivity_90001_main') #image_type, e.g PhysicalActivity_90001_main, Liver_20204_main or Heart_20208_3chambers
#    sys.argv.append('raw') #transformation
#    sys.argv.append('DenseNet121') #architecture
#    sys.argv.append('Adam') #optimizer
#    sys.argv.append('0.0001') #learning_rate
#    sys.argv.append('0.0') #weight decay
#    sys.argv.append('0.0') #dropout
#    sys.argv.append('val') #fold
#    sys.argv.append('id_set') #id_set

#default parameters
if len(sys.argv) != 11:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age') #target
    sys.argv.append('*_*_*') #image_type, e.g PhysicalActivity_90001_main, Liver_20204_main or Heart_20208_3chambers
    sys.argv.append('*') #transformation
    sys.argv.append('*') #architecture
    sys.argv.append('*') #optimizer
    sys.argv.append('*') #learning_rate
    sys.argv.append('*') #weight decay
    sys.argv.append('*') #dropout
    sys.argv.append('val') #fold
    sys.argv.append('B') #id_set

#set other parameters accordingly
target, image_type, organ, field_id, view, preprocessing, architecture, optimizer, learning_rate, weight_decay, dropout_rate, fold, id_set = read_parameters_from_command(sys.argv)
version = target + '_' + image_type + '_' + preprocessing + '_' + architecture + '_' + optimizer + '_' + str(learning_rate) + '_' + str(weight_decay) + '_' + str(dropout_rate)
names_metrics = dict_metrics_names[dict_prediction_types[target]]

if os.path.exists(path_store + 'Performances_' + version + '_' + fold + '_str.csv') and (not regenerate_performances):
    print('The performances have already been generated.')
    sys.exit(0)

#Preprocess the predictions
data_features = preprocess_data_features_predictions_for_performances(path_store, id_set, target)
Predictions = preprocess_predictions_for_performances(data_features, path_store, version, fold, id_set)

#Fill the columns for this model, outer_fold by outer_fold
PERFORMANCES = fill_performances_matrix_for_single_model(Predictions, target, version, fold, id_set, names_metrics, n_bootstrap_iterations, save_performances)

#print the results
print('PERFORMANCES: ')
print(PERFORMANCES)

#exit
print('Done.')
sys.exit(0)

