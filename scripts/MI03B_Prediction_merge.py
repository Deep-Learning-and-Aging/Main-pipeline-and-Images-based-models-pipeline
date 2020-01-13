#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 18:16:14 2019

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

#load the selected features
#Define dictionary of Predictions_tables, one for each id_set
PREDICTIONS_TABLES={}
for id_set in id_sets:
    if id_set == 'A':
        data_features = pd.read_csv("/n/groups/patel/uk_biobank/main_data_9512/data_features.csv")[['f.eid', 'f.31.0.0', 'f.21003.0.0']]
        data_features.replace({'f.31.0.0': {'Male': 0, 'Female': 1}}, inplace=True)
    elif id_set == 'B':
        data_features = pd.read_csv('/n/groups/patel/uk_biobank/main_data_52887/ukb37397.csv', usecols=['eid', '31-0.0', '21003-0.0'])
    else:
        print('ERROR: id_set must be either A or B')
        sys.exit()
    data_features.columns = ['eid', 'Sex', 'Age']
    data_features['eid'] = data_features['eid'].astype(str)
    data_features['eid'] = data_features['eid'].apply(append_ext)
    data_features = data_features.set_index('eid', drop=False)
    data_features.index.name = 'column_names'
    PREDICTIONS_TABLES[id_set] = data_features

#For the training set, each sample is predicted n_CV_outer_folds times, so prepare a larger dataframe to receive the predictions
if fold == 'train':
    for id_set in id_sets:
        for outer_fold in outer_folds:
            df_fold = PREDICTIONS_TABLES[id_set].copy()
            df_fold['outer_fold'] = outer_fold
            df_total = df_fold if outer_fold == outer_folds[0] else df_total.append(df_fold)
        PREDICTIONS_TABLES[id_set] = df_total

#generate list of predictions that will be integrated in the Predictions dataframe
list_models = glob.glob(path_store + 'Predictions_' + target + '_*_' + fold + '.csv')
list_models.sort()
#merge the predictions
for file_name in list_models:
    id_set = dict_organ_to_idset[file_name.split('_')[2]]
    #load csv
    prediction = pd.read_csv(path_store + file_name)
    prediction['outer_fold'] = prediction['outer_fold'].apply(str)
    prediction['outer_fold_' + '_'.join(file_name.split('_')[1:-1])] = prediction['outer_fold']
    #merge csv
    if fold == 'train':
        PREDICTIONS_TABLES[id_set] = PREDICTIONS_TABLES[id_set].merge(prediction, how='outer', on=['eid', 'outer_fold'])
    else:
        prediction = prediction.drop(['outer_fold'], axis=1)
        PREDICTIONS_TABLES[id_set] = PREDICTIONS_TABLES[id_set].merge(prediction, how='outer', on=['eid']) #not supported for panda version > 0.23.4 for now

#remove rows for which no prediction is available, before saving the Prediction tables
for id_set in id_sets:
    PREDICTIONS_TABLES[id_set].dropna(subset=[col for col in PREDICTIONS_TABLES[id_set].columns if 'Pred' in col], how='all', inplace=True)
    PREDICTIONS_TABLES[id_set].to_csv(path_store + 'PREDICTIONS_' + target + '_' + fold + '_' + id_set + '.csv', index=False)

#exit
print('Done.')
sys.exit(0)
