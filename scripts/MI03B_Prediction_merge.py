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
    sys.argv.append('Age') #target
    sys.argv.append('val') #inner fold

#read parameters from command
target = sys.argv[1]
fold = sys.argv[2]

# load the selected features
if organ in ["PhysicalActivity"]: #different set of eids
    data_features = pd.read_csv("/n/groups/patel/uk_biobank/main_data_9512/data_features.csv")[['f.eid', 'f.31.0.0', 'f.21003.0.0']]
    data_features.replace({'f.31.0.0': {'Male': 0, 'Female': 1}}, inplace=True)
else:
    data_features = pd.read_csv('/n/groups/patel/uk_biobank/main_data_52887/ukb37397.csv', usecols=['eid', '31-0.0', '21003-0.0'])
data_features.columns = ['eid', 'Sex', 'Age']
data_features['eid'] = data_features['eid'].astype(str)
data_features['eid'] = data_features['eid'].apply(append_ext)
data_features = data_features.set_index('eid', drop=False)
Predictions_table = data_features

#load source dataframe (version depends on application)
list_models = glob.glob(path_store + 'Predictions_' + target + '_*_' + fold + '.csv')
list_models = glob.glob(path_store + 'Predictions_' + target + '_*.csv')
list_models.sort()
list_versions = ['_'.join(parameters_list) for parameters_list in list_models_available]

Predictions_table = []
for file_name in list_models:
    #load csv
    prediction_model = pd.read_csv(path_store + file_name)
    #merge csv
    Predictions_table.merge(prediction_model, left_on='eid', right_on='eid')