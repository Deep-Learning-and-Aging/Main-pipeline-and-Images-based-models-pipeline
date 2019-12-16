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
#Define dictionary of Predictions_tables, one for each application
PTs={}
#Application A (if organ in ["PhysicalActivity"]: #different set of eids)
#id set specific lines
data_features = pd.read_csv("/n/groups/patel/uk_biobank/main_data_9512/data_features.csv")[['f.eid', 'f.31.0.0', 'f.21003.0.0']]
data_features.replace({'f.31.0.0': {'Male': 0, 'Female': 1}}, inplace=True)
#non specific lines
data_features.columns = ['eid', 'Sex', 'Age']
data_features['eid'] = data_features['eid'].astype(str)
data_features['eid'] = data_features['eid'].apply(append_ext)
data_features = data_features.set_index('eid', drop=False)
PTs['A'] = data_features 
#Application B
data_features = pd.read_csv('/n/groups/patel/uk_biobank/main_data_52887/ukb37397.csv', usecols=['eid', '31-0.0', '21003-0.0'])
data_features.columns = ['eid', 'Sex', 'Age']
data_features['eid'] = data_features['eid'].astype(str)
data_features['eid'] = data_features['eid'].apply(append_ext)
data_features = data_features.set_index('eid', drop=False)
PTs['B'] = data_features

dict_organ_to_idset={'PhysicalActivity':'A', 'Liver':'B', 'Heart':'B'}

#load source dataframe (version depends on application)
list_models = glob.glob(path_store + 'Predictions_' + target + '_*_' + fold + '.csv')
list_models.sort()

for file_name in list_models:
    application = dict_organ_to_idset[file_name.split('_')[2]]
    #load csv
    prediction = pd.read_csv(path_store + file_name)
    prediction.rename(columns = {'outer_fold': 'outer_fold_' + prediction.columns.values[2]}, inplace = True)
    #merge csv
    PTs[application] = PTs[application].merge(prediction, how='outer', left_on='eid', right_on='eid')

