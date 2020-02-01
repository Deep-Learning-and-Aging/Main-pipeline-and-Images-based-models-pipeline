#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 18:16:14 2019

@author: Alan
"""

#load libraries, import functions and import parameters (nested import in the line below)
from MI_helpers import *
    
#default parameters
if len(sys.argv) != 4:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age') #target
    sys.argv.append('train') #fold
    sys.argv.append('A') #id_set

#read parameters from command
target = sys.argv[1]
fold = sys.argv[2]
id_set = sys.argv[3]

#load the selected features
#Define dictionary of Predictions_tables, one for each id_set
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
data_features['eid'] = data_features['eid'].apply(append_ext) #TODO remove this after the predictions generated do not have .jpg anymore in the eid column
data_features = data_features.set_index('eid', drop=False)
data_features.index.name = 'column_names'

#For the training set, each sample is predicted n_CV_outer_folds times, so prepare a larger dataframe to receive the predictions
if fold == 'train':
    for outer_fold in outer_folds:
        df_fold = data_features.copy()
        df_fold['outer_fold'] = outer_fold
        data_features = df_fold if outer_fold == outer_folds[0] else data_features.append(df_fold)

#rename the dataframe afterpreprocessing it
Predictions_df = data_features

#generate list of predictions that will be integrated in the Predictions dataframe
list_models = glob.glob(path_store + 'Predictions_' + target + '_*_' + fold + '.csv')
#get rid of ensemble models
list_models = [ model for model in list_models if '*' not in model ]
list_models.sort()
#select the models relevant for the id_set
list_models = [model for model in list_models if any('_' + organ + '_' in model for organ in dict_idset_to_organ[id_set])]


#merge the predictions
for file_name in list_models:
    #garbage collector
    gc.collect()
    #load csv and format the predictions
    prediction = pd.read_csv(path_store + file_name)
    prediction['eid'] = prediction['eid'].apply(str)
    prediction['outer_fold'] = prediction['outer_fold'].apply(str)
    version = '_'.join(file_name.split('_')[1:-1])
    prediction['outer_fold_' + version] = prediction['outer_fold'] #create an extra column for further merging purposes on fold == 'train'
    #prediction.rename(columns={'pred': 'pred_' + version}) #TODO add this line after changing the MI03A script
    
    #merge data frames
    if fold == 'train':
        Predictions_df = Predictions_df.merge(prediction, how='outer', on=['eid', 'outer_fold'])
    else:
        prediction = prediction.drop(['outer_fold'], axis=1)
        Predictions_df = Predictions_df.merge(prediction, how='outer', on=['eid']) #not supported for panda version > 0.23.4 for now

#remove rows for which no prediction is available, before saving the Prediction tables
Predictions_df.dropna(subset=[col for col in Predictions_df.columns if 'Pred_' in col], how='all', inplace=True) #TODO change Pred to pred
Predictions_df.to_csv(path_store + 'PREDICTIONS_withoutEnsembles_' + target + '_' + fold + '_' + id_set + '.csv', index=False)

#exit
print('Done.')
sys.exit(0)
