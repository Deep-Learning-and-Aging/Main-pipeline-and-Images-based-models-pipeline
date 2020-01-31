#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 17:17:38 2019

@author: Alan
"""

#load libraries, import functions and import parameters (nested import in the line below)
from MI_helpers import *
    
#default parameters
if len(sys.argv) != 2:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age') #target

#read parameters from command
target = sys.argv[1]

#TODO TEMP DEBUG

folds = folds[1:]
id_sets=['B']

#set other parameters
parameters = {'target':target, 'organ':'*', 'field_id':'*', 'view':'*', 'transformation':'*', 'architecture':'*', 'optimizer':'*', 'learning_rate':'*', 'weight_decay':'*', 'dropout_rate':'*'}
version = parameters_to_version(parameters)
main_metric_name = dict_main_metrics_names[target]
main_metric_function = dict_metrics[main_metric_name]['sklearn']
main_metric_mode = main_metrics_modes[main_metric_name]
init_perf = -np.Inf if main_metrics_modes[main_metric_name] == 'max' else np.Inf
Performances = pd.read_csv(path_store + 'PERFORMANCES_withoutEnsembles_ranked_' + target + '_val.csv')

#compute the new prediction for each inner fold, and save them in the Predictions dataframes
PREDICTIONS={}
for id_set in id_sets:
    PREDICTIONS[id_set] = {}
    for fold in folds:
        PREDICTIONS[id_set][fold] = pd.read_csv(path_store + 'PREDICTIONS_withoutEnsembles_' + target + '_' + fold + '_' + id_set + '.csv')
    
    Predictions = PREDICTIONS[id_set]['val']
    y = Predictions[target]
    print(id_set)
    #Compute the most general Ensemble model
    Performances_subset = Performances[Performances['version'].isin(fnmatch.filter(Performances['version'], version))]
    update_predictions_with_ensemble(PREDICTIONS, version, id_set, folds, Performances_subset, Predictions, y, main_metric_function, main_metric_mode)
    #For each organ, compute a separate ensemble model
    list_organs = Performances_subset['organ'].unique()
    for organ in list_organs:
        print(organ)
        parameters_organ = parameters.copy()
        parameters_organ['organ'] = organ
        version_organ = parameters_to_version(parameters_organ)
        Performances_subset_organ = Performances_subset[Performances_subset['version'].isin(fnmatch.filter(Performances_subset['version'], version_organ))]
        update_predictions_with_ensemble(PREDICTIONS, version_organ, id_set, folds, Performances_subset_organ, Predictions, y, main_metric_function, main_metric_mode)
        
        #For each field_id, compute an ensemble model
        list_field_ids = [str(field_id) for field_id in Performances_subset_organ['field_id'].unique()]
        for field_id in list_field_ids:
            print(field_id)
            parameters_field_id = parameters_organ.copy()
            parameters_field_id['field_id'] = field_id
            version_field_id = parameters_to_version(parameters_field_id)
            Performances_subset_field_id = Performances_subset_organ[Performances_subset_organ['version'].isin(fnmatch.filter(Performances_subset_organ['version'], version_field_id))]
            update_predictions_with_ensemble(PREDICTIONS, version_field_id, id_set, folds, Performances_subset_field_id, Predictions, y, main_metric_function, main_metric_mode)
            
            #For each view, compute an ensemble model
            list_views = Performances_subset_field_id['view'].unique()
            for view in list_views:
                print(view)
                parameters_view = parameters_field_id.copy()
                parameters_view['view'] = view
                version_view = parameters_to_version(parameters_view)
                Performances_subset_view = Performances_subset_field_id[Performances_subset_field_id['version'].isin(fnmatch.filter(Performances_subset_field_id['version'], version_view))]
                update_predictions_with_ensemble(PREDICTIONS, version_view, id_set, folds, Performances_subset_view, Predictions, y, main_metric_function, main_metric_mode)
                
                #For each transformation, compute a separate ensemble model
                list_transformations = Performances_subset_view['transformation'].unique()
                for transformation in list_transformations:
                    print(transformation)
                    parameters_transformation = parameters_view.copy()
                    parameters_transformation['transformation'] = transformation
                    version_transformation = parameters_to_version(parameters_transformation)
                    Performances_subset_transformation = Performances_subset_view[Performances_subset_view['version'].isin(fnmatch.filter(Performances_subset_view['version'], version_view))]
                    update_predictions_with_ensemble(PREDICTIONS, version_transformation, id_set, folds, Performances_subset_transformation, Predictions, y, main_metric_function, main_metric_mode)
    
    #save ensemble predictions
    for id_set in id_sets:
        for fold in folds:
            PREDICTIONS[id_set][fold].to_csv(path_store + 'PREDICTIONS_withEnsembles_' + target + '_' + fold + '_' + id_set + '.csv', index=False)
