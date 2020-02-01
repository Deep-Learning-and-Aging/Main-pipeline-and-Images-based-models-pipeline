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
version_ensemble = parameters_to_version(parameters)
main_metric_name = dict_main_metrics_names[target]
init_perf = -np.Inf if main_metrics_modes[main_metric_name] == 'max' else np.Inf
Performances = pd.read_csv(path_store + 'PERFORMANCES_withoutEnsembles_ranked_' + target + '_val_' + id_set + '.csv').set_index('version', drop=False)

#compute the new prediction for each inner fold, and save them in the Predictions dataframes
#prepare variables
PREDICTIONS = {}
for fold in folds:
    PREDICTIONS[fold] = pd.read_csv(path_store + 'PREDICTIONS_withoutEnsembles_' + target + '_' + fold + '_' + id_set + '.csv')

Predictions = PREDICTIONS['val']
y = Predictions[target]

#Compute the most general Ensemble model
Performances_subset = Performances[Performances['version'].isin(fnmatch.filter(Performances['version'], version_ensemble))]
#update_predictions_with_ensemble(PREDICTIONS, version_ensemble, folds, Performances_subset, Predictions, y, main_metric_name)


#define the parameters
main_metric_function = dict_metrics[main_metric_name]['sklearn']
main_metric_mode = main_metrics_modes[main_metric_name]
best_perf = -np.Inf if main_metric_mode == 'max' else np.Inf
ENSEMBLE_COLS={'select':{'noweights':[], 'weighted':[]}, 'all':{'noweights':[], 'weighted':[]}}

#iteratively add models to the ensemble
#for i, version in enumerate(Performances_subset['version']):
for i, version in enumerate(Performances_subset['version'][:1]):
    added_pred_name = 'Pred_' + version
    #two different strategies for the ensemble: 'select' only keeps the previous models that improved the accuracy. 'all' keeps all the previous models.
    for ensemble_type in ENSEMBLE_COLS.keys():
        #two different kinds of weighting: 'noweights' gives the same weight (1) to all the models included. 'weighted' weights based on the validation main metric.
        for weighting_type in ENSEMBLE_COLS[ensemble_type].keys():
            ENSEMBLE_COLS[ensemble_type][weighting_type].append(added_pred_name)
            if weighting_type == 'weighted':
                weights = Performances_subset.loc[[version.replace('Pred_','') for version in ENSEMBLE_COLS[ensemble_type][weighting_type]], main_metric_name + '_all'].values
            else:
                weights = np.ones(len(ENSEMBLE_COLS[ensemble_type][weighting_type]))
            Predictions_subset = Predictions[ENSEMBLE_COLS[ensemble_type][weighting_type]]
            Ensemble_predictions = Predictions_subset*weights
            Ensemble_predictions = Ensemble_predictions.sum(axis=1)/np.sum(weights)
        
        #format the dataframe
        df_ensemble = pd.concat([y, Ensemble_predictions], axis=1).dropna()
        df_ensemble.columns = ['y', 'pred']
        
        #evaluate the ensemble predictions
        print('The new perf is:' + str(round(new_perf,3)))
        new_perf = main_metric_function(df_ensemble['y'],df_ensemble['pred'])
        if new_perf > best_perf:
            best_perf = new_perf
            print('The new best perf is:' + str(round(best_perf,3)))
            best_pred = df_ensemble['pred']
            best_ensemble_namecols = ENSEMBLE_COLS[ensemble_type][weighting_type].copy()
            best_weights = weights
        elif ensemble_type == 'select':
            popped = ENSEMBLE_COLS[ensemble_type][weighting_type].pop()

def update_predictions_with_ensemble(PREDICTIONS, version_ensemble, folds, Performances_subset, Predictions, y, main_metric_name):
    best_ensemble_namecols, weights = build_ensemble_model(Performances_subset, Predictions, y, main_metric_name)
    best_ensemble_outerfolds = [model_name.replace('Pred_', 'outer_fold_') for model_name in best_ensemble_namecols]
    Ensemble_predictions = Predictions[best_ensemble_namecols]*weights
    Ensemble_predictions = Ensemble_predictions.mean(axis=1)/np.sum(weights)
    Ensemble_outerfolds = Predictions[best_ensemble_outerfolds]
    for fold in folds:
        PREDICTIONS[fold]['Pred_' + version_ensemble] = Ensemble_predictions
        if is_rank_one(Ensemble_outerfolds):
            print('The folds were shared by all the models in the ensemble models. Saving the folds too.')
            PREDICTIONS[fold]['outer_fold_' + version_ensemble] = Ensemble_outerfolds.mean(axis=1)
            print(PREDICTIONS[fold]['outer_fold_' + version_ensemble])
        else:
            PREDICTIONS[fold]['outer_fold_' + version_ensemble] = np.nan


def build_ensemble_model(Performances_subset, Predictions, y, main_metric_name):
    main_metric_function = dict_metrics[main_metric_name]['sklearn']
    main_metric_mode = main_metrics_modes[main_metric_name]
    best_perf = -np.Inf if main_metric_mode == 'max' else np.Inf
    ENSEMBLE_COLS={'select':{'noweights':[], 'weighted':[]}, 'all':{'noweights':[], 'weighted':[]}}
    #iteratively add models to the ensemble
    for i, version in enumerate(Performances_subset['version']):
        added_pred_name = 'Pred_' + version
        #added_pred_name = 'Pred_' + version.replace('_str.csv','') CAN I DELETE? is line above working? TODO
        for ensemble_type in ENSEMBLE_COLS.keys():
            for weighting_type in ENSEMBLE_COLS[ensemble_type].keys():
                #print(ensemble_type)
                ENSEMBLE_COLS[ensemble_type].append(added_pred_name)
                if weighting_type == 'weighted':
                    weights = Performances_subset.loc[ENSEMBLE_COLS[ensemble_type], main_metric_name + '_val']
                else:
                    weights = np.ones(len(ENSEMBLE_COLS[ensemble_type]))
                Ensemble_predictions = Predictions[ENSEMBLE_COLS[ensemble_type]]*weights
                Ensemble_predictions = Ensemble_predictions.mean(axis=1)/np.sum(weights)
            df_ensemble = pd.concat([y, Ensemble_predictions], axis=1).dropna()
            df_ensemble.columns = ['y', 'pred']
            #evaluate the ensemble predictions
            new_perf = main_metric_function(df_ensemble['y'],df_ensemble['pred'])
            new_perf_weighted = main_metric_function(df_ensemble['y'],df_ensemble_weighted['pred'])
            if new_perf > best_perf:
                best_perf = new_perf
                best_pred = df_ensemble['pred']
                best_ensemble_namecols = ENSEMBLE_COLS[ensemble_type].copy()
                best_weights = weights
            elif ensemble_type == 'select':
                ENSEMBLE_COLS[ensemble_type].pop()
    return best_ensemble_namecols, weights









best_ensemble_namecols, weights = build_ensemble_model(Performances_subset, Predictions, y, main_metric_name)
best_ensemble_outerfolds = [model_name.replace('Pred_', 'outer_fold_') for model_name in best_ensemble_namecols]
Ensemble_predictions = Predictions[best_ensemble_namecols]*weights
Ensemble_predictions = Ensemble_predictions.mean(axis=1)/np.sum(weights)
Ensemble_outerfolds = Predictions[best_ensemble_outerfolds]
for fold in folds:
    PREDICTIONS[fold]['Pred_' + version] = Ensemble_predictions
    if is_rank_one(Ensemble_outerfolds):
        print('The folds were shared by all the models in the ensemble models. Saving the folds too.')
        PREDICTIONS[fold]['outer_fold_' + version] = Ensemble_outerfolds.mean(axis=1)
        print(PREDICTIONS[fold]['outer_fold_' + version])
    else:
        PREDICTIONS[fold]['outer_fold_' + version] = np.nan








#For each organ, compute a separate ensemble model
list_organs = Performances_subset['organ'].unique()
for organ in list_organs:
    print(organ)
    parameters_organ = parameters.copy()
    parameters_organ['organ'] = organ
    version_organ = parameters_to_version(parameters_organ)
    Performances_subset_organ = Performances_subset[Performances_subset['version'].isin(fnmatch.filter(Performances_subset['version'], version_organ))]
    update_predictions_with_ensemble(PREDICTIONS, version_organ, folds, Performances_subset_organ, Predictions, y, main_metric_name)
    
    #For each field_id, compute an ensemble model
    list_field_ids = [str(field_id) for field_id in Performances_subset_organ['field_id'].unique()]
    for field_id in list_field_ids:
        print(field_id)
        parameters_field_id = parameters_organ.copy()
        parameters_field_id['field_id'] = field_id
        version_field_id = parameters_to_version(parameters_field_id)
        Performances_subset_field_id = Performances_subset_organ[Performances_subset_organ['version'].isin(fnmatch.filter(Performances_subset_organ['version'], version_field_id))]
        update_predictions_with_ensemble(PREDICTIONS, version_field_id, folds, Performances_subset_field_id, Predictions, y, main_metric_name)
        
        #For each view, compute an ensemble model
        list_views = Performances_subset_field_id['view'].unique()
        for view in list_views:
            print(view)
            parameters_view = parameters_field_id.copy()
            parameters_view['view'] = view
            version_view = parameters_to_version(parameters_view)
            Performances_subset_view = Performances_subset_field_id[Performances_subset_field_id['version'].isin(fnmatch.filter(Performances_subset_field_id['version'], version_view))]
            update_predictions_with_ensemble(PREDICTIONS, version_view, folds, Performances_subset_view, Predictions, y, main_metric_name)
            
            #For each transformation, compute a separate ensemble model
            list_transformations = Performances_subset_view['transformation'].unique()
            for transformation in list_transformations:
                print(transformation)
                parameters_transformation = parameters_view.copy()
                parameters_transformation['transformation'] = transformation
                version_transformation = parameters_to_version(parameters_transformation)
                Performances_subset_transformation = Performances_subset_view[Performances_subset_view['version'].isin(fnmatch.filter(Performances_subset_view['version'], version_view))]
                update_predictions_with_ensemble(PREDICTIONS, version_transformation, folds, Performances_subset_transformation, Predictions, y, main_metric_name)

#save ensemble predictions
for fold in folds:
    PREDICTIONS[fold].to_csv(path_store + 'PREDICTIONS_withEnsembles_' + target + '_' + fold + '_' + id_set + '.csv', index=False)
