#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:29:53 2019

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

#options
debug_mode = True
regenerate_performances = True
save_performances = True

if debug_mode:
    n_bootstrap = 10
    #outer_folds = ['2', '3']
    #id_sets = ['A']

#Define the columns of the Performances dataframe
#columns for sample sizes
names_sample_sizes = ['N']
if target in targets_binary:
    names_sample_sizes.extend(['N_0', 'N_1'])

#columns for metrics
names_metrics = dict_metrics_names[dict_prediction_types[target]]
#for normal folds, keep track of metric and bootstrapped metric's sd
names_metrics_with_sd = []
for name_metric in names_metrics:
    names_metrics_with_sd.extend([name_metric, name_metric + '_sd', name_metric + '_str'])

#for the 'all' fold, also keep track of the 'folds_sd', the metric's sd calculated using the folds' metrics results
names_metrics_with_folds_sd_and_sd = []
for name_metric in names_metrics:
    names_metrics_with_folds_sd_and_sd.extend([name_metric, name_metric + '_folds_sd', name_metric + '_sd', name_metric + '_str'])

#merge all the columns together. First description of the model, then sample sizes and metrics for each fold
names_col_Performances = ['version'] + names_model_parameters #.copy()
#special outer fold 'all'
names_col_Performances.extend(['_'.join([name,'all']) for name in names_sample_sizes + names_metrics_with_folds_sd_and_sd])
#other outer_folds
for outer_fold in outer_folds:
    names_col_Performances.extend(['_'.join([name,outer_fold]) for name in names_sample_sizes + names_metrics_with_sd])


#Fill the Performances tables (one for each id_set)
for id_set in id_sets:
    #Move on to next id set if no predictions table is available
    print('Evaluating models for id_set: ' + id_set)
    try:
        Predictions_table = pd.read_csv(path_store + 'Predictions_' + target + '_' + fold + '_' + id_set + '.csv')
    except:
        print('Predictions table ' + 'Predictions_' + target + '_' + fold + '_' + id_set + '.csv' + ' is not available.')
        continue
    
    #list the models that need to be evaluated
    list_models = [col for col in Predictions_table.columns if 'Pred' in col]
    #load the previously calculated Performances table and only compute performance for new models
    if (not regenerate_performances) & os.path.exists(path_store + 'Performances_alphabetical_' + target + '_' + fold + '_' + id_set + '.csv'):
        Performances_previous = pd.read_csv(path_store + 'Performances_alphabetical_' + target + '_' + fold + '_' + id_set + '.csv')
        list_models = [model for model in list_models if model.replace('Pred_', '') not in Performances_previous['version'].values]
    #move to next id_set if no models need to be evaluated
    if(len(list_models) == 0):
        print('No new models need to be evaluated in the prediction table ' + 'Predictions_' + target + '_' + fold + '_' + id_set + '.csv')
        continue
    print(str(len(list_models)) + ' new models need to be evaluated.')
    
    #Generate the Performance table from the rows and columns.
    Performances = np.empty((len(list_models),len(names_col_Performances),))
    Performances.fill(np.nan)
    Performances = pd.DataFrame(Performances)
    Performances.columns = names_col_Performances
    for colname in Performances.columns.values:
        if (colname in names_model_parameters) | ('_str' in colname):
            col_type = str
        else:
            col_type = float
        Performances[colname] = Performances[colname].astype(col_type)
    
    #Fill the Performance table row by row
    for i, model in enumerate(list_models):
        #Fill the columns corresponding to the model's parameters
        model = '_'.join(model.split('_')[1:])
        model_parameters = version_to_parameters(model, names_model_parameters)
        #fill the columns for model parameters
        Performances['version'][i] = model
        for parameter_name in names_model_parameters:
            Performances[parameter_name][i] = model_parameters[parameter_name]
        
        #generate a subdataframe from the main predictions table, specific to this model
        predictions_model = Predictions_table[['eid', target, 'outer_fold_' + model, 'Pred_' + model]].dropna(how='any')
        predictions_model.columns = ['eid', 'y', 'outer_fold', 'pred']
        predictions_model['outer_fold'] = predictions_model['outer_fold'].apply(int).apply(str)
        performances_model = model.split('_')
        
        #Fill the columns for this model, outer_fold by outer_fold
        print('Evaluating the model: ' + model)
        for outer_fold in ['all'] + outer_folds:
            #Generate a subdataframe from the predictions table for each outerfold
            if outer_fold == 'all':
                predictions_fold = predictions_model.copy()
            else:
                predictions_fold = predictions_model[predictions_model['outer_fold'] == outer_fold]
            
            #if no samples are available for this fold, fill columns with nans
            sample_sizes_fold = []
            if(len(predictions_fold.index) == 0):
                print('NO SAMPLES AVAILABLE FOR MODEL ' + model + ' IN OUTER_FOLD ' + outer_fold)                    
            else:
                #Fill sample size columns
                Performances['N_' + outer_fold][i] = len(predictions_fold.index)
                
                #For binary classification, calculate sample sizes for each class and generate class prediction
                if target in targets_binary:
                    Performances['N_0_' + outer_fold][i] = len(predictions_model[predictions_model['y']==0].index)
                    Performances['N_1_' + outer_fold][i] = len(predictions_model[predictions_model['y']==1].index)
                    predictions_fold_class = predictions_fold.copy()
                    predictions_fold_class['pred'] = predictions_fold_class['pred'].round()
                
                #Fill the Performances dataframe metric by metric
                y = predictions_model['y']
                for name_metric in names_metrics:
                    predictions_metric = predictions_fold_class if name_metric in metrics_needing_classpred else predictions_fold
                    metric_function = dict_metrics[name_metric]['sklearn']
                    Performances[name_metric + '_' + outer_fold][i] = metric_function(predictions_metric['y'], predictions_metric['pred'])
                    Performances[name_metric + '_sd_' + outer_fold][i] = bootstrap(predictions_metric, n_bootstrap, metric_function)[1]
                    Performances[name_metric + '_str_' + outer_fold][i] = str(round(Performances[name_metric + '_' + outer_fold][i],3)) + '+-' + str(round(Performances[name_metric + '_sd_' + outer_fold][i],3))

        #Display the performance for this model
        print(dict_main_metrics_names[model_parameters['target']] + ' = ' + Performances[dict_main_metrics_names[model_parameters['target']] + '_str_all'][i])

    #Calculate folds_sd: standard deviation in the metrics between the different folds
    for name_metric in names_metrics:
        names_cols =[]
        for outer_fold in outer_folds:
            names_cols.append(name_metric + '_' + outer_fold)
        Performances[name_metric + '_folds_sd_all'] = Performances[names_cols].std(axis=1, skipna=True)
        for i in range(len(Performances.index)):
            Performances[name_metric + '_str_all'][i] = str(round(Performances[name_metric + '_all'][i],3)) + '+-' + str(round(Performances[name_metric + '_folds_sd_all'][i],3)) + '+-' + str(round(Performances[name_metric + '_sd_all'][i],3))
    
    #Convert float to int for sample sizes and some metrics.
    for name_col in Performances.columns.values:
        if name_col.startswith('N_') | any(metric in name_col for metric in metrics_displayed_in_int) & (not '_sd' in name_col) & (not '_str' in name_col):
            Performances[name_col] = Performances[name_col].astype('Int64') #need recent version of pandas to use this type. Otherwise nan cannot be int
    
    #Merging the results of the previous Performances table with the new one
    try:
        Performances_old_and_new = pd.concat([Performances_previous,Performances], axis=0)
    except:
        Performances_old_and_new = Performances
        print('No previous Performances table to to merge with.')
    
    #Ranking, printing and saving
    Performances_alphabetical = Performances_old_and_new.sort_values(by='version')
    print('Performances of the models ranked by models\'names:')
    print(Performances_alphabetical)
    Performances_sorted = Performances_old_and_new.sort_values(by=dict_main_metrics_names[target] + '_all', ascending=main_metrics_modes[dict_main_metrics_names[target]] == 'min')
    print('Performances of the models ranked by validation score on the main metric:')
    print(Performances_sorted)
    if save_performances:
        Performances_alphabetical.to_csv(path_store + 'Performances_alphabetical_' + target + '_' + fold + '_' + id_set + '.csv', index=False)
        Performances_sorted.to_csv(path_store + 'Performances_ranked_' + target + '_' + fold + '_' + id_set + '.csv', index=False)
    
    print('Done calculating the performances.')
    print('Now reformating the performances for each model as a separated dataframe...')
    
    #Define an empty performances dataframe that will serve for each model
    row_names = ['all'] + outer_folds
    col_names = dict_metrics_names[dict_prediction_types[target]]
    performances = np.empty((len(row_names),len(col_names),))
    performances.fill(np.nan)
    performances = pd.DataFrame(performances)
    performances.index = row_names
    performances.columns = col_names

    #For each model, fill the dataframe with the data contained in the central dataframe
    for i, model in enumerate(list_models):
        for info in ['', '_sd', '_str']:
            performances_fold = performances
            for metric in dict_metrics_names[dict_prediction_types[target]]:
                for outer_fold in ['all'] + outer_folds:
                    performances_fold.loc[outer_fold, metric] = Performances[metric+ info + '_' + outer_fold ][i].values[0]
            print('Performance for model ' + model)
            print(performances_fold)
        
print('Done.')

