#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 18:53:37 2019

@author: Alan
"""

#Define Performances dataframe
Performances = pd.DataFrame.from_records(list_models_available)
Performances.columns = ['target', 'organ', 'field_id', 'view', 'preprocessing', 'architecture', 'optimizer', 'learning_rate', 'weight_decay', 'dropout_rate']
Performances[['learning_rate', 'weight_decay', 'dropout_rate']] = Performances[['learning_rate', 'weight_decay', 'dropout_rate']].apply(pd.to_numeric)
Performances['backup_used'] = False

#add columns for metrics
for metric_name in metrics_names:
    for fold in folds:
        Performances[metric_name + '_' + fold] = np.nan






        #for each fold and for each metric, compute the model's performance
        for fold in folds:
            pred=model.predict_generator(GENERATORS[fold], steps=STEP_SIZES[fold], verbose=1).squeeze()
            try:
                PREDICTIONS[fold][model_version] = pred
                #convert to pred class?
                for metric_name in metrics_names:
                    Performances_architecture.loc[i, metric_name + '_' + fold] = dict_metrics[metric_name][functions_version](Ys[fold], pred)
            except:
                print("Mismatch between length of pred and y")
           
            
            
                
Performances_architecture.loc[i, 'backup_used'] = True
