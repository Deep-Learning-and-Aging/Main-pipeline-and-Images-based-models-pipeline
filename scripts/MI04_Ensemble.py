#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 18:44:24 2019

@author: Alan
"""

#Select all models matching the parameters requested
list_model_weights = glob.glob(path_store + 'model-weights_' + version + '*.h5')
list_model_weights.sort()
list_models_available = [os.path.splitext(model_name)[0].split('_')[1:] for model_name in list_model_weights]
list_versions = ['_'.join(parameters_list) for parameters_list in list_models_available]

#take subset of models to explore based on conditions in the input
for parameter in ['architecture', 'learning_rate', 'weight_decay', 'dropout_rate']:
    parameter_value = globals()[parameter]
    if parameter_value != '*':
        Performances = Performances[Performances[parameter] == parameter_value]
        
#load the architecture
list_architectures = Performances['architecture'].unique()

Performances_architecture = Performances[Performances['architecture'] == architecture]

    #Record architecture's results in general dataframe before printing them
    Performances[Performances['architecture'] == architecture] = Performances_architecture
    print('Completed evaluation for architecture ' + architecture + ". Results below:")
    print(Performances_architecture)





    


#Ranking, printing and saving
print('Performances of the models ranked by models\'names:')
print(Performances)
Performances_sorted = Performances.sort_values(by=main_metric_name + '_val', ascending=main_metrics_modes[main_metric_name] == 'min')
print('Performances of the models ranked by validation score on the main metric:')
print(Performances_sorted)
if save_performances:
    Performances.to_csv(path_store + 'Performances_alphabetical_' + version + '.csv', index=False)
    Performances_sorted.to_csv(path_store + 'Performances_ranked_' + version + '.csv', index=False)