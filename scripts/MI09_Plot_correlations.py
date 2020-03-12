#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:21:38 2020

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

#options
save_figures = True

#Load data
Correlations = pd.read_csv(path_store + 'ResidualsCorrelations' + '_' + target + '_' + id_set + '.csv', index_col='Unnamed: 0')

#Crop the names to make the reading of the labels easier
idx_to_print = [names_model_parameters[1:].index(i) for i in ['organ', 'view','architecture']]
Correlations.index = ['_'.join(np.array(idx.split('_'))[idx_to_print]) for idx in Correlations.index.values]
Correlations.columns = ['_'.join(np.array(idx.split('_'))[idx_to_print]) for idx in Correlations.columns.values]

#Plot the figure
plot_correlations(data = Correlations, save_figure = save_figures, title_save = 'Correlations_AllModels_' + target + '_' + id_set)

#Plot the "ensemble models only" correlation plots
for ensemble_type in ensemble_types:
    index_ensembles_only = [idx for idx in Correlations.columns.values if ensemble_type in idx]
    Correlations_Ensembles_only = Correlations.loc[index_ensembles_only, index_ensembles_only]
    plot_correlations(data = Correlations_Ensembles_only, save_figure = True, title_save = 'Correlations_Ensembles' + ensemble_type + 'Only_' + target + '_' + id_set)
