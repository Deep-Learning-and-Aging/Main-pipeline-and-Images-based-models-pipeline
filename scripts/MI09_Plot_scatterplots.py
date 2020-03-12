#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 23:14:16 2020

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

#Load the predictions
PREDICTIONS={}
for fold in folds:
    PREDICTIONS[fold] = pd.read_csv(path_store + 'PREDICTIONS_withEnsembles_' + target + '_' + fold + '_' + id_set + '.csv')

#print scatter plots for each model
list_versions = [col_name.replace('pred_', '') for col_name in PREDICTIONS['test'].columns.values if 'pred_' in col_name]
for version in list_versions[:1]:
    #concatenate the predictions, format the data before plotting
    for fold in folds:
        df_version = PREDICTIONS[fold][[target, 'pred_' + version, 'outer_fold_' + version]]
        df_version.dropna(inplace=True)
        df_version.rename(columns={'pred_' + version: 'Prediction', 'outer_fold_' + version: 'outer_fold'}, inplace = True)
        df_version['outer_fold'] = df_version['outer_fold'].astype(int).astype(str)
        df_version['set'] = dict_folds_names[fold]
        if fold == 'train':
            df_allsets = df_version
        else:
            df_allsets = df_allsets.append(df_version)
    
    #generate the plots and save them
    single_plot = sns.lmplot( x=target, y='Prediction', data=df_version, fit_reg=False, hue='outer_fold', scatter_kws={'alpha':0.3})
    single_plot.savefig('../figures/ScatterPlot_' + version + '.png')
    multi_plot = sns.FacetGrid(df_allsets, col='set', hue='outer_fold')
    multi_plot.map(plt.scatter, 'Age', 'Prediction', alpha=.1)
    multi_plot.add_legend();
    multi_plot.savefig('../figures/Scatter_Plots/ScatterPlots_' + version + '.png')
    
    