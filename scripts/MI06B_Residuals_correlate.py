#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:25:21 2020

@author: Alan
"""

#load libraries, import functions and import parameters (nested import in the line below)
from MI_helpers import *

#default parameters
if len(sys.argv) != 3:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age') #target
    sys.argv.append('test')
    sys.argv.append('B') #id_set

#read parameters from command
target = sys.argv[1]
fold = sys.argv[2]
id_set = sys.argv[3]

#options
debug_mode = False
if debug_mode:
    n_bootstrap_iterations = 10

#Load the residuals
Residuals = pd.read_csv(path_store + 'RESIDUALS_' + target + '_' + fold + '_' + id_set + '.csv')

#Format the dataframe
Residuals_only = Residuals[[col_name for col_name in Residuals.columns.values if 'res_' in col_name]]
Residuals_only.rename(columns=lambda x: x.replace('res_' + target + '_',''), inplace=True)
#Generate the correlation matrix
Residuals_correlations = Residuals_only.corr()
#Gerate the std by boostrapping
_, Residuals_correlations_sd = bootstrap_correlations(Residuals_only, n_bootstrap_iterations)
#Merge both as a dataframe of strings
Residuals_correlations_str = Residuals_correlations.round(3).applymap(str) + '+-' + Residuals_correlations_std.round(3).applymap(str)

#save the correlations
for mode in modes:
    globals()['Residuals_correlations' + mode].to_csv(path_store + 'ResidualsCorrelations' + mode + '_' + target + '_' + fold + '_' + id_set + '.csv', index=True)

#exit
print('Done.')
sys.exit(0)

