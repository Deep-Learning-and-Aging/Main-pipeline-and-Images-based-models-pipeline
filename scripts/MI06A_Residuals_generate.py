#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:24:04 2020

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

Predictions = pd.read_csv(path_store + 'PREDICTIONS_withEnsembles_' + target + '_' + fold + '_' + id_set + '.csv')
Residuals = Predictions[['eid', 'Sex', 'Age']]
list_models = [col_name.replace('pred_', '') for col_name in Predictions.columns.values if 'pred_' in col_name]
for model in list_models:
    print('Generating residuals for model ' + model)
    df_model = Predictions[['eid', 'Age', 'pred_' + model, 'outer_fold_' + model]].dropna(subset=['eid', 'Age', 'pred_' + model])
    age = df_model.loc[:, ['Age']]
    res = df_model['pred_' + model] - df_model['Age']
    regr = linear_model.LinearRegression()
    regr.fit(age, res)
    res_correction = regr.predict(age)
    res_corrected = res - res_correction
    if debug_mode:
        print('Bias for the residuals ' + model, regr.coef_)
        pyplot.scatter(age, res)
        pyplot.scatter(age, res_corrected)
        regr2 = linear_model.LinearRegression()
        regr2.fit(age, res_corrected)
        print('Coefficients after: \n', regr2.coef_)
    df_model['res_' + model] = res_corrected
    df_model = df_model.drop(columns=['Age', 'pred_' + model])
    Residuals = Residuals.merge(df_model, how='outer', on=['eid'])
#save the residuals
Residuals.to_csv(path_store + 'RESIDUALS_' + target + '_' + fold + '_' + id_set + '.csv', index=False)

#exit
print('Done.')
sys.exit(0)

