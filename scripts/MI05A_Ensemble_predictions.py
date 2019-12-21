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
    sys.argv.append('val') #inner fold
    

for id_set in id_sets:
    Predictions = pd.read_csv(path_store + 'Predictions_' + target + '_' + fold + '_' + id_set + '.csv')
    Performances = pd.read_csv(path_store + 'Performances_alphabetical_' + target + '_' + fold + '_' + id_set + '.csv')


#which ensemble models to create?
#best models altogether no discrimination
#best models from a specific organ
#
    
    