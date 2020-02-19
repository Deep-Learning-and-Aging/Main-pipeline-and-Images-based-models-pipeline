#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:25:54 2020

@author: Alan
"""

from MI_helpers import *

#list of environmental variables
#list of diseases
#list of biomarkers

id_set = 'B'

if id_set == 'A': #different set of eids
    data_features = pd.read_csv("/n/groups/patel/uk_biobank/main_data_9512/data_features.csv")
    data_features.replace({'f.31.0.0': {'Male': 0, 'Female': 1}}, inplace=True)
else:
    data_features = pd.read_csv('/n/groups/patel/uk_biobank/main_data_52887/ukb37397.csv')