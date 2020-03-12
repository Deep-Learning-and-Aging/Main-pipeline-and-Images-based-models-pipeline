#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 09:59:15 2020

@author: Alan
"""

from MI_helpers import *

#extract the relevant columns from the main UKB dataset
data_features = pd.read_csv('/n/groups/patel/uk_biobank/main_data_52887/ukb37397.csv', usecols=['eid', '31-0.0', '21003-0.0', '21003-2.0', '22414-2.0'])
data_features.rename(columns=dict_UKB_fields_to_names, inplace=True)
data_features['eid'] = data_features['eid'].astype(str)
data_features = data_features.set_index('eid', drop=False)
data_features.to_csv(path_store + 'data-features.csv', index=False)

print('Done.')
sys.exit(0)

