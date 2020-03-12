#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:58:19 2019

@author: Alan
"""

#load libraries, import functions and import parameters (nested import in the line below)
from MI_helpers import *

#default parameters
if len(sys.argv) != 3:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age') #target
    sys.argv.append('ArterialStiffness_4205') #image_field, e.g 'PhysicalActivity_90001', 'Liver_20204' or 'Heart_20208'

#read parameters from command
target = sys.argv[1]
image_field = sys.argv[2]
organ, field_id = image_field.split('_')

#configure cpus
n_cpus = len(os.sched_getaffinity(0))

#get the list of the ids available for the field_id
if field_id in images_field_ids:
    list_available_ids = [e.replace('.jpg', '') for e in os.listdir(dict_default_dir_images[image_field])]
else:
    list_available_ids = pd.read_csv(path_store + 'IDs_' + field_id + '.csv')
    list_available_ids = list_available_ids.values.squeeze().astype(str)

#generate data_features
generate_data_features(image_field=image_field, organ=organ, field_id=field_id, target=target, list_available_ids=list_available_ids, image_quality_id=image_quality_ids[organ])

#exit
print('Done.')
sys.exit(0)
