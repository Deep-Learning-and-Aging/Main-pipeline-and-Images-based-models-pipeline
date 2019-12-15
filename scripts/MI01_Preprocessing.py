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
    sys.argv.append('Heart_20208') #image_field, e.g 'PhysicalActivity_90001'

#read parameters from command
target = sys.argv[1]
image_field = sys.argv[2]
organ, field_id = image_field.split('_')

#configure cpus
n_cpus = len(os.sched_getaffinity(0))

#generate data_features
DATA_FEATURES = generate_data_features(image_field=image_field, organ=organ, target=target, dir_images=dict_default_dir_images[image_field], image_quality_id=image_quality_ids[organ])

