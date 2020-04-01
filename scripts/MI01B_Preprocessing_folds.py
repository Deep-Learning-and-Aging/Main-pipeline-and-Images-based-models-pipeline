#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:58:19 2019

@author: Alan
"""

from MI_Libraries import *
from MI_Classes import PreprocessingFolds

# Default parameters
if len(sys.argv) != 3:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('ArterialStiffness_4205')  # image_field, e.g 'PhysicalActivity_90001' or 'Heart_20208'

# Compute results
Preprocessing_Folds = PreprocessingFolds(target=sys.argv[1], image_field=sys.argv[2])
Preprocessing_Folds.generate_folds()

# Exit
print('Done.')
sys.exit(0)
