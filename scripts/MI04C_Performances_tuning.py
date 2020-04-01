#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 21:00:53 2020

@author: Alan
"""

from MI_Libraries import *
from MI_Classes import PerformancesTuning

# Default parameters
if len(sys.argv) != 2:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target

# Compute results
Performances_Tuning = PerformancesTuning(target=sys.argv[1])
Performances_Tuning.load_data()
Performances_Tuning.preprocess_data()
Performances_Tuning.select_models()
Performances_Tuning.save_performances()

# Exit
print('Done.')
sys.exit(0)
