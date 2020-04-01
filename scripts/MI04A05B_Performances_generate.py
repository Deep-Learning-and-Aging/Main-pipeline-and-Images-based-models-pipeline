#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 15:40:44 2019

@author: Alan
"""

from MI_Libraries import *
from MI_Classes import PerformancesGenerate

# Options
# Use a small number for the bootstrapping
debug_mode = False

# Default parameters
# if len(sys.argv) != 10:
#    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
#    sys.argv = ['']
#    sys.argv.append('Age') #target
#    sys.argv.append('Heart_20208_3chambers') #image_type, e.g PhysicalActivity_90001_main, or Heart_20208_3chambers
#    sys.argv.append('raw') #transformation
#    sys.argv.append('InceptionResNetV2') #architecture
#    sys.argv.append('Adam') #optimizer
#    sys.argv.append('0.000001') #learning_rate
#    sys.argv.append('0.0') #weight decay
#    sys.argv.append('0.0') #dropout
#    sys.argv.append('val') #fold

# Default parameters for ensemble models
if len(sys.argv) != 10:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('*_*_*')  # image_type, e.g PhysicalActivity_90001_main, Liver_20204_main or Heart_20208_3chambers
    sys.argv.append('*')  # transformation
    sys.argv.append('*')  # architecture
    sys.argv.append('*')  # optimizer
    sys.argv.append('*')  # learning_rate
    sys.argv.append('*')  # weight decay
    sys.argv.append('*')  # dropout
    sys.argv.append('val')  # fold

# Compute results
Performances_Generate = PerformancesGenerate(target=sys.argv[1], image_type=sys.argv[2], transformation=sys.argv[3],
                                             architecture=sys.argv[4], optimizer=sys.argv[5], learning_rate=sys.argv[6],
                                             weight_decay=sys.argv[7], dropout_rate=sys.argv[8], fold=sys.argv[9],
                                             debug_mode=False)
Performances_Generate.preprocessing()
Performances_Generate.compute_performances()
Performances_Generate.save_performances()

# Exit
print('Done.')
sys.exit(0)
