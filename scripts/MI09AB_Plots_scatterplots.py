#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 23:14:16 2020

@author: Alan
"""

from MI_Libraries import *
from MI_Classes import PlotsScatter

# Default parameters
if len(sys.argv) != 3:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target

# Generate results
Plots_Scatter = PlotsScatter(target=sys.argv[1])
Plots_Scatter.generate_plots()

# Exit
print('Done.')
sys.exit(0)
