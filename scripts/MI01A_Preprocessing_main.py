#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 09:59:15 2020

@author: Alan
"""

from MI_Libraries import *
from MI_Classes import PreprocessingMain

# Compute results
Preprocessing_Main = PreprocessingMain()
Preprocessing_Main.generate_data()

# Exit
print('Done.')
sys.exit(0)
