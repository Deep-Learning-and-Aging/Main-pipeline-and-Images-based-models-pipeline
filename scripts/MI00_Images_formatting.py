#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 10:09:33 2019

@author: Alan
"""

"""For Heart_20208, merge the 3 channels (2 chambers, 3 chambers and 4 chambers)
into a single "RGB" picture. Do this for both raw and contrasted images."""

import sys
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imsave
import glob

#default parameters
if len(sys.argv) != 2:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('raw') #transformation

#read parameters from command
transformation = sys.argv[1]

#For Heart, concatenate the 3 views (2chambers, 3chambers, 4chambers) into a single "RGB" view: allchambersRGB
print('Saving the images with the transformation: ' + transformation)
list_images = glob.glob('../images/Heart/20208/' + '2chambers' + '/' + transformation + '/*.jpg')
list_images = [image.replace('../images/Heart/20208/' + '2chambers' + '/' + transformation + '/', '') for image in list_images]
print(str(len(list_images)) + ' images to save.')
for i, image in enumerate(list_images):
    image_arr = []
    for view in ['2chambers', '3chambers', '4chambers']:
        path_view = '../images/Heart/20208/' + view + '/' + transformation + '/' + image
        image_arr.append(imread(path_view))
    array_3d = np.moveaxis(np.array(image_arr), 0, -1)
    imsave('../images/Heart/20208/' + 'allviewsRGB' + '/' + transformation + '/' + image, array_3d)
    if(i%1000 == 0):
        print('Saved ' + str(i+1) + ' image(s).')

print('Done.')
sys.exit(0)
