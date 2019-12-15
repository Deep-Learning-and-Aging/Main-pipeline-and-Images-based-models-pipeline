#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:52:20 2019

@author: Alan
"""

###load libraries

#read and write
import os
import sys
import json
import pickle
import tarfile
import shutil
import pyreadr
import csv

#maths
import numpy as np
import pandas as pd
import math
import random
from math import sqrt
from numpy.polynomial.polynomial import polyfit

#images
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import shift, rotate
from skimage.color import gray2rgb

#miscellaneous
import warnings
import multiprocessing as mp
from tqdm import tqdm_notebook as tqdm
import gc

#tensorflow
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow import set_random_seed

#sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.utils import class_weight

#keras
import keras
from keras import backend as K
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, Activation, Input, Reshape, BatchNormalization, InputLayer, Dropout, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Conv3D, MaxPooling3D, GlobalAveragePooling2D, LSTM
from keras.models import Sequential, Model, model_from_json, clone_model
from keras import regularizers, optimizers
from keras.optimizers import Adam, RMSprop, Adadelta
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, TensorBoard, ModelCheckpoint
from sklearn.metrics import r2_score, mean_squared_error
#from tensorflow.keras.metrics import Recall, Precision, AUC




