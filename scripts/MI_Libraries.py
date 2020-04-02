# LIBRARIES
# read and write
import os
import sys
import glob
import re
import fnmatch

# maths
import numpy as np
import pandas as pd
import math
import random

# miscellaneous
import warnings
import gc

# sklearn
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, log_loss, roc_auc_score, accuracy_score, f1_score, \
    precision_score, recall_score, confusion_matrix, average_precision_score
from sklearn.utils import resample

# GPUs
from GPUtil import GPUtil
# tensorflow
import tensorflow as tf
from tensorflow import set_random_seed
# keras
from keras import backend as k
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras import regularizers
from keras.optimizers import Adam, RMSprop, Adadelta
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger

# Model's attention
import innvestigate
from vis.utils import utils
from vis.visualization import visualize_cam

# plot figures
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # allows to generate and save figures with ssh -x11
import matplotlib.pyplot as plt
