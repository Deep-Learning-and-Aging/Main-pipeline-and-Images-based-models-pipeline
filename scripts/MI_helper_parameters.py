#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 00:18:17 2019

@author: Alan
"""

from MI_helper_functions import *

#parameters to put in helper file
folds = ['train', 'val', 'test']
folds_tune = ['train', 'val']
models_names = ['VGG16', 'VGG19', 'MobileNet', 'MobileNetV2', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'NASNetMobile', 'NASNetLarge', 'Xception', 'InceptionV3', 'InceptionResNetV2']
images_sizes = ['224', '299', '331']
dict_prediction_types={'Age':'regression', 'Sex':'binary'}
dict_activations={'regression':'linear', 'binary':'sigmoid', 'multiclass':'softmax', 'saliency':'linear'}
dict_losses={'regression':'mean_squared_error', 'binary':'binary_crossentropy', 'multiclass':'categorical_crossentropy'}
dict_metrics={'regression':['R-squared', 'RMSE'], 'binary':['AUC', 'Binary-Accuracy'], 'multiclass':['Categorical-Accuracy']}
dict_metrics_functions_names={'R-squared':'R_squared', 'RMSE':'root_mean_squared_error', 'AUC':'auc', 'Binary-Accuracy':'binary_accuracy', 'Categorical-Accuracy':'categorical_accuracy'}
dict_dir_images ={'Liver':'/n/groups/patel/uk_biobank/main_data_52887/Liver/Liver_20204/'}
main_metrics = dict.fromkeys(['regression'], 'R_squared')
main_metrics.update(dict.fromkeys(['binary'], 'AUC'))
#dict_metrics_functions_names={'R-squared':'R_squared', 'RMSE':'root_mean_squared_error', 'AUC':'auc', 'Binary-Accuracy':'binary_accuracy', 'Categorical-Accuracy':'categorical_accuracy'}
dict_metrics_functions={'R-squared':R_squared, 'RMSE':root_mean_squared_error, 'AUC':auc, 'Binary-Accuracy':'binary_accuracy', 'Categorical-Accuracy':'categorical_accuracy'}


#metric_functions = {'R_squared':r2_score, 'root_mean_squared_error':rmse}
image_quality_ids = {'Liver':'22414-2.0'}
targets_regression = ['Age']
targets_binary = ['Sex']
regenerate_data_features = False

#define dictionary to resize the images to the right size depending on the model
input_size_models = dict.fromkeys(['VGG16', 'VGG19', 'MobileNet', 'MobileNetV2', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'NASNetMobile'], 224)
input_size_models.update(dict.fromkeys(['Xception', 'InceptionV3', 'InceptionResNetV2'], 299))
input_size_models.update(dict.fromkeys(['NASNetLarge'], 331))

#define dictionaries to format the text
dict_folds={'train':'Training', 'val':'Validation', 'test':'Testing'}

#define paths
if '/Users/Alan/' in os.getcwd():
    os.chdir('/Users/Alan/Desktop/Aging/Medical_Images/scripts/')
    path_store = '../data/'
    path_compute = '../data/'
else:
    os.chdir('/n/groups/patel/Alan/Aging/Medical_Images/scripts/')
    path_store = '../data/'
    path_compute = '/n/scratch2/al311/Aging/Medical_Images/data/'

#model
import_weights = 'imagenet' #choose between None and 'imagenet'

#compiler
batch_size = 32
n_epochs_max = 1000
debunk_mode = False
continue_training = False

#postprocessing
boot_iterations=10000

#set parameters
seed=0
random.seed(seed)
set_random_seed(seed)
