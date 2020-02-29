#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 18:59:15 2019

@author: Alan
"""

###LIBRARIES

#read and write
import os
import sys
import glob
import csv
import json
import pickle
import tarfile
import shutil
import pyreadr
import fnmatch
import re

#maths
import numpy as np
import pandas as pd
import math
import random
from math import sqrt
from numpy.polynomial.polynomial import polyfit

#images
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy.ndimage import shift, rotate
from skimage.color import gray2rgb

#miscellaneous
import warnings
import multiprocessing as mp
from tqdm import tqdm_notebook as tqdm
import gc
import GPUtil
from datetime import datetime

#sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, log_loss, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.utils import class_weight, resample
from sklearn import linear_model

#tensorflow
import tensorflow as tf
from tensorflow import set_random_seed

#keras
import keras
from keras import backend as K
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, Activation, Input, Reshape, BatchNormalization, InputLayer, Dropout, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Conv3D, MaxPooling3D, GlobalAveragePooling2D, LSTM
from keras.models import Sequential, Model, model_from_json, clone_model
from keras import regularizers, optimizers
from keras.optimizers import Adam, RMSprop, Adadelta
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger, TerminateOnNaN, TensorBoard
from keras.constraints import max_norm


### PARAMETERS
names_model_parameters = ['target', 'organ', 'field_id', 'view', 'transformation', 'architecture', 'optimizer', 'learning_rate', 'weight_decay', 'dropout_rate']
dict_prediction_types={'Age':'regression', 'Sex':'binary'}
dict_losses_names={'regression':'MSE', 'binary':'Binary-Crossentropy', 'multiclass':'categorical_crossentropy'}
dict_main_metrics_names={'Age':'R-Squared', 'Sex':'ROC-AUC', 'imbalanced_binary_placeholder':'F1-Score'}
main_metrics_modes={'loss':'min', 'R-Squared':'max', 'ROC-AUC':'max'}
dict_metrics_names={'regression':['RMSE', 'R-Squared'], 'binary':['ROC-AUC', 'F1-Score', 'PR-AUC', 'Binary-Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'Recall', 'True-Positives', 'False-Positives', 'False-Negatives', 'True-Negatives'], 'multiclass':['Categorical-Accuracy']}
dict_activations={'regression':'linear', 'binary':'sigmoid', 'multiclass':'softmax', 'saliency':'linear'}
folds = ['train', 'val', 'test']
folds_tune = ['train', 'val']
models_names = ['VGG16', 'VGG19', 'MobileNet', 'MobileNetV2', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'NASNetMobile', 'NASNetLarge', 'Xception', 'InceptionV3', 'InceptionResNetV2']
images_sizes = ['224', '299', '331']
targets_regression = ['Age']
targets_binary = ['Sex']
image_quality_ids = {'Liver':'22414-2.0'}
image_quality_ids .update(dict.fromkeys(['Heart', 'Brain', 'DXA', 'Pancreas', 'Carotid', 'ECG', 'ArterialStiffness'], None))

id_sets = ['A', 'B']
dict_idset_to_organ={'A':['Biomarkers', 'ArterialStiffness', 'ECG', 'EyeFundus', 'PhysicalActivity'], 'B':['Liver', 'Heart', 'Brain', 'DXA', 'Pancreas', 'Carotid']}
dict_organ_to_idset = dict.fromkeys(['Biomarkers', 'ArterialStiffness', 'ECG', 'EyeFundus', 'PhysicalActivity'], 'A')
dict_organ_to_idset.update(dict.fromkeys(['Liver', 'Heart', 'Brain', 'DXA', 'Pancreas', 'Carotid'], 'B'))
dict_field_id_to_age_instance = dict.fromkeys(['Placeholder', '6025', '4205'], 'Age_Assessment')
dict_field_id_to_age_instance.update(dict.fromkeys(['20204', '20208', '20205'], 'Age_Imaging'))
dict_field_id_to_age_instance.update(dict.fromkeys(['90001'], 'Age_Accelerometer'))

metrics_needing_classpred = ['F1-Score', 'Binary-Accuracy', 'Precision', 'Recall']
metrics_displayed_in_int = ['True-Positives', 'True-Negatives', 'False-Positives', 'False-Negatives']
modes = ['', '_sd', '_str']

#define dictionary of batch sizes to fit as many samples as the model's architecture allows
dict_batch_sizes = dict.fromkeys(['NASNetMobile'], 128)
dict_batch_sizes.update(dict.fromkeys(['MobileNet', 'MobileNetV2'], 64))
dict_batch_sizes.update(dict.fromkeys(['InceptionV3', 'VGG16', 'VGG19', 'DenseNet121', 'DenseNet169'], 32))
dict_batch_sizes.update(dict.fromkeys(['DenseNet201', 'InceptionResNetV2', 'Xception'], 16))
dict_batch_sizes.update(dict.fromkeys(['NASNetLarge'], 4))

#define dictionaries to format the text
dict_folds_names={'train':'Training', 'val':'Validation', 'test':'Testing'}

#define dictionary to resize the images to the right size depending on the model
input_size_models = dict.fromkeys(['VGG16', 'VGG19', 'MobileNet', 'MobileNetV2', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'NASNetMobile'], 224)
input_size_models.update(dict.fromkeys(['Xception', 'InceptionV3', 'InceptionResNetV2'], 299))
input_size_models.update(dict.fromkeys(['NASNetLarge'], 331))

#dictionary of dir_images used to generate the IDs split during preprocessing
dict_default_dir_images={}
dict_default_dir_images['Liver_20204'] = '../images/Liver/20204/main/raw/'
dict_default_dir_images['Heart_20208'] = '../images/Heart/20208/4chambers/raw/'
dict_default_dir_images['PhysicalActivity_90001'] = '../images/PhysicalActivity/90001/main/raw/'

#dict to choose the IDs mapping, depending on the UKB research proposal application that provided the data
dict_eids_version = dict.fromkeys(['PhysicalActivity'], 'A')
dict_eids_version.update(dict.fromkeys(['Liver', 'Heart'], 'B'))

#dict to decide which field is used to generate the ids when several organs/fields share the same ids (e.g Liver_20204 and Heart_20208)
dict_image_field_to_ids = dict.fromkeys(['PhysicalActivity_90001'], 'PhysicalActivity_90001')
dict_image_field_to_ids.update(dict.fromkeys(['Liver_20204', 'Heart_20208'], 'Liver_20204'))
dict_image_field_to_ids.update(dict.fromkeys(['Heart_20208'], 'Heart_20208'))

#dict to decide which field is used to generate the ids when several targets share the same ids (e.g Age and Sex)
dict_target_to_ids = dict.fromkeys(['Age', 'Sex'], 'Age')

#dict to decide in which order targets should be used when trying to transfer weight from a similar model
dict_alternative_targets_for_transfer_learning={'Age':['Age', 'Sex'], 'Sex':['Sex', 'Age']}


#define paths
if '/Users/Alan/' in os.getcwd():
    os.chdir('/Users/Alan/Desktop/Aging/Medical_Images/scripts/')
    path_store = '../data/'
    path_compute = '../data/'
else:
    os.chdir('/n/groups/patel/Alan/Aging/Medical_Images/scripts/')
    path_store = '../data/'
    path_compute = '/n/scratch2/al311/Aging/Medical_Images/data/'

#preprocessing
n_CV_outer_folds = 10
outer_folds = [str(x) for x in list(range(n_CV_outer_folds))]

#compiler
n_epochs_max = 1000

#architecture
keras_max_norm = 4

#ensemble models
ensembles_performance_cutoff_percent = 0

#postprocessing
n_bootstrap_iterations = 1000

#set parameters
seed=0
random.seed(seed)
set_random_seed(seed)

#garbage collector
gc.enable()


### HELPER FUNCTIONS

def read_parameters_from_command(args):
    parameters={}
    parameters['target'] = args[1]
    parameters['image_type'] = args[2]
    parameters['transformation'] = args[3]
    parameters['architecture'] = args[4]
    parameters['optimizer'] = args[5]
    parameters['learning_rate'] = args[6]
    parameters['weight_decay'] = args[7]
    parameters['dropout_rate'] = args[8]
    if len(args) > 10:
        parameters['outer_fold'] = args[9]
        parameters['id_set'] = args[10]
    elif len(args) > 9:
        parameters['outer_fold'] = args[9]
        parameters['id_set'] = None
    else:
        parameters['outer_fold'] = None
        parameters['id_set'] = None
    
    parameters['organ'], parameters['field_id'], parameters['view'] = parameters['image_type'].split('_')
    #convert parameters to float if a specific value other than 'all' was selected
    for parameter_name in ['learning_rate', 'weight_decay', 'dropout_rate']:
        if(parameters[parameter_name] != '*'):
            parameters[parameter_name] = float(parameters[parameter_name])
    return parameters['target'], parameters['image_type'], parameters['organ'], parameters['field_id'], parameters['view'], parameters['transformation'], parameters['architecture'], parameters['optimizer'], parameters['learning_rate'], parameters['weight_decay'], parameters['dropout_rate'], parameters['outer_fold'], parameters['id_set']

def version_to_parameters(model_name, names_model_parameters):
    parameters={}
    parameters_list = model_name.split('_')
    for i, parameter in enumerate(names_model_parameters):
        parameters[parameter] = parameters_list[i]
    if len(parameters_list) > 10:
        parameters['outer_fold'] = parameters_list[10]
    return parameters

def parameters_to_version(parameters):
    return '_'.join(parameters.values())

def convert_string_to_boolean(string):
    if string == 'True':
        boolean = True
    elif string == 'False':
        boolean = False
    else:
        print('ERROR: string must be either \'True\' or \'False\'')
        sys.exit(1)
    return boolean

def configure_gpus():
   print('tensorflow version : ', tf.__version__)
   print('Build with Cuda : ', tf.test.is_built_with_cuda())
   print('Gpu available : ', tf.test.is_gpu_available())
   #print('Available ressources : ', tf.config.experimental.list_physical_devices())
   #device_count = {'GPU': 1, 'CPU': mp.cpu_count() },log_device_placement =  True)
   config = tf.ConfigProto()
   config.gpu_options.allow_growth = True
   gpu_session = tf.Session(config = config)
   K.set_session(session= gpu_session)
   K.tensorflow_backend._get_available_gpus()
   warnings.filterwarnings('ignore')
   return gpu_session

def append_ext(fn):
    return fn+".jpg"

def load_data_features(path_store, image_field, target, folds, outer_fold, images_ext):
    DATA_FEATURES = {}
    for fold in folds:
        DATA_FEATURES[fold] = pd.read_csv(path_store + 'data-features_' + image_field + '_' + target + '_' + fold + '_' + outer_fold + '.csv')
        if images_ext:
            DATA_FEATURES[fold]['eid'] = DATA_FEATURES[fold]['eid'].astype(str).apply(append_ext)
            DATA_FEATURES[fold] = DATA_FEATURES[fold].set_index('eid', drop=False)
    return DATA_FEATURES

def generate_data_features(image_field, organ, field_id, target, dir_images, image_quality_id):
    cols_data_features = ['eid', 'Sex', dict_field_id_to_age_instance[field_id]]
    dict_rename_cols = {dict_field_id_to_age_instance[field_id]: 'Age'}
    if dict_organ_idset[organ] == 'A':
            data_features = pd.read_csv("/n/groups/patel/uk_biobank/main_data_9512/data_features.csv")[['f.eid', 'f.31.0.0', 'f.21003.0.0']]
            data_features.replace({'f.31.0.0': {'Male': 0, 'Female': 1}}, inplace=True)
            print('THIS IS PROBABLY NOT CORRECT. Implement if new organ and make sure age matches')
    else:
        if image_quality_id != None:
            col_data_features.append(image_quality_id)
            dict_rename_cols[organ + '_images_quality'] = 'Data_quality'
        data_features = pd.read_csv(path_store + 'data-features.csv', usecols = cols_data_features)
        data_features.rename(columns={dict_rename_cols}, inplace=True)
        data_features['eid'] = data_features['eid'].astype(str).apply(append_ext)
        data_features = data_features.set_index('eid', drop=False)
    if image_quality_id != None:
        data_features = data_features[data_features['Data_quality'] != np.nan]
        data_features = data_features.drop('Data_quality', axis=1)
    # get rid of samples with NAs
    data_features.dropna(inplace=True)
    # list the samples' ids for which liver images are available
    all_files = os.listdir(dir_images)
    data_features = data_features.loc[all_files]
    #files = data_features.index.values
    ids = data_features.index.values.copy()
    np.random.shuffle(ids)
    #distribute the ids between the different outer and inner folds
    n_samples = len(ids)
    n_samples_by_fold = n_samples/n_CV_outer_folds
    FOLDS_IDS = {}
    for outer_fold in outer_folds:
        FOLDS_IDS[outer_fold] = np.ndarray.tolist(ids[int((int(outer_fold))*n_samples_by_fold):int((int(outer_fold)+1)*n_samples_by_fold)])
    TRAINING_IDS={}
    VALIDATION_IDS={}
    TEST_IDS={}
    for i in outer_folds:
        TRAINING_IDS[i] = []
        VALIDATION_IDS[i] = []
        TEST_IDS[i] = []
        for j in outer_folds:
            if (j == i):
                VALIDATION_IDS[i].extend(FOLDS_IDS[j])
            elif (int(j) == ((int(i)+1)%n_CV_outer_folds)):
                TEST_IDS[i].extend(FOLDS_IDS[j])
            else:
                TRAINING_IDS[i].extend(FOLDS_IDS[j])
    IDS = {'train':TRAINING_IDS, 'val':VALIDATION_IDS, 'test':TEST_IDS}
    #generate inner fold split for each outer fold
    for outer_fold in outer_folds:
        print(outer_fold)
        # compute values for scaling of regression targets
        if target in targets_regression:
            idx = np.where(np.isin(data_features.index.values, TRAINING_IDS[outer_fold]))[0]
            data_features_train = data_features.iloc[idx, :]
            target_mean = data_features_train[target].mean()
            target_std = data_features_train[target].std()
        #generate folds
        indices = {}
        for fold in folds:
            indices[fold] = np.where(np.isin(data_features.index.values, IDS[fold][outer_fold]))[0]
            data_features_fold = data_features.iloc[indices[fold], :]
            data_features_fold['outer_fold'] = outer_fold
            data_features_fold = data_features_fold[['eid', 'outer_fold', 'Sex', 'Age']]
            if target in targets_regression:
                data_features_fold[target + '_raw'] = data_features_fold[target]
                data_features_fold[target] = (data_features_fold[target] - target_mean) / target_std
            data_features_fold.to_csv(path_store + 'data-features_' + image_field + '_' + target + '_' + fold + '_' + outer_fold + '.csv', index=False)

def take_subset_data_features(DATA_FEATURES, batch_size, fraction=0.1):
    for fold in DATA_FEATURES.keys():
        DATA_FEATURES[fold] = DATA_FEATURES[fold].iloc[:(batch_size*int(len(DATA_FEATURES[fold].index)/batch_size*fraction)), :]
    return DATA_FEATURES

def generate_class_weights(data_features, target):
    class_weights = None
    if dict_prediction_types[target] == 'binary':
        class_weights={}
        counts = data_features[target].value_counts()
        for i in counts.index.values:
            class_weights[i] = 1/counts.loc[i]
    return class_weights

def generate_generators(DATA_FEATURES, target, dir_images, image_size, batch_size, folds, seed, mode):
    GENERATORS = {}
    STEP_SIZES = {}
    for fold in folds:
        #Do not generate a generator if there are no samples (can happen for leftovers generators)
        if len(DATA_FEATURES[fold].index) == 0:
            continue
        
        if fold == 'train':
            datagen = ImageDataGenerator(rescale=1./255., rotation_range=20, width_shift_range=0.2, height_shift_range=0.2)
            shuffle = True if mode == 'model_training' else False
        else:
            datagen = ImageDataGenerator(rescale=1./255.)
            shuffle = False
        
        #define batch size for testing: data is split between a part that fits in batches, and leftovers
        batch_size_fold = min(batch_size, len(DATA_FEATURES[fold].index)) if mode == 'model_testing' else batch_size
        
        # define data generator
        generator_fold = datagen.flow_from_dataframe(
            dataframe=DATA_FEATURES[fold],
            directory=dir_images,
            x_col='eid',
            y_col=target,
            color_mode='rgb',
            batch_size= batch_size_fold,
            seed=seed,
            shuffle=shuffle,
            class_mode='raw',
            target_size=(image_size, image_size))
        
        # assign variables to their names
        GENERATORS[fold] = generator_fold
        STEP_SIZES[fold] = generator_fold.n//generator_fold.batch_size
    return GENERATORS, STEP_SIZES

def generate_base_model(architecture, weight_decay, dropout_rate, keras_weights):
    if architecture in ['VGG16', 'VGG19']:
        if architecture == 'VGG16':
            from keras.applications.vgg16 import VGG16
            base_model = VGG16(include_top=False, weights=keras_weights, input_shape=(224,224,3))
        elif architecture == 'VGG19':
            from keras.applications.vgg19 import VGG19
            base_model = VGG19(include_top=False, weights=keras_weights, input_shape=(224,224,3))
        x = base_model.output
        x = Flatten()(x)
        x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Dropout(dropout_rate)(x) 
    elif architecture in ['MobileNet', 'MobileNetV2']:
        if architecture == 'MobileNet':
            from keras.applications.mobilenet import MobileNet
            base_model = MobileNet(include_top=False, weights=keras_weights, input_shape=(224,224,3))
        elif architecture == 'MobileNetV2':
            from keras.applications.mobilenet_v2 import MobileNetV2
            base_model = MobileNetV2(include_top=False, weights=keras_weights, input_shape=(224,224,3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
    elif architecture in ['DenseNet121', 'DenseNet169', 'DenseNet201']:
        if architecture == 'DenseNet121':
            from keras.applications.densenet import DenseNet121
            base_model = DenseNet121(include_top=True, weights=keras_weights, input_shape=(224,224,3))
        elif architecture == 'DenseNet169':
            from keras.applications.densenet import DenseNet169
            base_model = DenseNet169(include_top=True, weights=keras_weights, input_shape=(224,224,3))
        elif architecture == 'DenseNet201':
            from keras.applications.densenet import DenseNet201
            base_model = DenseNet201(include_top=True, weights=keras_weights, input_shape=(224,224,3))            
        base_model = Model(base_model.inputs, base_model.layers[-2].output)
        x = base_model.output
    elif architecture in ['NASNetMobile', 'NASNetLarge']:
        if architecture == 'NASNetMobile':
            from keras.applications.nasnet import NASNetMobile
            base_model = NASNetMobile(include_top=True, weights=keras_weights, input_shape=(224,224,3))
        elif architecture == 'NASNetLarge':
            from keras.applications.nasnet import NASNetLarge
            base_model = NASNetLarge(include_top=True, weights=keras_weights, input_shape=(331,331,3))
        base_model = Model(base_model.inputs, base_model.layers[-2].output)
        x = base_model.output
    elif architecture == 'Xception':
        from keras.applications.xception import Xception
        base_model = Xception(include_top=False, weights=keras_weights, input_shape=(299,299,3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
    elif architecture == 'InceptionV3':
        from keras.applications.inception_v3 import InceptionV3
        base_model = InceptionV3(include_top=False, weights=keras_weights, input_shape=(299,299,3))
        x = base_model.output        
        x = GlobalAveragePooling2D()(x)
    elif architecture == 'InceptionResNetV2':
        from keras.applications.inception_resnet_v2 import InceptionResNetV2
        base_model = InceptionResNetV2(include_top=False, weights=keras_weights, input_shape=(299,299,3))
        x = base_model.output        
        x = GlobalAveragePooling2D()(x)
    return x, base_model.input

def complete_architecture(x, input_shape, activation, weight_decay, dropout_rate):
    x = Dense(1024, activation='selu', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(512, activation='selu', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256, activation='selu', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='selu', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='selu', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(1, activation=activation)(x)
    model = Model(inputs=input_shape, outputs=predictions)
    return model

"""
def complete_architecture(x, input_shape, activation, weight_decay, dropout_rate):
    x = Dense(1024, activation='selu', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(512, activation='selu', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256, activation='selu', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='selu')(x)
    x = Dense(64, activation='selu')(x)
    x = Dense(32, activation='selu')(x)
    x = Dense(16, activation='selu')(x)
    x = Dense(8, activation='selu')(x)
    x = Dense(4, activation='selu')(x)
    x = Dense(2, activation='selu')(x)
    predictions = Dense(1, activation=activation)(x)
    model = Model(inputs=input_shape, outputs=predictions)
    return model


def complete_architecture(x, input_shape, activation, weight_decay, dropout_rate):
    for n in [int(2**(10-i)) for i in range(10)]:
        x = Dense(n, activation='selu', kernel_regularizer=regularizers.l2(weight_decay), kernel_constraint=max_norm(keras_max_norm))(x)
        if n < 3:
            x = Dropout(dropout_rate/(n+1))(x)
    predictions = Dense(1, activation=activation)(x)
    model = Model(inputs=input_shape, outputs=predictions)
    return model


def generate_generators(DATA_FEATURES, target, dir_images, image_size, batch_size, folds, seed, mode):
    GENERATORS = {}
    STEP_SIZES = {}
    for fold in folds:
        #Do not generate a generator if there are no samples (can happen for leftovers generators)
        if len(DATA_FEATURES[fold].index) == 0:
            continue
        
        if fold == 'train':
            datagen = ImageDataGenerator(rescale=1./255., rotation_range=10, width_shift_range=0.1, height_shift_range=0.1)
            shuffle = True if mode == 'model_training' else False
        else:
            datagen = ImageDataGenerator(rescale=1./255.)
            shuffle = False
        
        #define batch size for testing: data is split between a part that fits in batches, and leftovers
        batch_size_fold = min(batch_size, len(DATA_FEATURES[fold].index)) if mode == 'model_testing' else batch_size
        
        # define data generator
        generator_fold = datagen.flow_from_dataframe(
            dataframe=DATA_FEATURES[fold],
            directory=dir_images,
            x_col='eid',
            y_col=target,
            color_mode='rgb',
            batch_size= batch_size_fold,
            seed=seed,
            shuffle=shuffle,
            class_mode='raw',
            target_size=(image_size, image_size))
        
        #TODO: modify to allow multi input
        a = 0
        while a == 0:
            a = 1
            X1i = generator_fold.next()
            yield [X1i[0], X1i[2]], X1i[1]
        # assign variables to their names
        GENERATORS[fold] = generator_fold
        STEP_SIZES[fold] = generator_fold.n//generator_fold.batch_size
    return GENERATORS, STEP_SIZES

def generate_cnn(architecture, weight_decay, dropout_rate, keras_weights):
    if architecture in ['VGG16', 'VGG19']:
        if architecture == 'VGG16':
            from keras.applications.vgg16 import VGG16
            cnn = VGG16(include_top=False, weights=keras_weights, input_shape=(224,224,3))
        elif architecture == 'VGG19':
            from keras.applications.vgg19 import VGG19
            cnn = VGG19(include_top=False, weights=keras_weights, input_shape=(224,224,3))
        x = cnn.output
        x = Flatten()(x)
        x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Dropout(dropout_rate)(x) 
    elif architecture in ['MobileNet', 'MobileNetV2']:
        if architecture == 'MobileNet':
            from keras.applications.mobilenet import MobileNet
            cnn = MobileNet(include_top=False, weights=keras_weights, input_shape=(224,224,3))
        elif architecture == 'MobileNetV2':
            from keras.applications.mobilenet_v2 import MobileNetV2
            cnn = MobileNetV2(include_top=False, weights=keras_weights, input_shape=(224,224,3))
        x = cnn.output
        x = GlobalAveragePooling2D()(x)
    elif architecture in ['DenseNet121', 'DenseNet169', 'DenseNet201']:
        if architecture == 'DenseNet121':
            from keras.applications.densenet import DenseNet121
            cnn = DenseNet121(include_top=True, weights=keras_weights, input_shape=(224,224,3))
        elif architecture == 'DenseNet169':
            from keras.applications.densenet import DenseNet169
            cnn = DenseNet169(include_top=True, weights=keras_weights, input_shape=(224,224,3))
        elif architecture == 'DenseNet201':
            from keras.applications.densenet import DenseNet201
            cnn = DenseNet201(include_top=True, weights=keras_weights, input_shape=(224,224,3))            
        cnn = Model(cnn.inputs, cnn.layers[-2].output)
        x = cnn.output
    elif architecture in ['NASNetMobile', 'NASNetLarge']:
        if architecture == 'NASNetMobile':
            from keras.applications.nasnet import NASNetMobile
            cnn = NASNetMobile(include_top=True, weights=keras_weights, input_shape=(224,224,3))
        elif architecture == 'NASNetLarge':
            from keras.applications.nasnet import NASNetLarge
            cnn = NASNetLarge(include_top=True, weights=keras_weights, input_shape=(331,331,3))
        cnn = Model(cnn.inputs, cnn.layers[-2].output)
        x = cnn.output
    elif architecture == 'Xception':
        from keras.applications.xception import Xception
        cnn = Xception(include_top=False, weights=keras_weights, input_shape=(299,299,3))
        x = cnn.output
        x = GlobalAveragePooling2D()(x)
    elif architecture == 'InceptionV3':
        from keras.applications.inception_v3 import InceptionV3
        cnn = InceptionV3(include_top=False, weights=keras_weights, input_shape=(299,299,3))
        x = cnn.output        
        x = GlobalAveragePooling2D()(x)
    elif architecture == 'InceptionResNetV2':
        from keras.applications.inception_resnet_v2 import InceptionResNetV2
        cnn = InceptionResNetV2(include_top=False, weights=keras_weights, input_shape=(299,299,3))
        x = cnn.output        
        x = GlobalAveragePooling2D()(x)
    return cnn.input, x

def generate_side_nn(dim):
	side_nn = Sequential()
	side_nn.add(Dense(8, input_dim=dim, activation="relu"))
	side_nn.add(Dense(4, activation="relu"))
	return side_nn.input, side_nn.output

def complete_architecture(cnn_input, cnn_output, side_nn_input,side_nn_output, activation, weight_decay, dropout_rate):
    x = concatenate([cnn_output, side_nn_output])
    for n in [int(2**(10-i)) for i in range(10)]:
        if n < 3:
            x = Dense(n, activation='selu', kernel_regularizer=regularizers.l2(weight_decay))(x)
            x = Dropout(dropout_rate)(x)
        else:
            x = Dense(n, activation='selu')(x)
        
    predictions = Dense(1, activation=activation)(x)
    model = Model(inputs=[input_cnn, input_nn], outputs=predictions)
    return model

"""

"""Find the most similar model for transfer learning.
The function returns path_load_weights, keras_weights """
def weights_for_transfer_learning(continue_training, max_transfer_learning, path_weights, list_parameters_to_match):
    
    print('Looking for models to transfer weights from...')
    
    #define parameters
    version = path_weights.replace('../data/model-weights_', '').replace('.h5', '')
    parameters = version_to_parameters(version, names_model_parameters)
    
    #continue training if possible
    if continue_training and os.path.exists(path_weights):
        print('Loading the weights from the model\'s previous training iteration.')
        return path_weights, None
    
    #Look for similar models, starting from very similar to less similar
    if max_transfer_learning:
        while(True):  
            #print('Matching models for the following criterias:'); print(['architecture', 'target'] + list_parameters_to_match)
            #start by looking for models trained on the same target, then move to other targets
            for target_to_load in dict_alternative_targets_for_transfer_learning[parameters['target']]:
                #print('Target used: ' + target_to_load)
                parameters_to_match = parameters.copy()
                parameters_to_match['target'] = target_to_load
                #load the ranked performances table to select the best performing model among the similar models available
                path_performances_to_load = path_store + 'PERFORMANCES_ranked_' + parameters_to_match['target'] + '_' + 'val' + '_' + dict_eids_version[parameters['organ']] + '.csv'
                try:
                    Performances = pd.read_csv(path_performances_to_load)
                    Performances['field_id'] = Performances['field_id'].astype(str)
                except:
                    #print("Could not load the file: " + path_performances_to_load)
                    break
                #iteratively get rid of models that are not similar enough, based on the list
                for parameter in ['architecture', 'target'] + list_parameters_to_match:
                    Performances = Performances[Performances[parameter] == parameters_to_match[parameter]]
                #if at least one model is similar enough, load weights from the best of them
                if(len(Performances.index) != 0):
                    path_weights_to_load = path_store + 'model-weights_' + Performances['version'][0] + '.h5'
                    print('transfering the weights from: ' + path_weights_to_load)
                    return path_weights_to_load, None
            
            #if no similar model was found, try again after getting rid of the last selection criteria
            if(len(list_parameters_to_match) == 0):
                print('No model found for transfer learning.')
                break
            list_parameters_to_match.pop()
    
    #Otherwise use imagenet weights to initialize
    print('Using imagenet weights.')
    return 'load_path_weights_should_not_be_used', 'imagenet'

class myModelCheckpoint(ModelCheckpoint):
    """Take as input baseline instead of np.Inf, useful if model has already been trained.
    """
    def __init__(self, filepath, monitor='val_loss', baseline=np.Inf, verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        
        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best = baseline
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = baseline
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = baseline

#model_checkpoint_backup is used in case the weights file gets corrupted because the job gets killed during its writing
def define_callbacks(path_store, version, baseline, continue_training, main_metric, main_metric_mode):
    csv_logger = CSVLogger(path_store + 'logger_' + version + '.csv', separator=',', append=continue_training)
    model_checkpoint_backup = myModelCheckpoint(path_store + 'backup-model-weights_' + version + '.h5', monitor='val_' + main_metric.__name__, baseline=baseline, verbose=1, save_best_only=True, save_weights_only=True, mode=main_metric_mode, period=1)
    model_checkpoint = myModelCheckpoint(path_store + 'model-weights_' + version + '.h5', monitor='val_' + main_metric.__name__, baseline=baseline, verbose=1, save_best_only=True, save_weights_only=True, mode=main_metric_mode, period=1)
    reduce_lr_on_plateau = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=1, mode='min', min_delta=0, cooldown=0, min_lr=0)
    early_stopping = EarlyStopping(monitor= 'val_' + main_metric.__name__, min_delta=0, patience=5, verbose=0, mode=main_metric_mode, baseline=None, restore_best_weights=False)
    return [csv_logger, model_checkpoint_backup, model_checkpoint, reduce_lr_on_plateau, early_stopping]

def R2_K(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def RMSE_K(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def sensitivity_K(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity_K(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def ROC_AUC_K(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred, curve='ROC')[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def recall_K(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_K(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def PR_AUC_K(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred, curve='PR', summation_method='careful_interpolation')[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def F1_K(y_true, y_pred):
    precision = precision_K(y_true, y_pred)
    recall = recall_K(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def true_positives_K(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

def false_positives_K(y_true, y_pred):
    return K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))

def false_negatives_K(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true * (1-y_pred), 0, 1)))

def true_negatives_K(y_true, y_pred):
    return K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))

def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

def sensitivity_score(y, pred):
    _, _, fn, tp = confusion_matrix(y, pred.round()).ravel()
    return tp/(tp+fn)

def specificity_score(y, pred):
    tn, fp, _, _ = confusion_matrix(y, pred.round()).ravel()
    return tn/(tn+fp)

def true_positives_score(y, pred):
    _, _, _, tp = confusion_matrix(y, pred.round()).ravel()
    return tp

def false_positives_score(y, pred):
    _, fp, _, _ = confusion_matrix(y, pred.round()).ravel()
    return fp

def false_negatives_score(y, pred):
    _, _, fn, _ = confusion_matrix(y, pred.round()).ravel()
    return fn

def true_negatives_score(y, pred):
    tn, _, _, _ = confusion_matrix(y, pred.round()).ravel()
    return tn

def bootstrap(data, n_bootstrap_iterations, function):
    results = []
    for i in range(n_bootstrap_iterations):
        data_i = resample(data, replace=True, n_samples=len(data.index))
        results.append(function(data_i['y'], data_i['pred']))
    return np.mean(results), np.std(results)

def set_learning_rate(model, optimizer, learning_rate, loss, metrics):
    opt = globals()[optimizer](lr=learning_rate, clipnorm=1.0)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)

def save_model_weights(model, path_store, version):
    model.save_weights(path_store + "model_weights_" + version + ".h5")
    print("Model's best weights for "+ version + " were saved.")

def plot_training(path_store, version, display_learning_rate):
    try:
        logger = pd.read_csv(path_store + 'logger_' + version + '.csv')
    except:
        print('ERROR: THE FILE logger_' + version + '.csv' ' WAS NOT FOUND OR WAS EMPTY/CORRUPTED. SKIPPING PLOTTING OF THE TRAINING FOR THIS MODEL.')
        return
    #Amend column names for consistency
    logger.columns = [name[:-2] if name.endswith('_K') else name for name in logger.columns]
    metrics_names = [metric[4:] for metric in logger.columns.values if metric.startswith('val_')]
    logger.columns = ['train_' + name if name in metrics_names else name for name in logger.columns]
    #rewrite epochs numbers based on nrows, because several loggers might have been appended if the model has been retrained.
    logger['epoch'] = [i+1 for i in range(len(logger.index))]
    #multiplot layout
    n_rows=3
    n_metrics = len(metrics_names)
    fig, axs = plt.subplots(math.ceil(n_metrics/n_rows), min(n_metrics,n_rows), sharey=False, sharex=True, squeeze=False)
    fig.set_figwidth(5*n_metrics)
    fig.set_figheight(5)

    #plot evolution of each metric during training, train and val values
    for k, metric in enumerate(metrics_names):
        i=int(k/n_rows)
        j=k%n_rows
        for fold in folds_tune:
            axs[i,j].plot(logger['epoch'], logger[fold + '_' + metric])
        axs[i,j].legend(['Training', 'Validation'], loc='upper left')
        axs[i,j].set_title(metric + ' = f(Epoch)')
        axs[i,j].set_xlabel('Epoch')
        axs[i,j].set_ylabel(metric)
        if metric not in ['true_positives', 'false_positives', 'false_negatives', 'true_negatives']:
            axs[i,j].set_ylim((-0.2, 1.1))
        #use second axis for learning rate
        if display_learning_rate & ('lr' in logger.columns):
            ax2 = axs[i,j].twinx()
            ax2.plot(logger['epoch'], logger['lr'], color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            ax2.legend(['Learning Rate'], loc='upper right')
    fig.tight_layout()
    #save figure as pdf before closing
    fig.savefig("../figures/Training_" + version + '.pdf', bbox_inches='tight')
    plt.close('all')

def preprocess_data_features_predictions_for_performances(path_store, id_set, target):
    #load dataset
    if id_set == 'A':
        data_features = pd.read_csv("/n/groups/patel/uk_biobank/main_data_9512/data_features.csv")[['f.eid', 'f.31.0.0', 'f.21003.0.0']]
        data_features.replace({'f.31.0.0': {'Male': 0, 'Female': 1}}, inplace=True)
    elif id_set == 'B':
        cols_data_features = ['eid', 'Sex', dict_field_id_to_age_instance[field_id]]
        data_features = pd.read_csv(path_store + 'data-features.csv', usecols = cols_data_features)
        data_features.rename(columns={dict_rename_cols}, inplace=True)
    else:
        print('ERROR: id_set must be either A or B')
        sys.exit(1)
    #format data_features to extract y
    data_features.columns = ['eid', 'Sex', 'Age']
    data_features.rename(columns={target:'y'}, inplace=True)
    data_features = data_features[['eid', 'y']]
    data_features['eid'] = data_features['eid'].astype(str)
    data_features['eid'] = data_features['eid']
    data_features = data_features.set_index('eid', drop=False)
    data_features.index.name = 'column_names'
    return data_features

def preprocess_predictions_for_performances(data_features, path_store, version, fold, id_set):
    #load Predictions the initial dataframe to extract 'y'
    Predictions = pd.read_csv(path_store + 'Predictions_' + version + '_' + fold + '_' + id_set + '.csv')
    Predictions['eid'] = Predictions['eid'].astype(str)
    Predictions.rename(columns={'Pred_' + version:'pred'}, inplace=True)
    Predictions = Predictions.merge(data_features, how='inner', on=['eid'])
    return Predictions

#Initialize performances dataframes and compute sample sizes
def initiate_empty_performances_df(Predictions, target, names_metrics):
    #Define an empty performances dataframe to store the performances computed
    row_names = ['all'] + outer_folds
    col_names_sample_sizes = ['N']
    if target in targets_binary:
        col_names_sample_sizes.extend(['N_0', 'N_1'])
    col_names = ['outer_fold'] + col_names_sample_sizes
    col_names.extend(names_metrics)
    performances = np.empty((len(row_names),len(col_names),))
    performances.fill(np.nan)
    performances = pd.DataFrame(performances)
    performances.index = row_names
    performances.columns = col_names
    performances['outer_fold'] = row_names
    #Convert float to int for sample sizes and some metrics.
    for col_name in col_names_sample_sizes:
        performances[col_name] = performances[col_name].astype('Int64') #need recent version of pandas to use this type. Otherwise nan cannot be int
    
    #compute sample sizes for the data frame
    performances.loc['all', 'N'] = len(Predictions.index)
    if target in targets_binary:
        performances.loc['all', 'N_0'] = len(Predictions.loc[Predictions['y']==0].index)
        performances.loc['all', 'N_1'] = len(Predictions.loc[Predictions['y']==1].index)
    for outer_fold in outer_folds:
        performances.loc[outer_fold, 'N'] = len(Predictions.loc[Predictions['outer_fold']==int(outer_fold)].index)
        if target in targets_binary:
            performances.loc[outer_fold, 'N_0'] = len(Predictions.loc[(Predictions['outer_fold']==int(outer_fold)) & (Predictions['y']==0)].index)
            performances.loc[outer_fold, 'N_1'] = len(Predictions.loc[(Predictions['outer_fold']==int(outer_fold)) & (Predictions['y']==1)].index)
            
    #initialize the dataframes
    PERFORMANCES={}
    for mode in modes:
        PERFORMANCES[mode] = performances.copy()
    
    #Convert float to int for sample sizes and some metrics.
    for col_name in PERFORMANCES[''].columns.values:
        if any(metric in col_name for metric in metrics_displayed_in_int):
            PERFORMANCES[''][col_name] = PERFORMANCES[''][col_name].astype('Int64') #need recent version of pandas to use this type. Otherwise nan cannot be int
    
    return PERFORMANCES

#Fill the columns for this model, outer_fold by outer_fold
def fill_performances_matrix_for_single_model(Predictions, target, version, fold, id_set, names_metrics, n_bootstrap_iterations, save_performances):
    #initialize dictionary of empty dataframes to save the performances
    PERFORMANCES = initiate_empty_performances_df(Predictions, target, names_metrics)
    
    #fill it outer_fold by outer_fold
    for outer_fold in ['all'] + outer_folds:
        print('Calculating the performances for the outer fold ' + outer_fold)
        #Generate a subdataframe from the predictions table for each outerfold
        if outer_fold == 'all':
            predictions_fold = Predictions.copy()
        else:
            predictions_fold = Predictions.loc[Predictions['outer_fold'] == int(outer_fold),:]
        
        #if no samples are available for this fold, fill columns with nans
        if(len(predictions_fold.index) == 0):
            print('NO SAMPLES AVAILABLE FOR MODEL ' + version + ' IN OUTER_FOLD ' + outer_fold)                    
        else:
            #For binary classification, generate class prediction
            if target in targets_binary:
                predictions_fold_class = predictions_fold.copy()
                predictions_fold_class['pred'] = predictions_fold_class['pred'].round()
            
            #Fill the Performances dataframe metric by metric
            for name_metric in names_metrics:
                #print('Calculating the performance using the metric ' + name_metric)
                predictions_metric = predictions_fold_class if name_metric in metrics_needing_classpred else predictions_fold
                metric_function = dict_metrics[name_metric]['sklearn']
                PERFORMANCES[''].loc[outer_fold, name_metric] = metric_function(predictions_metric['y'], predictions_metric['pred'])
                PERFORMANCES['_sd'].loc[outer_fold, name_metric] = bootstrap(predictions_metric, n_bootstrap_iterations, metric_function)[1]
                PERFORMANCES['_str'].loc[outer_fold, name_metric] = "{:.3f}".format(PERFORMANCES[''].loc[outer_fold, name_metric]) + '+-' + "{:.3f}".format(PERFORMANCES['_sd'].loc[outer_fold, name_metric])
    
    #calculate the fold sd (variance between the metrics values obtained on the different folds)
    folds_sd = PERFORMANCES[''].iloc[1:,:].std(axis=0)
    for name_metric in names_metrics:
        PERFORMANCES['_str'].loc['all', name_metric] = "{:.3f}".format(PERFORMANCES[''].loc['all', name_metric]) + '+-' + "{:.3f}".format(folds_sd[name_metric]) + '+-' + "{:.3f}".format(PERFORMANCES['_sd'].loc['all', name_metric])
    
    # save performances
    if save_performances:
        for mode in modes:
            PERFORMANCES[mode].to_csv(path_store + 'Performances_' + version + '_' + fold + '_' + id_set + mode + '.csv', index=False)
    
    return PERFORMANCES

def initiate_empty_performances_summary_df(target, list_models):
    #Define the columns of the Performances dataframe
    #columns for sample sizes
    names_sample_sizes = ['N']
    if target in targets_binary:
        names_sample_sizes.extend(['N_0', 'N_1'])
    
    #columns for metrics
    names_metrics = dict_metrics_names[dict_prediction_types[target]]
    #for normal folds, keep track of metric and bootstrapped metric's sd
    names_metrics_with_sd = []
    for name_metric in names_metrics:
        names_metrics_with_sd.extend([name_metric, name_metric + '_sd', name_metric + '_str'])
    
    #for the 'all' fold, also keep track of the 'folds_sd', the metric's sd calculated using the folds' metrics results
    names_metrics_with_folds_sd_and_sd = []
    for name_metric in names_metrics:
        names_metrics_with_folds_sd_and_sd.extend([name_metric, name_metric + '_folds_sd', name_metric + '_sd', name_metric + '_str'])
    
    #merge all the columns together. First description of the model, then sample sizes and metrics for each fold
    names_col_Performances = ['version'] + names_model_parameters #.copy()
    #special outer fold 'all'
    names_col_Performances.extend(['_'.join([name,'all']) for name in names_sample_sizes + names_metrics_with_folds_sd_and_sd])
    #other outer_folds
    for outer_fold in outer_folds:
        names_col_Performances.extend(['_'.join([name,outer_fold]) for name in names_sample_sizes + names_metrics_with_sd])
    
    #Generate the empty Performance table from the rows and columns.
    Performances = np.empty((len(list_models),len(names_col_Performances),))
    Performances.fill(np.nan)
    Performances = pd.DataFrame(Performances)
    Performances.columns = names_col_Performances
    #Format the types of the columns
    for colname in Performances.columns.values:
        if (colname in names_model_parameters) | ('_str' in colname):
            col_type = str
        else:
            col_type = float
        Performances[colname] = Performances[colname].astype(col_type)
    return Performances

def fill_summary_performances_matrix(list_models, target, fold, id_set, ensemble_models, save_performances):
    #define parameters
    names_metrics = dict_metrics_names[dict_prediction_types[target]]
    
    #initiate dataframe
    Performances = initiate_empty_performances_summary_df(target, list_models)
    
    #Fill the Performance table row by row
    for i, model in enumerate(list_models):
        #load the performances subdataframe
        PERFORMANCES={}
        for mode in modes:
            PERFORMANCES[mode] = pd.read_csv(model.replace('_str',mode))
            PERFORMANCES[mode].set_index('outer_fold', drop=False, inplace=True)
            
        #Fill the columns corresponding to the model's parameters
        version = '_'.join(model.split('_')[1:-3])
        parameters = version_to_parameters(version, names_model_parameters)
        
        #fill the columns for model parameters
        Performances['version'][i] = version
        for parameter_name in names_model_parameters:
            Performances[parameter_name][i] = parameters[parameter_name]
        
        #Fill the columns for this model, outer_fold by outer_fold
        for outer_fold in ['all'] + outer_folds:
            #Generate a subdataframe from the predictions table for each outerfold
            
            #Fill sample size columns
            Performances['N_' + outer_fold][i] = PERFORMANCES[''].loc[outer_fold,'N']
            
            #For binary classification, calculate sample sizes for each class and generate class prediction
            if target in targets_binary:
                Performances['N_0_' + outer_fold][i] = PERFORMANCES[''].loc[outer_fold,'N_0']
                Performances['N_1_' + outer_fold][i] = PERFORMANCES[''].loc[outer_fold,'N_1']
            
            #Fill the Performances dataframe metric by metric
            for name_metric in names_metrics:
                for mode in modes:
                    Performances[name_metric + mode + '_' + outer_fold][i] = PERFORMANCES[mode].loc[outer_fold,name_metric]
            
            #calculate the fold sd (variance between the metrics values obtained on the different folds)
            folds_sd = PERFORMANCES[''].iloc[1:,:].std(axis=0)
            for name_metric in names_metrics:
                Performances[name_metric + '_folds_sd_all'] = folds_sd[name_metric]
    
    #Convert float to int for sample sizes and some metrics.
    for name_col in Performances.columns.values:
        if name_col.startswith('N_') | any(metric in name_col for metric in metrics_displayed_in_int) & (not '_sd' in name_col) & (not '_str' in name_col):
            Performances[name_col] = Performances[name_col].astype('Int64') #need recent version of pandas to use this type. Otherwise nan cannot be int
    
    #rename the version column to get rid of fold
    #Performances['version'] = Performances['version'].str.rstrip('_' + fold)
    
    #For ensemble models, merge the new performances with the previously computed performances
    if ensemble_models:
        Performances_withoutEnsembles = pd.read_csv(path_store + 'PERFORMANCES_withoutEnsembles_alphabetical_' + target + '_' + fold + '_' + id_set + '.csv')
        Performances = Performances_withoutEnsembles.append(Performances)
    
    #Ranking, printing and saving
    Performances_alphabetical = Performances.sort_values(by='version')
    print('Performances of the models ranked by models\'names:')
    print(Performances_alphabetical)
    Performances_ranked = Performances.sort_values(by=dict_main_metrics_names[target] + '_all', ascending=main_metrics_modes[dict_main_metrics_names[target]] == 'min')
    print('Performances of the models ranked by the performance on the main metric on all the samples:')
    print(Performances_ranked)
    if save_performances:
        name_extension = 'withEnsembles' if ensemble_models else 'withoutEnsembles'
        Performances_alphabetical.to_csv(path_store + 'PERFORMANCES_' + name_extension + '_alphabetical_' + target + '_' + fold + '_' + id_set + '.csv', index=False)
        Performances_ranked.to_csv(path_store + 'PERFORMANCES_' + name_extension + '_ranked_' + target + '_' + fold + '_' + id_set + '.csv', index=False)
    return Performances_ranked

"""Build the best ensemble model. To do so, consider a 2x2 matrix of strategies.
1-Iteratively include all the models, or only the ones that improve the performance?
2-Weight the models by the validation performance, or give the same weight to every model?
This function is more sophisticated than the other method I used to ensemble, but possibly leads to overfitting on the testing set since the samples are shared between the validation and the testing set at this point"""
def build_ensemble_model_OVERFITTING(Performances_subset, Predictions, y, main_metric_name):
    #define the parameters
    main_metric_function = dict_metrics[main_metric_name]['sklearn']
    main_metric_mode = main_metrics_modes[main_metric_name]
    best_perf = -np.Inf if main_metric_mode == 'max' else np.Inf
    ENSEMBLE_COLS={'select':{'noweights':[], 'weighted':[]}, 'all':{'noweights':[], 'weighted':[]}}
    
    #iteratively add models to the ensemble
    print('The model is being built using ' + str(len(Performances_subset['version'])) + ' different models.')
    for i, version in enumerate(Performances_subset['version']):
        print('Considering the adding of the ' + str(i) + 'th model to the ensemble: ' + version)
        added_pred_name = 'Pred_' + version
        #two different strategies for the ensemble: 'select' only keeps the previous models that improved the accuracy. 'all' keeps all the previous models.
        for ensemble_type in ENSEMBLE_COLS.keys():
            #two different kinds of weighting: 'noweights' gives the same weight (1) to all the models included. 'weighted' weights based on the validation main metric.
            for weighting_type in ENSEMBLE_COLS[ensemble_type].keys():
                ENSEMBLE_COLS[ensemble_type][weighting_type].append(added_pred_name)
                if weighting_type == 'weighted':
                    weights = Performances_subset.loc[[version.replace('Pred_','') for version in ENSEMBLE_COLS[ensemble_type][weighting_type]], main_metric_name + '_all'].values
                else:
                    weights = np.ones(len(ENSEMBLE_COLS[ensemble_type][weighting_type]))
                Predictions_subset = Predictions[ENSEMBLE_COLS[ensemble_type][weighting_type]]
                Ensemble_predictions = Predictions_subset*weights
                Ensemble_predictions = Ensemble_predictions.sum(axis=1, skipna=False)/np.sum(weights)
                
                #format the dataframe
                df_ensemble = pd.concat([y, Ensemble_predictions], axis=1).dropna()
                df_ensemble.columns = ['y', 'pred']
                
                #evaluate the ensemble predictions
                new_perf = main_metric_function(df_ensemble['y'],df_ensemble['pred'])
                if new_perf > best_perf:
                    best_perf = new_perf
                    best_pred = df_ensemble['pred']
                    best_ensemble_namecols = ENSEMBLE_COLS[ensemble_type][weighting_type].copy()
                    best_weights = weights
                    print('THE MODEL IMPROVED! The new best perf is: ' + str(round(best_perf,3)) + '. It was found using ensemble type = ' + ensemble_type + ' and weighting_type = ' + weighting_type + '. The ensemble model was built using ' + str(len(best_ensemble_namecols)) + ' different models.')
                elif ensemble_type == 'select':
                    popped = ENSEMBLE_COLS[ensemble_type][weighting_type].pop()
    
    return best_ensemble_namecols, best_weights

#returns True if the dataframe is a single column duplicated. Used to check if the folds are the same for the entire ensemble model
def is_rank_one(df):
    for i in range(len(df.columns)):
        for j in range(i+1,len(df.columns)):
            if not df.iloc[:,i].equals(df.iloc[:,j]):
                return False
    return True 

def weighted_weights_by_category(weights, Performances, ensemble_level):
    weights_names = weights.index.values
    for category in Performances[ensemble_level].unique():
        n_category = len([name for name in weights_names if category in name])
        for weight_name in weights.index.values:
            if category in weight_name:
                weights[weight_name] = weights[weight_name]/n_category
    weights = weights.values/weights.values.sum()
    return weights

def weighted_weights_by_ensembles(Predictions, Performances, parameters, ensemble_level, main_metric_name):
    sub_levels = Performances[ensemble_level].unique()
    ensemble_namecols = []
    weights = []
    for sub in sub_levels:
        parameters_sub = parameters.copy()
        parameters_sub[ensemble_level] = sub
        version_sub = parameters_to_version(parameters_sub)
        ensemble_namecols.append('pred_' + version_sub)
        df_score = Predictions[[parameters['target'], 'pred_' + version_sub]]
        df_score.dropna(inplace=True)
        weight = dict_metrics[main_metric_name]['sklearn'](df_score[parameters['target']], df_score['pred_' + version_sub])
        weights.append(weight)
    weights = np.array(weights)
    weights = weights/weights.sum()
    return ensemble_namecols, weights

def build_single_ensemble(PREDICTIONS, target, main_metric_name, id_set, Performances, parameters, version, list_ensemble_levels, ensemble_level):
    #define which models should be integrated into the ensemble model, and how they should be weighted
    Predictions = PREDICTIONS['val']
    y = Predictions[target]
    performance_cutoff = np.max(Performances[main_metric_name + '_all'])*ensembles_performance_cutoff_percent
    ensemble_namecols = ['pred_' + model_name for model_name in Performances['version'][Performances[main_metric_name + '_all'] > performance_cutoff]]
    
    #calculate the ensemble model using two different kinds of weights
    ensemble_outerfolds = [model_name.replace('pred_', 'outer_fold_') for model_name in ensemble_namecols]
    #weighted by performance
    weights_with_names = Performances[main_metric_name + '_all'][Performances[main_metric_name + '_all'] > performance_cutoff]
    weights = weights_with_names.values/weights_with_names.values.sum()
    if len(list_ensemble_levels) > 0:
        #weighted by both performance and subcategories
        weights_by_category = weighted_weights_by_category(weights_with_names, Performances, ensemble_level)
        #weighted by the performance of the ensemble models right below it
        sub_ensemble_names, weights_by_ensembles = weighted_weights_by_ensembles(Predictions, Performances, parameters, ensemble_level, main_metric_name)
    
    #for each fold, build the ensemble model
    for fold in folds:
        Ensemble_predictions = PREDICTIONS[fold][ensemble_namecols]*weights
        PREDICTIONS[fold]['pred_' + version] = Ensemble_predictions.sum(axis=1, skipna=False)
        if len(list_ensemble_levels) > 0:
            Ensemble_predictions_weighted_by_category = PREDICTIONS[fold][ensemble_namecols]*weights_by_category
            Ensemble_predictions_weighted_by_ensembles = PREDICTIONS[fold][sub_ensemble_names]*weights_by_ensembles
            PREDICTIONS[fold]['pred_' + version.replace('*', ',')] = Ensemble_predictions_weighted_by_category.sum(axis=1, skipna=False)
            PREDICTIONS[fold]['pred_' + version.replace('*', '?')] = Ensemble_predictions_weighted_by_ensembles.sum(axis=1, skipna=False)

def build_single_ensemble_wrapper(PREDICTIONS, target, main_metric_name, id_set, Performances, parameters, version, list_ensemble_levels, ensemble_level):
    Predictions = PREDICTIONS['val']
    #Select the outerfolds columns for the model
    ensemble_outerfolds_cols = [name_col for name_col in Predictions.columns.values if bool(re.compile('outer_fold_' + version).match(name_col))]
    Ensemble_outerfolds = Predictions[ensemble_outerfolds_cols]
    
    #Evaluate if the model can be built piece by piece on each outer_fold, or if the folds are not shared and the model should be built on all the folds at once
    if not is_rank_one(Ensemble_outerfolds):
        build_single_ensemble(PREDICTIONS, target, main_metric_name, id_set, Performances, parameters, version, list_ensemble_levels, ensemble_level)
        for fold in folds:
            PREDICTIONS[fold]['outer_fold_' + version] = np.nan
    else:
        PREDICTIONS_ENSEMBLE = {}
        for outer_fold in outer_folds:
            #take the subset of the rows that correspond to the outer_fold
            col_outer_fold = ensemble_outerfolds_cols[0]
            PREDICTIONS_outerfold = {}
            for fold in folds:
                PREDICTIONS[fold]['outer_fold_' + version] = PREDICTIONS[fold][col_outer_fold]
                PREDICTIONS_outerfold[fold] = PREDICTIONS[fold][PREDICTIONS[fold]['outer_fold_' + version] == float(outer_fold)]
            
            #build the ensemble model
            build_single_ensemble(PREDICTIONS_outerfold, target, main_metric_name, id_set, Performances, parameters, version, list_ensemble_levels, ensemble_level)
            
            #merge the predictions on each outer_fold
            for fold in folds:
                PREDICTIONS_outerfold[fold]['outer_fold_' + version]= float(outer_fold)
                if not fold in PREDICTIONS_ENSEMBLE.keys():
                    if ensemble_level == None:
                        PREDICTIONS_ENSEMBLE[fold] = PREDICTIONS_outerfold[fold][['eid', 'outer_fold_' + version, 'pred_' + version]]
                    else:
                        PREDICTIONS_ENSEMBLE[fold] = PREDICTIONS_outerfold[fold][['eid', 'outer_fold_' + version, 'pred_' + version, 'pred_' + version.replace('*', ','), 'pred_' + version.replace('*', '?')]]
                else:
                    if ensemble_level == None:
                        PREDICTIONS_ENSEMBLE[fold] = PREDICTIONS_ENSEMBLE[fold].append(PREDICTIONS_outerfold[fold][['eid', 'outer_fold_' + version, 'pred_' + version]])
                    else:
                        PREDICTIONS_ENSEMBLE[fold] = PREDICTIONS_ENSEMBLE[fold].append(PREDICTIONS_outerfold[fold][['eid', 'outer_fold_' + version, 'pred_' + version, 'pred_' + version.replace('*', ','), 'pred_' + version.replace('*', '?')]])
        
        #Add the ensemble predictions to the dataframe
        for fold in folds:
            if fold == 'train':
                PREDICTIONS[fold] = PREDICTIONS[fold].merge(PREDICTIONS_ENSEMBLE[fold], how='outer', on =['eid', 'outer_fold_' + version])
            else:
                PREDICTIONS_ENSEMBLE[fold].drop('outer_fold_' + version, axis=1, inplace=True)
                PREDICTIONS[fold] = PREDICTIONS[fold].merge(PREDICTIONS_ENSEMBLE[fold], how='outer', on =['eid'])
    
    #build and save a dataset for this specific ensemble model
    for ensemble_type in ['*', ',', '?']:
        version_type = version.replace('*', ensemble_type)
        if 'pred_' + version_type in PREDICTIONS[fold].columns.values:
            for fold in folds:
                df_single_ensemble = PREDICTIONS[fold][['eid', 'outer_fold_' + version, 'pred_' + version_type]]
                df_single_ensemble.rename(columns={'outer_fold_' + version: 'outer_fold', 'pred_' + version_type: 'pred'}, inplace=True)
                df_single_ensemble.dropna(inplace=True, subset=['pred'])
                df_single_ensemble.to_csv(path_store + 'Predictions_' + version_type + '_' + fold + '_' + id_set +'.csv', index=False)

def recursive_ensemble_builder(PREDICTIONS, target, main_metric_name, id_set, Performances_grandparent, parameters_parent, version_parent, list_ensemble_levels_parent):   
    #Compute the ensemble models for the children first, so that they can be used for the parent model
    Performances_parent = Performances_grandparent[Performances_grandparent['version'].isin(fnmatch.filter(Performances_grandparent['version'], version_parent))]
    #if the last ensemble level has not been reached, go down one level and create a branch for each child. Otherwise the leaf has been reached
    if len(list_ensemble_levels_parent) > 0:
        list_ensemble_levels_child = list_ensemble_levels_parent.copy()
        ensemble_level = list_ensemble_levels_child.pop()
        list_children = Performances_parent[ensemble_level].unique()
        for child in list_children:
            parameters_child = parameters_parent.copy()
            parameters_child[ensemble_level] = child
            version_child = parameters_to_version(parameters_child)
            #recursive call to the function
            recursive_ensemble_builder(PREDICTIONS, target, main_metric_name, id_set, Performances_parent, parameters_child, version_child, list_ensemble_levels_child)
    else:
        ensemble_level = None
    
    #compute the ensemble model for the parent
    print('Building the ensemble model ' + version_parent)
    build_single_ensemble_wrapper(PREDICTIONS, target, main_metric_name, id_set, Performances_parent, parameters_parent, version_parent, list_ensemble_levels_parent, ensemble_level)

def bootstrap_correlations(data, n_bootstrap_iterations):
    names = data.columns.values
    results = []
    for i in range(n_bootstrap_iterations):
        if (i % 100 == 0):
            print('Bootstrap iteration ' + str(i) + ' out of ' + str(n_bootstrap_iterations))
        data_i = resample(data, replace=True, n_samples=len(data.index))
        results.append(np.array(data_i.corr()))
    results = np.array(results)
    for op in ['mean', 'std']:
        results_op = pd.DataFrame(getattr(np, op)(results, axis=0))
        results_op.index = names
        results_op.columns = names
        globals()['results_' + op] = results_op
    return results_mean, results_std


### PARAMETERS THAT DEPEND ON FUNCTIONS

#dict of Keras and sklearn losses and metrics
dict_metrics={}
dict_metrics['MSE']={'Keras':'mean_squared_error', 'sklearn':mean_squared_error}
dict_metrics['RMSE']={'Keras':RMSE_K, 'sklearn':rmse}
dict_metrics['R-Squared']={'Keras':R2_K, 'sklearn':r2_score}
dict_metrics['Binary-Crossentropy']={'Keras':'binary_crossentropy', 'sklearn':log_loss}
dict_metrics['ROC-AUC']={'Keras':ROC_AUC_K, 'sklearn':roc_auc_score}
dict_metrics['F1-Score']={'Keras':F1_K, 'sklearn':f1_score}
dict_metrics['PR-AUC']={'Keras':PR_AUC_K, 'sklearn':average_precision_score}
dict_metrics['Binary-Accuracy']={'Keras':'binary_accuracy', 'sklearn':accuracy_score}
dict_metrics['Sensitivity']={'Keras':sensitivity_K, 'sklearn':sensitivity_score}
dict_metrics['Specificity']={'Keras':specificity_K, 'sklearn':specificity_score}
dict_metrics['Precision']={'Keras':precision_K, 'sklearn':precision_score}
dict_metrics['Recall']={'Keras':recall_K, 'sklearn':recall_score}
dict_metrics['True-Positives']={'Keras':true_positives_K, 'sklearn':true_positives_score}
dict_metrics['False-Positives']={'Keras':false_positives_K, 'sklearn':false_positives_score}
dict_metrics['False-Negatives']={'Keras':false_negatives_K, 'sklearn':false_negatives_score}
dict_metrics['True-Negatives']={'Keras':true_negatives_K, 'sklearn':true_negatives_score}

