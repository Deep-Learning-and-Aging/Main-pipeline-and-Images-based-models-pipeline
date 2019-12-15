#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:57:11 2019

@author: Alan
"""

import innvestigate
import innvestigate.utils
import keras.applications.vgg16 as vgg16
import sys
import cv2

if len(sys.argv)==1:
    sys.argv.append('Liver') #image_type
    sys.argv.append('Sex') #target
    sys.argv.append('VGG16') #model_name
    sys.argv.append('Adam') #optimizer
    sys.argv.append('0.0001') #learning_rate
    sys.argv.append('0.0') #lam regularization: weight shrinking
    sys.argv.append('0.0') #dropout

#read parameters from command
image_type = sys.argv[1]
target = sys.argv[2]
model_name = sys.argv[3]
optimizer_name = sys.argv[4]
learning_rate = float(sys.argv[5])
lam = float(sys.argv[6])
dropout_rate = float(sys.argv[7])

#load libraries, import functions and import parameters (nested import in the line below)
from MI_helper_parameters import *

#set other parameters accordingly
image_size = input_size_models[model_name]
prediction_type = 'saliency'
#loss = dict_losses[target]
#metrics = dict_metrics[prediction_type]
#main_metric = main_metrics[target]
version = target + '_' + image_type + '_' + model_name + '_' + optimizer_name + '_' + str(learning_rate) + '_' + str(lam) + '_' + str(dropout_rate) + '_' + str(batch_size)
dir_images = dict_dir_images[image_type]

#load data features
DATA_FEATURES = {}
for fold in folds:
    DATA_FEATURES[fold] = pd.read_csv(path_store + 'data_features_' + image_type + '_' + target + '_' + fold + '.csv')

#generate the data generators
datagen_train = ImageDataGenerator(rescale=1./255., rotation_range=20, width_shift_range=0.2, height_shift_range=0.2)
datagen_test = ImageDataGenerator(rescale=1./255.)
class_mode_train = 'raw'
class_mode_test = None
GENERATORS = {}
STEP_SIZES = {}
for fold in folds:
    if fold == 'test':
        datagen = datagen_test
        class_mode = class_mode_test
    else:
        datagen = datagen_train
        class_mode = class_mode_train
    #
    # define data generator
    generator_fold = datagen.flow_from_dataframe(
        dataframe=DATA_FEATURES[fold],
        directory=dir_images,
        x_col='eid',
        y_col=target,
        color_mode='rgb',
        batch_size=batch_size,
        seed=0,
        shuffle=True,
        class_mode='raw',
        target_size=(image_size, image_size))
    #
    # assign variables to their names
    GENERATORS[fold] = generator_fold
    STEP_SIZES[fold] = generator_fold.n // generator_fold.batch_size


#define the model
x, base_model_input = generate_base_model(model_name=model_name, lam=lam, dropout_rate=dropout_rate, import_weights=import_weights)
model = complete_architecture(x=x, input_shape=base_model_input, activation=dict_activations[prediction_type], lam=lam, dropout_rate=dropout_rate)
#set_learning_rate(model=model, optimizer_name=optimizer_name, learning_rate=learning_rate, loss=dict_losses[prediction_type], metrics=[dict_metrics_functions[metric] for metric in dict_metrics[prediction_type]])

#load weights
path_weights = path_store + 'model_weights_' + version + '.h5'
model.load_weights(path_weights)

#pop last layer (unclear why) + dimensions don't work
#model.layers.pop()

# Create analyzer
analyzer = innvestigate.create_analyzer("gradient", model)

batch_number = 0
n_images = 10
image = GENERATORS['test'].__getitem__(0)[batch_number][0:n_images,:,:,:]

a = analyzer.analyze(image)

# Aggregate along color channels and normalize to [-1, 1]
a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
a /= np.max(np.abs(a))
cv2.imwrite(path_store + '../figures/test.jpg', a[0])

# Plot
plt.imshow(a[0], cmap="seismic", clim=(-1, 1))










#DATA_FEATURES = {}
#for fold in folds:
#    DATA_FEATURES[fold] = pd.read_csv(path_store + 'data_features_' + image_type + '_' + target + '_' + fold + '.csv')

#data_test = pd.read_csv(path_store + 'data_features_' + image_type + '_' + target + '_' + 'test' + '.csv', index_col=0)
#image = np.load(dir_images + data_test.index.values[0])



# Get model
#model = vgg16.VGG16()
#preprocess = vgg16.preprocess_input

# Strip softmax layer
model = innvestigate.utils.model_wo_softmax(model)

# Create analyzer
analyzer = innvestigate.create_analyzer("gradient", model)

# Add batch axis and preprocess
x = preprocess(image[None])
# Apply analyzer w.r.t. maximum activated output-neuron
a = analyzer.analyze(x)

# Aggregate along color channels and normalize to [-1, 1]
a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
a /= np.max(np.abs(a))
# Plot
plt.imshow(a[0], cmap="seismic", clim=(-1, 1))

