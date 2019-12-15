#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 00:16:14 2019

@author: Alan
"""

from MI_helper_libraries import *


#functions to put in helper file
def append_ext(fn):
    return fn+".jpg"

def generate_base_model(model_name, lam, dropout_rate, import_weights):
    if model_name in ['VGG16', 'VGG19']:
        if model_name == 'VGG16':
            from keras.applications.vgg16 import VGG16
            base_model = VGG16(include_top=False, weights=import_weights, input_shape=(224,224,3))
        elif model_name == 'VGG19':
            from keras.applications.vgg19 import VGG19
            base_model = VGG19(include_top=False, weights=import_weights, input_shape=(224,224,3))
        x = base_model.output
        x = Flatten()(x)
        x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(lam))(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(lam))(x)
        x = Dropout(dropout_rate)(x) 
    elif model_name in ['MobileNet', 'MobileNetV2']:
        if model_name == 'MobileNet':
            from keras.applications.mobilenet import MobileNet
            base_model = MobileNet(include_top=False, weights=import_weights, input_shape=(224,224,3))
        elif model_name == 'MobileNetV2':
            from keras.applications.mobilenet_v2 import MobileNetV2
            base_model = MobileNetV2(include_top=False, weights=import_weights, input_shape=(224,224,3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
    elif model_name in ['DenseNet121', 'DenseNet169', 'DenseNet201']:
        if model_name == 'DenseNet121':
            from keras.applications.densenet import DenseNet121
            base_model = DenseNet121(include_top=True, weights=import_weights, input_shape=(224,224,3))
        elif model_name == 'DenseNet169':
            from keras.applications.densenet import DenseNet169
            base_model = DenseNet169(include_top=True, weights=import_weights, input_shape=(224,224,3))
        elif model_name == 'DenseNet201':
            from keras.applications.densenet import DenseNet201
            base_model = DenseNet201(include_top=True, weights=import_weights, input_shape=(224,224,3))            
        base_model = Model(base_model.inputs, base_model.layers[-2].output)
        x = base_model.output
    elif model_name in ['NASNetMobile', 'NASNetLarge']:
        if model_name == 'NASNetMobile':
            from keras.applications.nasnet import NASNetMobile
            base_model = NASNetMobile(include_top=True, weights=import_weights, input_shape=(224,224,3))
        elif model_name == 'NASNetLarge':
            from keras.applications.nasnet import NASNetLarge
            base_model = NASNetLarge(include_top=True, weights=import_weights, input_shape=(331,331,3))
        base_model = Model(base_model.inputs, base_model.layers[-2].output)
        x = base_model.output
    elif model_name == 'Xception':
        from keras.applications.xception import Xception
        base_model = Xception(include_top=False, weights=import_weights, input_shape=(299,299,3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
    elif model_name == 'InceptionV3':
        from keras.applications.inception_v3 import InceptionV3
        base_model = InceptionV3(include_top=False, weights=import_weights, input_shape=(299,299,3))
        x = base_model.output        
        x = GlobalAveragePooling2D()(x)
    elif model_name == 'InceptionResNetV2':
        from keras.applications.inception_resnet_v2 import InceptionResNetV2
        base_model = InceptionResNetV2(include_top=False, weights=import_weights, input_shape=(299,299,3))
        x = base_model.output        
        x = GlobalAveragePooling2D()(x)
    return x, base_model.input

def complete_architecture(x, input_shape, activation, lam, dropout_rate):
    x = Dense(1024, activation='selu', kernel_regularizer=regularizers.l2(lam))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(512, activation='selu', kernel_regularizer=regularizers.l2(lam))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256, activation='selu', kernel_regularizer=regularizers.l2(lam))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='selu', kernel_regularizer=regularizers.l2(lam))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='selu', kernel_regularizer=regularizers.l2(lam))(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(1, activation=activation)(x)
    model = Model(inputs=input_shape, outputs=predictions)
    return model

def R_squared(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
  
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def set_learning_rate(model, optimizer_name, learning_rate, loss, metrics):
    opt = globals()[optimizer_name](lr=learning_rate, clipnorm=1.0)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    
#def initialize_history(metrics, folds_tune):
#    HISTORY = {}
#    for metric in ['loss'] + metrics:
#        for fold in folds_tune:
#            if(fold=='train'):
#                HISTORY[metric] = []
#            else:
#                HISTORY[fold + '_' + metric] = []
#    return HISTORY
#
#def update_history(HISTORY, history, folds_tune, metrics):
#    for metric in ['loss'] + metrics:
#        HISTORY[metric] = HISTORY[metric] + history.history[metric]
#        HISTORY['val_' + metric] = HISTORY['val_' + metric] + history.history['val_' + metric]
#    return HISTORY
    
def plot_training(HISTORY, metrics, version):
    keys = HISTORY.keys()
    fig, axs = plt.subplots(1, int(len(keys)/2), sharey=False, sharex=True)
    fig.set_figwidth(15)
    fig.set_figheight(5)
    epochs = np.array(range(len(HISTORY[metrics[0]])))
    for i, metric in enumerate(['loss'] + metrics):
        for key in [key for key in keys if metric in key][::-1]:
            axs[i].plot(epochs, HISTORY[key])
        axs[i].legend(['Training ' + metric, 'Validation ' + metric])
        axs[i].set_title(metric + ' = f(Epoch)')
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(metric)
        axs[i].set_ylim((-0.2, 1.1))
    #save figure as pdf
    fig.savefig("../figures/Training_" + version + '.pdf', bbox_inches='tight')
    #close
    plt.close('all')
    
def save_model_weights(model, path_store, version):
    model.save_weights(path_store + "model_weights_" + version + ".h5")
    print("Model's best weights for "+ version + " were saved.")
    
def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

def generate_predictions_and_performances(model, target, DATA_FEATURES, GENERATORS, STEPSIZES, folds, metrics, metric_functions):
    PREDS={}
    PERFORMANCES={}
    for fold in folds:
        generator = GENERATORS[fold]
        generator.reset()
        PREDS[fold]=model.predict_generator(generator, STEPSIZES[fold], verbose=1)
    for metric in metrics:
            PERFORMANCES[metric] = {}
            for fold in folds:
                PERFORMANCES[metric][fold] = metric_functions[metric](DATA_FEATURES[fold][target], PREDS[fold])
    return PREDS, PERFORMANCES
