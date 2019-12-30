#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:29:53 2019

@author: Alan
"""

#load libraries, import functions and import parameters (nested import in the line below)
from MI_helpers import *

#options
#debug mode: exclude train set
debug_mode = True
#generate training plots
generate_training_plots = False
#regenerate predictions if already exist TODO
regenerate_predictions = True
#save_predictions
save_predictions = True

#default parameters
if len(sys.argv) != 9:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age') #target
    sys.argv.append('Liver_20204_main') #image_type, e.g PhysicalActivity_90001_main, Liver_20204_main or Heart_20208_3chambers
    sys.argv.append('contrast') #transformation
    sys.argv.append('NASNetLarge') #architecture
    sys.argv.append('Adam') #optimizer
    sys.argv.append('0.0001') #learning_rate
    sys.argv.append('0.0') #weight decay
    sys.argv.append('0.0') #dropout

#read parameters from command
target, image_type, organ, field_id, view, preprocessing, architecture, optimizer, learning_rate, weight_decay, dropout_rate, outer_fold = read_parameters_from_command(sys.argv)

#set other parameters accordingly
version = target + '_' + image_type + '_' + preprocessing + '_' + architecture + '_' + optimizer + '_' + str(learning_rate) + '_' + str(weight_decay) + '_' + str(dropout_rate)
dir_images = dir_images = path_store + '../images/' + organ + '/' + field_id + '/' + view + '/' + preprocessing + '/'
prediction_type = dict_prediction_types[target]
image_size = input_size_models[architecture]

#debug mode: exclude train set
if debug_mode:
    folds = ['val', 'test']

#double the batch size for the teslaM40 cores that have bigger memory
GPUs = GPUtil.getGPUs()
double_batch_size = True if GPUs[0].memoryTotal > 20000 else False
batch_size = 2*dict_batch_sizes[architecture] if double_batch_size else dict_batch_sizes[architecture]

#configure gpus
configure_gpus()

#load model's architecture
x, base_model_input = generate_base_model(architecture=architecture, weight_decay=weight_decay, dropout_rate=dropout_rate, keras_weights=None)
model = complete_architecture(x=x, input_shape=base_model_input, activation=dict_activations[prediction_type], weight_decay=weight_decay, dropout_rate=dropout_rate)

print('Starting model evaluation for version ' + version + '...')

# Define Predictions dataframe dictionary
PREDICTIONS={}
for outer_fold in outer_folds:
    print('Predicting samples for the outer_fold = ' + outer_fold)
    # load data_features, 
    DATA_FEATURES = load_data_features(path_store=path_store, image_field=dict_image_field_to_ids[organ + '_' + field_id], target=dict_target_to_ids[target], folds=['train', 'val', 'test'], outer_fold=outer_fold)
    # If regression target: calculate the mean and std of the target
    if target in targets_regression:
        mean_train = DATA_FEATURES['train'][target+'_raw'].mean()
        std_train = DATA_FEATURES['train'][target+'_raw'].std()
    # split the samples into two groups: what can fit into the batch size, and the leftovers.
    DATA_FEATURES_BATCH = {}
    DATA_FEATURES_LEFTOVERS = {}
    for fold in folds:
        n_leftovers = len(DATA_FEATURES[fold].index) % batch_size
        DATA_FEATURES_BATCH[fold] = DATA_FEATURES[fold].iloc[:-n_leftovers]
        DATA_FEATURES_LEFTOVERS[fold] = DATA_FEATURES[fold].tail(n_leftovers)
    
    # generate the generators
    GENERATORS_BATCH, STEP_SIZES_BATCH = generate_generators(DATA_FEATURES=DATA_FEATURES_BATCH, target=target, dir_images=dir_images, image_size=image_size, batch_size=batch_size, folds=folds, seed=seed, mode='model_testing')
    GENERATORS_LEFTOVERS, STEP_SIZES_LEFTOVERS = generate_generators(DATA_FEATURES=DATA_FEATURES_LEFTOVERS, target=target, dir_images=dir_images, image_size=image_size, batch_size=batch_size, folds=folds, seed=seed, mode='model_testing')
    
    # load the weights
    model_version = version + '_' + outer_fold
    try:
        model.load_weights(path_store + 'model-weights_' + model_version + '.h5')
    except:
        #if the weights are corrupted, load the backup weights.
        try:
            model.load_weights(path_store + 'backup-model-weights_' + model_version + '.h5')
            print('THE FILE FOR THE WEIGHTS ' + model_version + ' COULD NOT BE OPENED. USING THE BACKUP INSTEAD.')
        except:
            print('NEITHER THE NORMAL NOR THE BACKUP FILE FOR THE WEIGHTS ' + model_version + ' COULD BE OPENED. MOVING ON TO THE NEXT MODEL.')
            break
    
    # Generate predictions
    for fold in folds:
        print('Predicting the samples in the fold: ' + fold)
        pred_batch = model.predict_generator(GENERATORS_BATCH[fold], steps=STEP_SIZES_BATCH[fold], verbose=0)
        pred_leftovers = model.predict_generator(GENERATORS_LEFTOVERS[fold], steps=STEP_SIZES_LEFTOVERS[fold], verbose=0)
        pred_full = np.concatenate((pred_batch, pred_leftovers)).squeeze()
        if target in targets_regression:
            pred_full = pred_full*std_train + mean_train
        DATA_FEATURES[fold]['Pred_' + version] = pred_full
        if fold in PREDICTIONS.keys():
            PREDICTIONS[fold] = pd.concat([PREDICTIONS[fold], DATA_FEATURES[fold]])
        else:
            PREDICTIONS[fold] = DATA_FEATURES[fold]
    
    # plot the training from the logger
    if generate_training_plots:
        plot_training(path_store=path_store, version=model_version, display_learning_rate=True)

# save predictions
if save_predictions:
    for fold in folds:
        PREDICTIONS[fold][['eid', 'outer_fold', 'Pred_' + version]].to_csv(path_store + 'Predictions_' + version + '_' + fold + '.csv', index=False)

print("Done")