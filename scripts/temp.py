#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:22:08 2019

@author: Alan
"""

"""Determine which weights to load, if any.
The tricky part is that for regularized models, we can initialize the weights 
with the values of the nonregularized model if the actual weights are not 
available."""
path_weights = path_store + 'model_weights_' + version + '.h5'
regularized_model = False if ((weight_decay == 0) & (dropout_rate == 0)) else True
path_weights_nonreg = path_store + 'model_weights_' + target + '_' + image_type + '_' + transformation + '_' + architecture + '_' + optimizer + '_' + str(learning_rate) + '_' + str(0.0) + '_' + str(0.0) + str(outer_fold) + '.h5' if regularized_model else ""
if continue_training:
    if os.path.exists(path_weights):
        keras_weights = None
        path_load_weights = path_weights
    else:
        if regularized_model & os.path.exists(path_weights_nonreg):
            keras_weights = None
            path_load_weights = path_weights_nonreg
        else:
            keras_weights = 'imagenet'
else:
    if regularized_model & os.path.exists(path_weights_nonreg):
        keras_weights = None
        path_load_weights = path_weights_nonreg
    else:
        keras_weights = 'imagenet'


def weights_for_transferlearning(continue_training, max_transfer_learning, path_weights, path_weights_nonreg, regularized_model):
    #load weights is weights from a similar model are available
    if max_transfer_learning:
        if continue_training and os.path.exists(path_weights):
            return None, path_weights
        else:
            #in thos order test if
            #1-same architecture non-regularized exists ()
            #2-same model used to predict other phenotype exists (e.g)
            #3-same model built on different transformation exists (e.g raw vs contrast)
            #4-same model built on a different view exists (e.g 2chambers vs 3 chambers)
            #5-same model built with the same organ but different field ID exists
            #6-same model built on a different organ exists
            #7-if nothing else worked, just use keras weights
            if 
    #only load weights from previous training iteration of the same model
    else:
        if continue_training and os.path.exists(path_weights):
            return None, path_weights
        else:
            return'imagenet', 'load_path_weights_should_not_be_used'


        
    #return path_load_weights, keras_weights, 
    
    
    
    
    
    
    
def weights_for_transferlearning(continue_training, max_transfer_learning, path_weights, path_weights_nonreg, regularized_model):
    
    if continue_training: and os.path.exists(path_weights):
        return None, path_weights
    else:
        if max_transfer_learning & continue_training: #max_transfer_learning can only be used if continue_training is on. 
            
            #1
            #2
            
            
            
            
        #if not max_transfer_learning or no similar model was found, use imagenet weights    
        return 'imagenet', 'load_path_weights_should_not_be_used'

    
    
    
    
    
    
    
    
    
    
    
    
version = 'Sex_Heart_20208_4chambers_raw_Xception_Adam_0.0001_0.0_0.0'



dict_alternative_targets_for_transfer_learning={'Age':['Age', 'Sex'], 'Sex':['Sex', 'Age']}

continue_training = True
max_transfer_learning = True
path_weights = '../data/model-weights_Sex_Heart_20208_4chambers_raw_Xception_Adam_0.0001_0.0_0.0_0.h5'
path_load_weights, keras_weights= weights_for_transfer_learning(continue_training=continue_training, max_transfer_learning=max_transfer_learning, path_weights=path_weights, list_parameters_to_match = ['organ', 'transformation', 'field_id', 'view'])

#WORK HERE

    
    
    
    if continue_training and os.path.exists(path_weights):
        return None, path_weights
    
    #0-same architecture has already been trained
    #1-same architecture non-regularized exists
    #2-same model used to predict other phenotype exists (e.g)
    if max_transfer_learning:
        
        #0-if same architecture has already been trained
        if os.path.exists(path_weights):
            return None, path_weights
        
#1-if same architecture with different regularization exists
        Performances = pd.read_csv(path_store + 'Performances_ranked_' + parameters['target'] + '_' + 'val' + '_' + dict_eids_version[parameters['organ']] + '.csv')
        Performances['field_id'] = Performances['field_id'].astype(str)
        for parameter in ['target', 'organ', 'field_id', 'view', 'transformation', 'architecture']:
            print(parameter)
            Performances = Performances[Performances[parameter] == parameters[parameter]]
            print(Performances)
        if(len(Performances.index) != 0):
            path_weights_to_load = path_store + 'model-weights_' + Performances['version'][0] + '.h5'
            return None, path_weights_to_load

        #2-if same model used to predict other phenotype exists
dict_alternative_targets_for_transfer_learning={'Age':['Age', 'Sex'], 'Sex':['Sex', 'Age']}
parameters_to_match = parameters.copy()
parameters_to_match['target'] = dict_alternative_target_for_transfer_learning[parameters['target']]
Performances = pd.read_csv(path_store + 'Performances_ranked_' + parameters_to_match['target'] + '_' + 'val' + '_' + dict_eids_version[parameters['organ']] + '.csv')
Performances['field_id'] = Performances['field_id'].astype(str)
for parameter in ['target', 'organ', 'field_id', 'view', 'transformation', 'architecture']:
    print(parameter)
    Performances = Performances[Performances[parameter] == parameters_to_mach[parameter]]
    print(Performances)
if(len(Performances.index) != 0):
    path_weights_to_load = path_store + 'model-weights_' + Performances['version'][0] + '.h5'
    return None, path_weights_to_load
        
        #3-same model built on different transformation exists (e.g raw vs contrast)
        if 
        
        #4-same model built on a different view exists (e.g 2chambers vs 3 chambers)
        if
        
        #5-same model built with the same organ but different field ID exists
        if
        
        #6-same model built on a different organ exists (e.g heart and liver)
        if
        
        
    #Otherwise use imagenet weights to initialize
    return 'imagenet', 'load_path_weights_should_not_be_used'
        
        
"""continue_training is a special case of transfer learning: transfering weights
 from the last training iteration. If some weights are already available but
 were not used in the "continue_training and os.path.exists(path_weights)" then
 using max_transfer_learning from a  """
        
            #1
            #2
            
            
            
            
        #if not max_transfer_learning or no similar model was found, use imagenet weights    
        return 'imagenet', 'load_path_weights_should_not_be_used'
    
    
    
    #load weights is weights from a similar model are available
    
    
    
    if max_transfer_learning:
        if continue_training and os.path.exists(path_weights):
            return None, path_weights
        else:
            #in thos order test if

            #7-if nothing else worked, just use keras weights
            if 
    #only load weights from previous training iteration of the same model
    else:
        if continue_training and os.path.exists(path_weights):
            return None, path_weights
        else:
            return'imagenet', 'load_path_weights_should_not_be_used'




GENERATORS = {}
STEP_SIZES = {}
for fold in folds:
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

    