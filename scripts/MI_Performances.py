#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 20:31:21 2019

@author: Alan
"""

def split_model_name_to_parameters(model_name):
    parameters={}
    parameters_list = model_name.split('_')
    parameters['target'] = parameters_list[0]
    parameters['']
    return parameters

#load libraries, import functions and import parameters (nested import in the line below)
from MI_helpers import *

#options
#debunk mode: exclude train set
debunk_mode = True
#generate training plots
generate_training_plots = True
#save performances
save_performances = True

#default parameters. use '*' to not exclude any model based on the parameter
if len(sys.argv) != 11:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age') #target
    sys.argv.append('Heart_20208_4chambers') #image_type
    sys.argv.append('raw') #preprocessing
    sys.argv.append('*') #architecture
    sys.argv.append('*') #optimizer
    sys.argv.append('*') #learning_rate
    sys.argv.append('*') #weight decay
    sys.argv.append('*') #dropout

#read parameters from command
image_type, organ, field_id, view, preprocessing, target, architecture, optimizer, learning_rate, weight_decay, dropout_rate = read_parameters_from_command(sys.argv)

#set other parameters accordingly
version = target + '_' + image_type + '_' + preprocessing + '_' + architecture + '_' + optimizer + '_' + str(learning_rate) + '_' + str(weight_decay) + '_' + str(dropout_rate)
dir_images = dir_images = path_store + '../images/' + organ + '/' + field_id + '/' + view + '/' + preprocessing + '/'
functions_version = 'sklearn' #use 'Keras' for functions during training, and 'sklearn' for functions during testing
prediction_type = dict_prediction_types[target]
metrics_names = [dict_losses_names[prediction_type]] + dict_metrics_names[prediction_type]
metrics = [dict_metrics[metric_name][functions_version] for metric_name in metrics_names]
main_metric_name=dict_main_metrics_names[target]

#debunk mode: exclude train set
if debunk_mode:
    folds = ['val', 'test']

#double the batch size for the teslaM40 cores that have bigger memory
GPUs = GPUtil.getGPUs()
double_batch_size = True if GPUs[0].memoryTotal > 20000 else False

#configure gpus
configure_gpus()

#load data_features
DATA_FEATURES = load_data_features(folds=folds, path_store=path_store, image_type=image_type, target=target)
Ys={}
for fold in folds:
    Ys[fold] = DATA_FEATURES[fold][target]

#Select all models matching the parameters requested
list_model_weights = glob.glob(path_store + 'model-weights_' + version + '*.h5')
list_model_weights.sort()
list_models_available = [os.path.splitext(model_name)[0].split('_')[1:] for model_name in list_model_weights]
list_versions = ['_'.join(parameters_list) for parameters_list in list_models_available]

#Define Predictions dataframe
PREDICTIONS={}
for fold in folds:
    PREDICTIONS[fold] = pd.DataFrame(index=[os.path.splitext(id)[0] for id in DATA_FEATURES[fold]['eid']], columns=list_versions)

#Define Performances dataframe
Performances = pd.DataFrame.from_records(list_models_available)
Performances.columns = ['target', 'organ', 'field_id', 'view', 'preprocessing', 'architecture', 'optimizer', 'learning_rate', 'weight_decay', 'dropout_rate']
Performances[['learning_rate', 'weight_decay', 'dropout_rate']] = Performances[['learning_rate', 'weight_decay', 'dropout_rate']].apply(pd.to_numeric)
Performances['backup_used'] = False

#add columns for metrics
for metric_name in metrics_names:
    for fold in folds:
        Performances[metric_name + '_' + fold] = np.nan

#take subset of models to explore based on conditions in the input
for parameter in ['architecture', 'learning_rate', 'weight_decay', 'dropout_rate']:
    parameter_value = globals()[parameter]
    if parameter_value != '*':
        Performances = Performances[Performances[parameter] == parameter_value]

#load the architecture
list_architectures = Performances['architecture'].unique()

#Evaluate the performances of each model, architecture by architecture
for architecture in list_architectures:
    print('Starting models\' evaluation for architecture ' + architecture + '...')
    image_size = input_size_models[architecture]
    batch_size = 2*dict_batch_sizes[architecture] if double_batch_size else dict_batch_sizes[architecture]
    Performances_architecture = Performances[Performances['architecture'] == architecture]
    GENERATORS, STEP_SIZES = generate_generators(DATA_FEATURES=DATA_FEATURES, target=target, dir_images=dir_images, image_size=image_size, batch_size=batch_size, folds=folds, seed=seed, mode='model_testing')
    
    #for each architecture, test the different weights
    for i, model_row in Performances_architecture.iterrows():
        x, base_model_input = generate_base_model(architecture=architecture, weight_decay=0, dropout_rate=0, import_weights=None)
        model = complete_architecture(x=x, input_shape=base_model_input, activation=dict_activations[prediction_type], weight_decay=model_row['weight_decay'], dropout_rate=model_row['dropout_rate'])
        model_version = model_row['target'] + '_' + model_row['organ'] + '_' + model_row['field_id'] + '_' + model_row['view'] + '_' + model_row['preprocessing'] + '_' + model_row['architecture'] + '_' + model_row['optimizer'] + '_' + str(model_row['learning_rate']) + '_' + str(model_row['weight_decay']) + '_' + str(model_row['dropout_rate'])
        try:
            model.load_weights(path_store + 'model-weights_' + model_version + '.h5')
        except:
            #if the weights are corrupted, load the backup weights.
            try:
                model.load_weights(path_store + 'backup-model-weights_' + model_version + '.h5')
                Performances_architecture.loc[i, 'backup_used'] = True
                print('THE FILE FOR THE WEIGHTS ' + model_version + ' COULD NOT BE OPENED. USING THE BACKUP INSTEAD.')
            except:
                print('NEITHER THE NORMAL NOR THE BACKUP FILE FOR THE WEIGHTS ' + model_version + ' COULD BE OPENED. MOVING ON TO THE NEXT MODEL.')
                break
        
        #plot the training from the logger
        if generate_training_plots:
            plot_training(path_store=path_store, version=model_version, display_learning_rate=True)
        
        #for each fold and for each metric, compute the model's performance
        for fold in folds:
            pred=model.predict_generator(GENERATORS[fold], steps=STEP_SIZES[fold], verbose=1).squeeze()
            try:
                PREDICTIONS[fold][model_version] = pred
                #convert to pred class?
                for metric_name in metrics_names:
                    Performances_architecture.loc[i, metric_name + '_' + fold] = dict_metrics[metric_name][functions_version](Ys[fold], pred)
            except:
                print("Mismatch between length of pred and y")

    
    #Record architecture's results in general dataframe before printing them
    Performances[Performances['architecture'] == architecture] = Performances_architecture
    print('Completed evaluation for architecture ' + architecture + ". Results below:")
    print(Performances_architecture)

#Ranking, printing and saving
print('Performances of the models ranked by models\'names:')
print(Performances)
Performances_sorted = Performances.sort_values(by=main_metric_name + '_val', ascending=main_metrics_modes[main_metric_name] == 'min')
print('Performances of the models ranked by validation score on the main metric:')
print(Performances_sorted)
if save_performances:
    Performances.to_csv(path_store + 'Performances_alphabetical_' + version + '.csv', index=False)
    Performances_sorted.to_csv(path_store + 'Performances_ranked_' + version + '.csv', index=False)
#TODO save predictions
if save_predictions:
    

    
    
    
try:
    PREDICTIONS[fold][model_version] = pred
except:
    print("Mismatch between length of pred and y")