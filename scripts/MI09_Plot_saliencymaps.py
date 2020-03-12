#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:00:08 2020

@author: Alan
"""

#load libraries, import functions and import parameters (nested import in the line below)
from MI_helpers import *

if len(sys.argv) != 5:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age') #target
    sys.argv.append('Heart_20208_4chambers') #organ_id_view
    sys.argv.append('raw') #transformation
    sys.argv.append('B') #id_set

#read parameters from command
target = sys.argv[1]
organ_id_view = sys.argv[2]
transformation = sys.argv[3]
id_set = sys.argv[4]
organ, field_id, view = organ_id_view.split('_')

#Pick the best model based on the performances
Performances = pd.read_csv(path_store + 'PERFORMANCES_withoutEnsembles_ranked_' + target + '_test_' + id_set + '.csv').set_index('version', drop=False)
Performances = Performances[(Performances['organ'] == organ) & (Performances['field_id'].astype(str) == field_id) & (Performances['view'] == view) & (Performances['transformation'] == transformation)]
version = Performances['version'].values[0]
parameters = version_to_parameters(version, names_model_parameters)
image_size = input_size_models[parameters['architecture']]
prediction_type = dict_prediction_types[target]
batch_size = dict_batch_sizes[parameters['architecture']]
del Performances

#Load the model, using a linear final activation
x, base_model_input = generate_base_model(architecture=parameters['architecture'], weight_decay=float(parameters['weight_decay']), dropout_rate=float(parameters['dropout_rate']), keras_weights=None)
model = complete_architecture(x=x, input_shape=base_model_input, activation='linear', weight_decay=float(parameters['weight_decay']), dropout_rate=float(parameters['dropout_rate']))
penultimate_layer_idx = utils.find_layer_idx(model, dict_architecture_to_last_conv_layer_name[parameters['architecture']]) 

#Format the residuals
Residuals_full = pd.read_csv(path_store + 'RESIDUALS_' + target + '_' + 'test' + '_' + id_set + '.csv')
Residuals = Residuals_full[['eid', 'Age', 'Sex', 'res_' + version, 'outer_fold_' + version]]
del Residuals_full
Residuals.dropna(inplace=True)
Residuals.rename(columns={'res_' + version: 'res', 'outer_fold_' + version: 'outer_fold'}, inplace=True)
Residuals['eid'] = Residuals['eid'].astype(str).apply(append_ext)
Residuals['outer_fold'] = Residuals['outer_fold'].astype(int).astype(str)
Residuals['res_abs'] = Residuals['res'].abs()
Residuals = Residuals[['eid', 'outer_fold', 'Sex', 'Age', 'res', 'res_abs']]

#Select with samples to plot
Sexes = ['Male', 'Female']
dict_sexes_to_values = {'Male': 0, 'Female': 1}
for sex in Sexes:
    print('Analyzing sex: ' + sex)
    Residuals_sex = Residuals[Residuals['Sex'] == dict_sexes_to_values[sex]]
    Residuals_sex['Sex'] = sex
    for age_category in ['young', 'middle', 'old']:
        print('Analyzing age category: ' + age_category)
        if age_category == 'young':
            Residuals_age = Residuals_sex[Residuals_sex['Age'] <= Residuals_sex['Age'].min()+1]
        elif age_category == 'middle':
            Residuals_age = Residuals_sex[Residuals_sex['Age'] == int(Residuals_sex['Age'].median())]
        else:
            Residuals_age = Residuals_sex[Residuals_sex['Age'] >= Residuals_sex['Age'].max()-1]
        Residuals_age['age_category'] = age_category
        for aging_rate in ['accelerated', 'normal', 'decelerated']:
            print('Analyzing aging rate: ' + aging_rate)
            Residuals_ar = Residuals_age
            if aging_rate == 'accelerated':
                Residuals_ar.sort_values(by='res', ascending=False, inplace=True)
            elif aging_rate == 'decelerated':
                Residuals_ar.sort_values(by='res', ascending=True, inplace=True)
            else:
                Residuals_ar.sort_values(by='res_abs', ascending=True, inplace=True)
            Residuals_ar['aging_rate'] = aging_rate
            Residuals_ar = Residuals_ar.iloc[:N_samples_saliencymaps,]
            Residuals_ar['sample'] = range(len(Residuals_ar.index))
            if 'df_to_plot' not in globals():
                df_to_plot = Residuals_ar
            else:
                df_to_plot = df_to_plot.append(Residuals_ar)
df_to_plot['plot_title'] = 'Age = ' + df_to_plot['Age'].astype(str) + ', Predicted Age = ' + (df_to_plot['Age'] + df_to_plot['res']).round().astype(str) + ', Sex = ' + df_to_plot['Sex'] + ', sample ' + df_to_plot['sample'].astype(str)
df_to_plot['save_title'] = df_to_plot['Sex'] + '_' + df_to_plot['age_category'] + '_' + df_to_plot['aging_rate'] + '_' + df_to_plot['sample'].astype(str)
df_to_plot.head()
df_to_plot.to_csv(path_store + 'Saliency-samples_' + target + '_' + organ_id_view + '_' + transformation + '_' + id_set + '.csv', index=False)


#Analyze the images, outer_fold by outer_fold to minimize the time spent loading weights
for outer_fold in outer_folds:
    df_outer_fold = df_to_plot[df_to_plot['outer_fold'] == outer_fold]
    
    #generate the data generators
    datagen = ImageDataGenerator(rescale=1./255.)
    class_mode = None
    dir_images = '../images/' + parameters['organ'] + '/' + parameters['field_id'] + '/' + parameters['view'] + '/' + parameters['transformation'] + '/'
    generator = datagen.flow_from_dataframe(dataframe=df_outer_fold, directory=dir_images, x_col='eid', y_col='res', color_mode='rgb', batch_size=batch_size, seed=seed, shuffle=False, class_mode='raw', target_size=(image_size, image_size))
    step_size = generator.n // generator.batch_size
    
    #load the weights for the fold
    model.load_weights(path_store + 'model-weights_' + version + '_' + outer_fold + '.h5')
    
    #Generate analyzers
    saliency_analyzer = innvestigate.create_analyzer("gradient", model, allow_lambda_layers=True)
    guided_backprop_analyzer = innvestigate.create_analyzer("guided_backprop", model, allow_lambda_layers=True)
    
    #Generate the saliency maps
    n_images = len(df_outer_fold.index)
    for i in range(n_images//batch_size + 1):
        print(i)
        n_images_batch = np.min([batch_size, n_images - i*batch_size])
        images = generator.__getitem__(0)[i][:n_images_batch,:,:,:]
        saliencies = saliency_analyzer.analyze(images)
        guided_backprops = guided_backprop_analyzer.analyze(images)
        for j in range(saliencies.shape[0]):
            #select sample
            image = images[j]
            plot_title = df_outer_fold['plot_title'].values[i*batch_size + j]
            save_title = df_outer_fold['save_title'].values[i*batch_size + j]
    
            #generate the saliency map transparent filter
            saliency = saliencies[j]
            saliency = np.linalg.norm(saliency, axis=2)
            saliency = (saliency*255/saliency.max()).astype(int)
            r_ch = saliency.copy()
            g_ch = saliency.copy()*0
            b_ch = saliency.copy()*0
            a_ch = saliency*3
            saliency_filter = np.dstack((r_ch, g_ch, b_ch, a_ch))
            #generate plot
            plot_visualization_map(image=image, filter_map=saliency_filter, plot_title=plot_title, save_title = 'Saliency_' + save_title)
    
            #generate the grad-cam transparent filter
            grad_cam = visualize_cam(model = model, layer_idx = -1, filter_indices = 0, seed_input = image, penultimate_layer_idx = penultimate_layer_idx)
            r_ch = grad_cam[:,:,0]
            g_ch = grad_cam[:,:,1]
            b_ch = grad_cam[:,:,2]
            a_ch = ((255 - b_ch)*.5).astype(int)
            b_ch = b_ch
            grad_cam_filter = np.dstack((r_ch, g_ch, b_ch, a_ch))
            #generate plot
            plot_visualization_map(image=image, filter_map=grad_cam_filter, plot_title=plot_title, save_title = 'GradCam_' + save_title)
    
            #generate the saliency guided-backprop transparent filter
            guided_backprop = guided_backprops[j]
            guided_backprop = np.linalg.norm(guided_backprop, axis=2)
            guided_backprop = (guided_backprop*255/guided_backprop.max()).astype(int)
            r_ch = guided_backprop.copy()
            g_ch = guided_backprop.copy()*0
            b_ch = guided_backprop.copy()*0
            a_ch = guided_backprop*15
            guided_backprop_filter = np.dstack((r_ch, g_ch, b_ch, a_ch))
            #generate plot
            plot_visualization_map(image=image, filter_map=guided_backprop_filter, plot_title=plot_title, save_title = 'GuidedBackprop_' + save_title)
    
            #Generate summary plot
            plot_title = df_outer_fold['plot_title'].values[i*batch_size + j]
            save_title = df_outer_fold['save_title'].values[i*batch_size + j]
            plot_visualization_maps(image=image, saliency=saliency_filter, grad_cam=grad_cam_filter, guided_backprop=guided_backprop_filter, plot_title=plot_title, save_title=save_title)
