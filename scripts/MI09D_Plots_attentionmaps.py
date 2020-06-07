import sys
from MI_Classes import PlotsAttentionMaps

# Options
# Use a small subset of the data VS. run the actual full data pipeline to get accurate results
# /!\ if True, path to save weights will be automatically modified to avoid rewriting them
debug_mode = True

# Default parameters
if len(sys.argv) != 5:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('Pancreas')  # organ_view
    sys.argv.append('main')  # organ_view
    sys.argv.append('raw')  # transformation
    sys.argv.append('test')  # fold

# Generate results
Plots_AttentionMaps = PlotsAttentionMaps(target=sys.argv[1], organ=sys.argv[2], view=sys.argv[3],
                                         transformation=sys.argv[4], fold=sys.argv[5], debug_mode=debug_mode)
Plots_AttentionMaps.preprocessing()
Plots_AttentionMaps.generate_plots()

# Exit
print('Done.')
sys.exit(0)

#TODO: fix ids, instead of integer as index <- done.
#add side predictiors to df <- done.
#fix pb with repeated samples. maybe use name?
#convert float to int.

from MI_Classes import MyImageDataGenerator

self.generator = MyImageDataGenerator(target=self.target, organ=self.organ, data_features=self.df_outer_fold,
                                              n_samples_per_subepoch=self.n_samples_per_subepoch,
                                              batch_size=self.batch_size, training_mode=False, seed=self.seed,
                                              side_predictors=self.side_predictors, dir_images=self.dir_images,
                                              images_width=self.image_width, images_height=self.image_height,
                                              data_augmentation=False)

# Select with samples to plot
print('Selecting representative samples...')
Sexes = ['Male', 'Female']
dict_sexes_to_values = {'Male': 0, 'Female': 1}
df_to_plot = None
for sex in Sexes:
    print('Sex: ' + sex)
    Residuals_sex = self.Residuals[self.Residuals['Sex'] == dict_sexes_to_values[sex]]
    Residuals_sex['Sex'] = sex
    for age_category in ['young', 'middle', 'old']:
        print('Age category: ' + age_category)
        if age_category == 'young':
            Residuals_age = Residuals_sex[Residuals_sex['Age'] <= Residuals_sex['Age'].min() + 1]
        elif age_category == 'middle':
            Residuals_age = Residuals_sex[Residuals_sex['Age'] == int(Residuals_sex['Age'].median())]
        else:
            Residuals_age = Residuals_sex[Residuals_sex['Age'] >= Residuals_sex['Age'].max() - 1]
        Residuals_age['age_category'] = age_category
        for aging_rate in ['accelerated', 'normal', 'decelerated']:
            print('Aging rate: ' + aging_rate)
            Residuals_ar = Residuals_age
            if aging_rate == 'accelerated':
                Residuals_ar.sort_values(by='res', ascending=False, inplace=True)
            elif aging_rate == 'decelerated':
                Residuals_ar.sort_values(by='res', ascending=True, inplace=True)
            else:
                Residuals_ar.sort_values(by='res_abs', ascending=True, inplace=True)
            Residuals_ar['aging_rate'] = aging_rate
            Residuals_ar = Residuals_ar.iloc[:self.N_samples_attentionmaps, ]
            Residuals_ar['sample'] = range(len(Residuals_ar.index))
            if df_to_plot is None:
                df_to_plot = Residuals_ar
            else:
                df_to_plot = df_to_plot.append(Residuals_ar)
df_to_plot['plot_title'] = 'Age = ' + df_to_plot['Age'].astype(str) + ', Predicted Age = ' + (
        df_to_plot['Age'] - df_to_plot['res']).round().astype(str) + ', Sex = ' + df_to_plot[
                               'Sex'] + ', sample ' + \
                           df_to_plot['sample'].astype(str)
df_to_plot['save_title'] = self.target + '_' + self.organ + '_' + self.view + '_' + self.transformation + '_' \
                           + df_to_plot['Sex'] + '_' + df_to_plot['age_category'] + '_' \
                           + df_to_plot['aging_rate'] + '_' + df_to_plot['sample'].astype(str)
path_save = self.path_store + 'AttentionMaps-samples_' + self.target + '_' + self.organ + '_' + self.view + \
            '_' + self.transformation + '.csv'
df_to_plot.to_csv(path_save, index=False)
self.df_to_plot = df_to_plot


from vis.visualization import visualize_cam
self = Plots_AttentionMaps
x = self.df_to_plot[self.side_predictors].iloc[0,:]
from tf_keras_vis.gradcam import Gradcam
def loss(output):
    return output
Xs = self.generator.__getitem__(0)[0]
gradcam = Gradcam(self.model, model_modifier, clone=False)
cam = gradcam(loss, Xs)


grad_cam = visualize_cam(model=self.model, layer_idx=-1, filter_indices=0, seed_input=[self.image,x], penultimate_layer_idx=self.penultimate_layer_idx)


import tensorflow as tf
from tf_keras_vis.utils import print_gpus
print_gpus()
from tensorflow.keras.applications.vgg16 import VGG16 as Model
from tensorflow.keras.applications.vgg16 import preprocess_input
# Load model
model = Model(weights='imagenet', include_top=True)
model.summary()
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
# Load images
img1 = load_img('../images/6025202_2.jpg', target_size=(224, 224))
img2 = load_img('../images/6025202_2.jpg', target_size=(224, 224))
images = np.asarray([np.array(img1), np.array(img2)])
X = preprocess_input(images)
# Prepare datase
X = preprocess_input(images)
def loss(output):
    return (output[0][1], output[1][294])

# Define modifier to replace a softmax function of the last layer to a linear function.
def model_modifier(m):
    m.layers[-1].activation = tf.keras.activations.linear
    return m

from tensorflow.keras import backend as K
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
# Create Saliency object
saliency = Saliency(model, model_modifier, clone=False)
# Generate saliency map
saliency_map = saliency(loss, X)
saliency_map = normalize(saliency_map)