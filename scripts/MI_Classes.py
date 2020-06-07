# LIBRARIES
# set up backend for ssh -x11 figures
import matplotlib
matplotlib.use('Agg')

# read and write
import os
import sys
import glob
import re
import fnmatch
from datetime import datetime

# maths
import numpy as np
import pandas as pd
import math
import random

# miscellaneous
import warnings
import gc

# sklearn
from sklearn.utils import resample
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, log_loss, roc_auc_score, accuracy_score, f1_score, \
    precision_score, recall_score, confusion_matrix, average_precision_score
from sklearn import linear_model

# CPUs
from multiprocessing import Pool
# GPUs
from GPUtil import GPUtil
# tensorflow
import tensorflow as tf
# keras
from keras_preprocessing.image import ImageDataGenerator, Iterator
from keras_preprocessing.image.utils import load_img, img_to_array
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D, concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.metrics import RootMeanSquaredError, AUC, BinaryAccuracy, Precision, Recall, TruePositives, \
    FalsePositives, FalseNegatives, TrueNegatives
from tensorflow_addons.metrics import RSquare, F1Score

# Plots
import matplotlib.pyplot as plt
import seaborn as sns

# Model's attention
import innvestigate
from vis.utils import utils
from vis.visualization import visualize_cam

# Necessary to define MyCSVLogger
import collections
import csv
import io
import six
from tensorflow.python.lib.io import file_io
from tensorflow.python.util.compat import collections_abc
from tensorflow.keras.backend import eval

# Set display parameters
pd.set_option('display.max_rows', 200)


# CLASSES
class Hyperparameters:
    
    def __init__(self):
        # seeds for reproducibility
        self.seed = 0
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        # other parameters
        self.path_store = '../data/'
        self.folds = ['train', 'val', 'test']
        self.n_CV_outer_folds = 10
        #TODO DEBUG
        self.n_CV_outer_folds = 1
        self.outer_folds = [str(x) for x in list(range(self.n_CV_outer_folds))]
        self.ensemble_types = ['*', '?', ',']
        self.modes = ['', '_sd', '_str']
        self.id_vars = ['id', 'eid', 'instance']
        self.ethnicities_vars = ['Ethnicity.White', 'Ethnicity.British', 'Ethnicity.Irish', 'Ethnicity.White_Other',
                                 'Ethnicity.Mixed', 'Ethnicity.White_and_Black_Caribbean',
                                 'Ethnicity.White_and_Black_African', 'Ethnicity.White_and_Asian',
                                 'Ethnicity.Mixed_Other', 'Ethnicity.Asian', 'Ethnicity.Indian',
                                 'Ethnicity.Pakistani', 'Ethnicity.Bangladeshi', 'Ethnicity.Asian_Other',
                                 'Ethnicity.Black', 'Ethnicity.Caribbean', 'Ethnicity.African',
                                 'Ethnicity.Black_Other', 'Ethnicity.Chinese', 'Ethnicity.Other_ethnicity',
                                 'Ethnicity.Do_not_know', 'Ethnicity.Prefer_not_to_answer', 'Ethnicity.NA']
        self.demographic_vars = ['Age', 'Sex'] + self.ethnicities_vars
        self.names_model_parameters = ['target', 'organ', 'view', 'transformation', 'architecture',
                                       'optimizer', 'learning_rate', 'weight_decay', 'dropout_rate']
        self.targets_regression = ['Age']
        self.targets_binary = ['Sex']
        self.dict_prediction_types = {'Age': 'regression', 'Sex': 'binary'}
        self.dict_side_predictors = {'Age': ['Sex'] + self.ethnicities_vars, 'Sex': ['Age'] + self.ethnicities_vars}
        self.organs = ['Brain', 'Carotids', 'Eyes', 'Heart', 'Liver', 'Pancreas', 'FullBody', 'Spine', 'Hips', 'Knees',
                       'PhysicalActivity']
        self.left_right_organs = ['Carotids', 'Eyes', 'Hips', 'Knees']
        self.dict_organs_to_views = {'Brain': ['sagittal', 'coronal', 'transverse'],
                                           'Carotids': ['longaxis', 'shortaxis', 'CIMT120', 'CIMT150', 'mixed'],
                                           'Eyes': ['fundus', 'OCT'],
                                           'Heart': ['2chambers', '3chambers', '4chambers'],
                                           'Liver': ['main'],
                                           'Pancreas': ['main'],
                                           'FullBody': ['figure', 'skeleton', 'flesh', 'mixed'],
                                           'Spine': ['sagittal', 'coronal'],
                                           'Hips': ['main'],
                                           'Knees': ['main'],
                                           'PhysicalActivity': ['main']}
        # the number of epochs is too small for data augmentation to be helpful for now
        self.organs_not_to_augment = ['Brain', 'Carotids', 'Eyes', 'Heart', 'Liver', 'Pancreas', 'FullBody', 'Spine',
                                      'Hips', 'Knees', 'PhysicalActivity']
        self.organs_to_augment = []
        
        # Others
        if '/Users/Alan/' in os.getcwd():
            os.chdir('/Users/Alan/Desktop/Aging/Medical_Images/scripts/')
        else:
            os.chdir('/n/groups/patel/Alan/Aging/Medical_Images/scripts/')
        gc.enable()  # garbage collector
        warnings.filterwarnings('ignore')
    
    def _version_to_parameters(self, model_name):
        parameters = {}
        parameters_list = model_name.split('_')
        for i, parameter in enumerate(self.names_model_parameters):
            parameters[parameter] = parameters_list[i]
        if len(parameters_list) > 9:
            parameters['outer_fold'] = parameters_list[9]
        return parameters
    
    @staticmethod
    def _parameters_to_version(parameters):
        return '_'.join(parameters.values())
    
    @staticmethod
    def convert_string_to_boolean(string):
        if string == 'True':
            boolean = True
        elif string == 'False':
            boolean = False
        else:
            print('ERROR: string must be either \'True\' or \'False\'')
            sys.exit(1)
        return boolean


class Metrics(Hyperparameters):
    
    def __init__(self):
        # Parameters
        Hyperparameters.__init__(self)
        self.metrics_displayed_in_int = ['True-Positives', 'True-Negatives', 'False-Positives', 'False-Negatives']
        self.metrics_needing_classpred = ['F1-Score', 'Binary-Accuracy', 'Precision', 'Recall']
        self.dict_metrics_names_K = {'regression': ['RMSE'],  # For now, RSquare is buggy. Try again in a few months.
                                     'binary': ['ROC-AUC', 'PR-AUC', 'F1-Score', 'Binary-Accuracy', 'Precision',
                                                'Recall', 'True-Positives', 'False-Positives', 'False-Negatives',
                                                'True-Negatives'],
                                     'multiclass': ['Categorical-Accuracy']}
        self.dict_metrics_names = {'regression': ['RMSE', 'R-Squared'],
                                   'binary': ['ROC-AUC', 'F1-Score', 'PR-AUC', 'Binary-Accuracy', 'Sensitivity',
                                              'Specificity', 'Precision', 'Recall', 'True-Positives', 'False-Positives',
                                              'False-Negatives', 'True-Negatives'],
                                   'multiclass': ['Categorical-Accuracy']}
        self.dict_losses_names = {'regression': 'MSE', 'binary': 'Binary-Crossentropy',
                                  'multiclass': 'categorical_crossentropy'}
        self.dict_main_metrics_names_K = {'Age': 'RMSE', 'Sex': 'PR-AUC',
                                          'imbalanced_binary_placeholder': 'PR-AUC'}
        self.dict_main_metrics_names = {'Age': 'R-Squared', 'Sex': 'ROC-AUC',
                                        'imbalanced_binary_placeholder': 'PR-AUC'}
        self.main_metrics_modes = {'loss': 'min', 'R-Squared': 'max', 'RMSE': 'min', 'ROC-AUC': 'max', 'PR-AUC': 'max',
                                   'F1-Score': 'max'}
        
        def rmse(y_true, y_pred):
            return math.sqrt(mean_squared_error(y_true, y_pred))
        
        def sensitivity_score(y, pred):
            _, _, fn, tp = confusion_matrix(y, pred.round()).ravel()
            return tp / (tp + fn)
        
        def specificity_score(y, pred):
            tn, fp, _, _ = confusion_matrix(y, pred.round()).ravel()
            return tn / (tn + fp)
        
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
        
        self.dict_metrics_sklearn = {'mean_squared_error': mean_squared_error,
                                     'RMSE': rmse,
                                     'R-Squared': r2_score,
                                     'Binary-Crossentropy': log_loss,
                                     'ROC-AUC': roc_auc_score,
                                     'F1-Score': f1_score,
                                     'PR-AUC': average_precision_score,
                                     'Binary-Accuracy': accuracy_score,
                                     'Sensitivity': sensitivity_score,
                                     'Specificity': specificity_score,
                                     'Precision': precision_score,
                                     'Recall': recall_score,
                                     'True-Positives': true_positives_score,
                                     'False-Positives': false_positives_score,
                                     'False-Negatives': false_negatives_score,
                                     'True-Negatives': true_negatives_score}


class PreprocessingMain(Hyperparameters):
    
    def __init__(self):
        Hyperparameters.__init__(self)
        self.data_raw = None
        self.data_features = None
        self.data_features_eids = None
    
    def _compute_age(self):
        # Recompute age with greater precision by leveraging the month of birth
        self.data_raw['Year_of_birth'] = self.data_raw['Year_of_birth'].astype(int)
        self.data_raw['Month_of_birth'] = self.data_raw['Month_of_birth'].astype(int)
        self.data_raw['Date_of_birth'] = self.data_raw.apply(
            lambda row: datetime(row.Year_of_birth, row.Month_of_birth, 15), axis=1)
        for i in [str(i) for i in range(4)]:
            self.data_raw['Date_attended_center_' + i] = \
                self.data_raw['Date_attended_center_' + i].apply(
                    lambda x: pd.NaT if pd.isna(x) else datetime.strptime(x, '%Y-%m-%d'))
            self.data_raw['Age_' + i] = self.data_raw['Date_attended_center_' + i] - self.data_raw['Date_of_birth']
            self.data_raw['Age_' + i] = self.data_raw['Age_' + i].dt.days / 365.25
            self.data_raw = self.data_raw.drop(['Date_attended_center_' + i], axis=1)
        self.data_raw = self.data_raw.drop(['Year_of_birth', 'Month_of_birth', 'Date_of_birth'], axis=1)
    
    def _encode_ethnicity(self):
        # Fill NAs for ethnicity on instance 0 if available in other instances
        eids_missing_ethnicity = self.data_raw['eid'][self.data_raw['Ethnicity'].isna()]
        for eid in eids_missing_ethnicity:
            sample = self.data_raw.loc[eid, :]
            if not math.isnan(sample['Ethnicity_1']):
                self.data_raw.loc[eid, 'Ethnicity'] = self.data_raw.loc[eid, 'Ethnicity_1']
            elif not math.isnan(sample['Ethnicity_2']):
                self.data_raw.loc[eid, 'Ethnicity'] = self.data_raw.loc[eid, 'Ethnicity_2']
        self.data_raw.drop(['Ethnicity_1', 'Ethnicity_2'], axis=1, inplace=True)
        
        # One hot encode ethnicity
        dict_ethnicity_codes = {'1': 'Ethnicity.White', '1001': 'Ethnicity.British', '1002': 'Ethnicity.Irish',
                                '1003': 'Ethnicity.White_Other',
                                '2': 'Ethnicity.Mixed', '2001': 'Ethnicity.White_and_Black_Caribbean',
                                '2002': 'Ethnicity.White_and_Black_African',
                                '2003': 'Ethnicity.White_and_Asian', '2004': 'Ethnicity.Mixed_Other',
                                '3': 'Ethnicity.Asian', '3001': 'Ethnicity.Indian', '3002': 'Ethnicity.Pakistani',
                                '3003': 'Ethnicity.Bangladeshi', '3004': 'Ethnicity.Asian_Other',
                                '4': 'Ethnicity.Black', '4001': 'Ethnicity.Caribbean', '4002': 'Ethnicity.African',
                                '4003': 'Ethnicity.Black_Other',
                                '5': 'Ethnicity.Chinese',
                                '6': 'Ethnicity.Other_ethnicity',
                                '-1': 'Ethnicity.Do_not_know',
                                '-3': 'Ethnicity.Prefer_not_to_answer',
                                '-5': 'Ethnicity.NA'}
        self.data_raw['Ethnicity'] = self.data_raw['Ethnicity'].fillna(-5).astype(int).astype(str)
        ethnicities = pd.get_dummies(self.data_raw['Ethnicity'])
        self.data_raw = self.data_raw.drop(['Ethnicity'], axis=1)
        ethnicities.rename(columns=dict_ethnicity_codes, inplace=True)
        ethnicities['Ethnicity.White'] = ethnicities['Ethnicity.White'] + ethnicities['Ethnicity.British'] + \
                                         ethnicities['Ethnicity.Irish'] + ethnicities['Ethnicity.White_Other']
        ethnicities['Ethnicity.Mixed'] = ethnicities['Ethnicity.Mixed'] + \
                                         ethnicities['Ethnicity.White_and_Black_Caribbean'] + \
                                         ethnicities['Ethnicity.White_and_Black_African'] + \
                                         ethnicities['Ethnicity.White_and_Asian'] + \
                                         ethnicities['Ethnicity.Mixed_Other']
        ethnicities['Ethnicity.Asian'] = ethnicities['Ethnicity.Asian'] + ethnicities['Ethnicity.Indian'] + \
                                         ethnicities['Ethnicity.Pakistani'] + ethnicities['Ethnicity.Bangladeshi'] + \
                                         ethnicities['Ethnicity.Asian_Other']
        ethnicities['Ethnicity.Black'] = ethnicities['Ethnicity.Black'] + ethnicities['Ethnicity.Caribbean'] + \
                                         ethnicities['Ethnicity.African'] + ethnicities['Ethnicity.Black_Other']
        ethnicities['Ethnicity.Other'] = ethnicities['Ethnicity.Other_ethnicity'] + \
                                         ethnicities['Ethnicity.Do_not_know'] + \
                                         ethnicities['Ethnicity.Prefer_not_to_answer'] + \
                                         ethnicities['Ethnicity.NA']
        self.data_raw = self.data_raw.join(ethnicities)
    
    def generate_data(self):
        # Preprocessing
        dict_UKB_fields_to_names = {'34-0.0': 'Year_of_birth', '52-0.0': 'Month_of_birth',
                                    '53-0.0': 'Date_attended_center_0', '53-1.0': 'Date_attended_center_1',
                                    '53-2.0': 'Date_attended_center_2', '53-3.0': 'Date_attended_center_3',
                                    '31-0.0': 'Sex', '21000-0.0': 'Ethnicity', '21000-1.0': 'Ethnicity_1',
                                    '21000-2.0': 'Ethnicity_2', '22414-2.0': 'Abdominal_images_quality'}
        self.data_raw = pd.read_csv('/n/groups/patel/uk_biobank/project_52887_41230/ukb41230.csv',
                                    usecols=['eid', '31-0.0', '21000-0.0', '21000-1.0', '21000-2.0', '34-0.0', '52-0.0',
                                             '53-0.0', '53-1.0', '53-2.0', '53-3.0', '22414-2.0'])
        
        # Formatting
        self.data_raw.rename(columns=dict_UKB_fields_to_names, inplace=True)
        self.data_raw['eid'] = self.data_raw['eid'].astype(str)
        self.data_raw.set_index('eid', drop=False, inplace=True)
        self.data_raw = self.data_raw.dropna(subset=['Sex'])
        self._compute_age()
        self.data_raw = self.data_raw.dropna(how='all', subset=['Age_0', 'Age_1', 'Age_2', 'Age_3'])
        self._encode_ethnicity()
        
        # Concatenate the data from the different instances
        self.data_features = None
        for i in [str(i) for i in range(4)]:
            print('Preparing the samples for instance ' + i)
            df_i = self.data_raw[['eid', 'Sex', 'Age_' + i] + self.ethnicities_vars + ['Abdominal_images_quality']
                                 ].dropna(subset=['Age_' + i])
            print(str(len(df_i.index)) + ' samples found in instance ' + i)
            df_i.rename(columns={'Age_' + i: 'Age'}, inplace=True)
            df_i['instance'] = i
            df_i['id'] = df_i['eid'] + '_' + df_i['instance']
            df_i = df_i[self.id_vars + self.demographic_vars + ['Abdominal_images_quality']]
            if i != '2':
                df_i['Abdominal_images_quality'] = np.nan  # not defined for instance 3, not relevant for instances 0, 1
            if self.data_features is None:
                self.data_features = df_i
            else:
                self.data_features = self.data_features.append(df_i)
            print('The size of the full concatenated dataframe is now ' + str(len(self.data_features.index)))
        
        # Shuffle the rows before saving the dataframe
        self.data_features = self.data_features.sample(frac=1)
        
        # Generate dataframe for eids pipeline as opposed to instances pipeline
        self.data_features_eids = self.data_features[self.data_features.instance == '0']
        self.data_features_eids['instance'] = '*'
        self.data_features_eids['id'] = [ID.replace('_0', '_*') for ID in self.data_features_eids['id'].values]

    def save_data(self):
        self.data_features.to_csv(self.path_store + 'data-features_instances.csv', index=False)
        self.data_features_eids.to_csv(self.path_store + 'data-features_eids.csv', index=False)


class PreprocessingFolds(Metrics):
    """
    Gather all the hyperparameters of the algorithm
    """
    
    def __init__(self, target, organ):
        Metrics.__init__(self)
        self.target = target
        self.organ = organ
        self.side_predictors = self.dict_side_predictors[self.target]
        self.variables_to_normalize = self.side_predictors
        if self.target in self.targets_regression:
            self.variables_to_normalize.append(self.target)
        self.dict_image_quality_col = {'Liver': 'Abdominal_images_quality'}
        self.dict_image_quality_col.update(
            dict.fromkeys(['Brain', 'Carotids', 'Eyes', 'Heart', 'Liver', 'Pancreas', 'FullBody', 'Spine', 'Hips',
                           'Knees', 'PhysicalActivity', 'ECG', 'ArterialStiffness'], None))
        self.image_quality_col = self.dict_image_quality_col[self.organ]
        self.views = self.dict_organs_to_views[self.organ]
        self.list_ids = None
        self.list_ids_per_view = {}
        self.data = None
        self.EIDS = None
        self.EIDS_per_view = {'train': {}, 'val': {}, 'test': {}}
        self.data_fold = None
    
    def _get_list_ids(self):
        # get the list of the ids available for the organ
        if self.organ in self.organs:
            list_ids = []
            # if different views are available, take the union of the ids
            for view in self.views:
                path = '../images/' + self.organ + '/' + view + '/' + 'raw' + '/'
                list_ids_view = []
                # for paired organs, take the unions of the ids available on the right and the left sides
                if self.organ in self.left_right_organs:
                    for side in ['right', 'left']:
                        list_ids_view += os.listdir(path + side + '/')
                    list_ids_view = np.unique(list_ids_view).tolist()
                else:
                    list_ids_view += os.listdir(path)
                self.list_ids_per_view[view] = [im.replace('.jpg', '') for im in list_ids_view]
                list_ids += self.list_ids_per_view[view]
            self.list_ids = np.unique(list_ids).tolist()
            self.list_ids.sort()
        else:
            list_ids_raw = pd.read_csv(self.path_store + 'IDs_' + self.organ + '.csv')
            self.list_ids = list_ids_raw.values.squeeze().astype(str)
    
    def _filter_and_format_data(self):
        """
        Clean the data before it can be split between the rows
        """
        cols_data = self.id_vars + self.demographic_vars
        if self.image_quality_col is not None:
            cols_data.append(self.dict_image_quality_col[self.organ])
        data = pd.read_csv(self.path_store + 'data-features_instances.csv', usecols=cols_data)
        data.rename(columns={self.dict_image_quality_col[self.organ]: 'Data_quality'}, inplace=True)
        for col_name in self.id_vars:
            data[col_name] = data[col_name].astype(str)
        data.set_index('id', drop=False, inplace=True)
        if self.image_quality_col is not None:
            data = data[data['Data_quality'] != np.nan]
            data = data.drop('Data_quality', axis=1)
        # get rid of samples with NAs
        data.dropna(inplace=True)
        # list the samples' ids for which images are available
        data = data.loc[self.list_ids]
        self.data = data
    
    def _split_eids(self):
        # distribute the eids between the different outer and inner folds
        eids = self.data['eid'].unique()
        random.shuffle(eids)
        n_samples = len(eids)
        n_samples_by_fold = n_samples / self.n_CV_outer_folds
        FOLDS_EIDS = {}
        for outer_fold in self.outer_folds:
            FOLDS_EIDS[outer_fold] = np.ndarray.tolist(
                eids[int((int(outer_fold)) * n_samples_by_fold):int((int(outer_fold) + 1) * n_samples_by_fold)])
        TRAINING_EIDS = {}
        VALIDATION_EIDS = {}
        TEST_EIDS = {}
        for i in self.outer_folds:
            TRAINING_EIDS[i] = []
            VALIDATION_EIDS[i] = []
            TEST_EIDS[i] = []
            for j in self.outer_folds:
                if j == i:
                    VALIDATION_EIDS[i].extend(FOLDS_EIDS[j])
                elif ((int(i) + 1) % self.n_CV_outer_folds) == int(j):
                    TEST_EIDS[i].extend(FOLDS_EIDS[j])
                else:
                    TRAINING_EIDS[i].extend(FOLDS_EIDS[j])
        self.EIDS = {'train': TRAINING_EIDS, 'val': VALIDATION_EIDS, 'test': TEST_EIDS}
    
    def _split_data(self):
        # generate inner fold split for each outer fold
        for view in self.views:
            print('Splitting data for view ' + view)
            normalizing_values = {}
            for outer_fold in self.outer_folds:
                print('Splitting data for outer fold ' + outer_fold)
                # compute values for scaling of variables
                data_train = self.data.iloc[self.data['eid'].isin(self.EIDS['train'][outer_fold]).values &
                                            self.data['id'].isin(self.list_ids_per_view[view]).values, :]
                for var in self.variables_to_normalize:
                    var_mean = data_train[var].mean()
                    if len(data_train[var].unique()) < 2:
                        print('Variable ' + var + ' has a single value in fold ' + outer_fold +
                              '. Using 1 as std for normalization.')
                        var_std = 1
                    else:
                        var_std = data_train[var].std()
                    normalizing_values[var] = {'mean': var_mean, 'std': var_std}
                # generate folds
                for fold in self.folds:
                    data_fold = \
                        self.data.iloc[self.data['eid'].isin(self.EIDS[fold][outer_fold]).values &
                                       self.data['id'].isin(self.list_ids_per_view[view]).values, :]
                    data_fold['outer_fold'] = outer_fold
                    data_fold = data_fold[self.id_vars + ['outer_fold'] + self.demographic_vars]
                    # normalize the variables
                    for var in self.variables_to_normalize:
                        data_fold[var + '_raw'] = data_fold[var]
                        data_fold[var] = (data_fold[var] - normalizing_values[var]['mean']) \
                                         / normalizing_values[var]['std']
                    # report issue if NAs were detected, which most likely comes from a sample whose id did not match
                    n_mismatching_samples = data_fold.isna().sum().max()
                    if n_mismatching_samples > 0:
                        print(data_fold[data_fold.isna().any(axis=1)])
                        print('/!\\ WARNING! ' + str(n_mismatching_samples) + ' ' + fold + ' images ids out of ' +
                              str(len(data_fold.index)) + ' did not match the dataframe!')
                    data_fold.to_csv(self.path_store + 'data-features_' + self.organ + '_' + view + '_' +
                                     self.target + '_' + fold + '_' + outer_fold + '.csv', index=False)
                    print('For outer_fold ' + outer_fold + ', the ' + fold + ' fold has a sample size of ' +
                          str(len(data_fold.index)))
                    self.data_fold = data_fold
    
    def generate_folds(self):
        self._get_list_ids()
        self._filter_and_format_data()
        self._split_eids()
        self._split_data()


class MyImageDataGenerator(Hyperparameters, Sequence, ImageDataGenerator):
    
    def __init__(self, target=None, organ=None, data_features=None, n_samples_per_subepoch=None, batch_size=None,
                 training_mode=None, seed=None, side_predictors=None, dir_images=None, images_width=None,
                 images_height=None, data_augmentation=False):
        # Parameters
        Hyperparameters.__init__(self)
        self.target = target
        if self.target in self.targets_regression:
            self.labels = data_features[self.target]
        else:
            self.labels = data_features[self.target + '_raw']
        self.organ = organ
        self.training_mode = training_mode
        self.data_features = data_features
        self.list_ids = data_features.index.values
        self.batch_size = batch_size
        # for paired organs, take twice fewer ids (two images for each id), and add organ_side as side predictor
        if self.organ in self.left_right_organs:
            self.data_features['organ_side'] = np.nan
            self.n_ids_batch = self.batch_size // 2
        else:
            self.n_ids_batch = self.batch_size
        if self.training_mode & (n_samples_per_subepoch is not None):  # during training, 1 epoch = number of samples
            self.steps = math.ceil(n_samples_per_subepoch / self.batch_size)
        else:  # during prediction and other tasks, an epoch is defined as all the samples being seen once and only once
            self.steps = math.ceil(len(self.list_ids) / self.n_ids_batch)
        # initiate the indices and shuffle the ids
        self.shuffle = training_mode  # Only shuffle if the model is being trained. Otherwise no need.
        self.indices = np.arange(len(self.list_ids))
        self.idx_end = 0  # Keep track of last indice to permute indices accordingly at the end of epoch.
        if self.shuffle:
            np.random.shuffle(self.indices)
        # Input for side NN and CNN
        self.side_predictors = side_predictors
        self.dir_images = dir_images
        self.images_width = images_width
        self.images_height = images_height
        # Data augmentation
        self.data_augmentation = data_augmentation
        self.seed = seed
        # Parameters for data augmentation: (rotation range, width shift range, height shift range, zoom range)
        self.dict_augmentation_parameters = {}
        self.dict_augmentation_parameters.update(dict.fromkeys(self.organs_not_to_augment, (0, 0, 0, [0, 0])))
        self.dict_augmentation_parameters.update(dict.fromkeys(self.organs_to_augment, (10, 0.1, 0.1, [0.9, 1.1])))
        
        ImageDataGenerator.__init__(self, rescale=1. / 255.,
                                    rotation_range=self.dict_augmentation_parameters[self.organ][0],
                                    width_shift_range=self.dict_augmentation_parameters[self.organ][1],
                                    height_shift_range=self.dict_augmentation_parameters[self.organ][2],
                                    zoom_range=self.dict_augmentation_parameters[self.organ][3])
    
    def __len__(self):
        return self.steps
    
    def on_epoch_end(self):
        _ = gc.collect()
        self.indices = np.concatenate([self.indices[self.idx_end:], self.indices[:self.idx_end]])
    
    def _generate_image(self, path_image):
        img = load_img(path_image, target_size=(self.images_width, self.images_height), color_mode='rgb')
        Xi = img_to_array(img)
        if hasattr(img, 'close'):
            img.close()
        if self.data_augmentation:
            params = self.get_random_transform(Xi.shape, seed=self.seed)
            Xi = self.apply_transform(Xi, params)
        Xi = self.standardize(Xi)
        return Xi
    
    def _data_generation(self, list_ids_batch):
        # initialize empty matrices
        n_samples_batch = min(len(list_ids_batch), self.batch_size)
        X = np.empty((n_samples_batch, self.images_width, self.images_height, 3)) * np.nan
        x = np.empty((n_samples_batch, len(self.side_predictors))) * np.nan
        y = np.empty((n_samples_batch, 1)) * np.nan
        # fill the matrices sample by sample
        for i, ID in enumerate(list_ids_batch):
            y[i] = self.labels[ID]
            x[i] = self.data_features.loc[ID, self.side_predictors]
            if self.organ in self.left_right_organs:
                if i % 2 == 0:
                    path = self.dir_images + 'right/'
                    x[i][-1] = 0
                else:
                    path = self.dir_images + 'left/'
                    x[i][-1] = 1
                if not os.path.exists(path + ID + '.jpg'):
                    path = path.replace('/right/', '/left/') if i % 2 == 0 else path.replace('/left/', '/right/')
                    x[i][-1] = 1 - x[i][-1]
            else:
                path = self.dir_images
            X[i, ] = self._generate_image(path_image=path + ID + '.jpg')
        return [X, x], y
    
    def __getitem__(self, index):
        # Select the indices
        idx_start = (index * self.n_ids_batch) % len(self.list_ids)
        idx_end = (((index + 1) * self.n_ids_batch) - 1) % len(self.list_ids) + 1
        if idx_start > idx_end:
            # If this happens outside of training, that is a mistake
            if not self.training_mode:
                print('\nERROR: Outside of training, every sample should only be predicted once!')
                sys.exit(1)
            # Select part of the indices from the end of the epoch
            indices = self.indices[idx_start:]
            # Generate a new set of indices
            # print('\nThe end of the data was reached within this batch, looping.')
            if self.shuffle:
                np.random.shuffle(self.indices)
            # Complete the batch with samples from the new indices
            indices = np.concatenate([indices, self.indices[:idx_end]])
        else:
            indices = self.indices[idx_start: idx_end]
            if idx_end == len(self.list_ids) & self.shuffle:
                # print('\nThe end of the data was reached. Shuffling for the next epoch.')
                np.random.shuffle(self.indices)
        # Keep track of last indice for end of subepoch
        self.idx_end = idx_end
        # Select the corresponding ids
        list_ids_batch = [self.list_ids[i] for i in indices]
        # For paired organs, two images (left, right eyes) are selected for each id.
        if self.organ in self.left_right_organs:
            list_ids_batch = [ID for ID in list_ids_batch for _ in ('right', 'left')]
        return self._data_generation(list_ids_batch)


class MyCSVLogger(Callback):
    
    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.csv_file = None
        if six.PY2:
            self.file_flags = 'b'
            self._open_args = {}
        else:
            self.file_flags = ''
            self._open_args = {'newline': '\n'}
        Callback.__init__(self)
    
    def on_train_begin(self, logs=None):
        if self.append:
            if file_io.file_exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            mode = 'a'
        else:
            mode = 'w'
        self.csv_file = io.open(self.filename, mode + self.file_flags, **self._open_args)
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, collections_abc.Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k
        
        if self.keys is None:
            self.keys = sorted(logs.keys())
        
        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.keys])
        
        if not self.writer:
            
            class CustomDialect(csv.excel):
                delimiter = self.sep
            
            fieldnames = ['epoch', 'learning_rate'] + self.keys
            if six.PY2:
                fieldnames = [unicode(x) for x in fieldnames]
            
            self.writer = csv.DictWriter(
                self.csv_file,
                fieldnames=fieldnames,
                dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()
        
        row_dict = collections.OrderedDict({'epoch': epoch, 'learning_rate': eval(self.model.optimizer.lr)})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()
    
    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None


class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', baseline=-np.Inf, verbose=0, save_best_only=False,
                 save_weights_only=False, mode='auto', save_freq='epoch'):
        # Parameters
        ModelCheckpoint.__init__(self, filepath, monitor=monitor, verbose=verbose, save_best_only=save_best_only,
                                 save_weights_only=save_weights_only, mode=mode, save_freq=save_freq)
        if mode == 'min':
            self.monitor_op = np.less
            self.best = baseline
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = baseline
        else:
            print('Error. mode for metric must be either min or max')
            sys.exit(1)


class DeepLearning(Metrics):
    """
    Train models
    """
    
    def __init__(self, target=None, organ=None, view=None, transformation=None, architecture=None, optimizer=None,
                 learning_rate=None, weight_decay=None, dropout_rate=None, debug_mode=False):
        # Initialization
        Metrics.__init__(self)
        tf.random.set_seed(self.seed)
        
        # Model's version
        self.target = target
        self.organ = organ
        self.view = view
        self.transformation = transformation
        self.architecture = architecture
        self.optimizer = optimizer
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.dropout_rate = float(dropout_rate)
        self.outer_fold = None
        self.version = self.target + '_' + self.organ + '_' + self.view + '_' + self.transformation + '_' + \
                       self.architecture + '_' + self.optimizer + '_' + np.format_float_positional(self.learning_rate) \
                       + '_' + str(self.weight_decay) + '_' + str(self.dropout_rate)
        
        # NNet's architecture and weights
        self.side_predictors = self.dict_side_predictors[self.target]
        if self.organ in self.left_right_organs:
            self.side_predictors.append('organ_side')
        self.dict_final_activations = {'regression': 'linear', 'binary': 'sigmoid', 'multiclass': 'softmax',
                                       'saliency': 'linear'}
        self.path_load_weights = None
        self.keras_weights = None
        
        # Generators
        self.debug_mode = debug_mode
        self.debug_fraction = 0.005
        self.DATA_FEATURES = {}
        self.mode = None
        self.n_cpus = len(os.sched_getaffinity(0))
        self.dir_images = '../images/' + self.organ + '/' + self.view + '/' + self.transformation + '/'
        
        # define dictionary to fit the architecture's input size to the images sizes (take min (height, width))
        self.dict_organ_view_to_image_size = {
            'Brain_coronal': (316, 316),  # initial size (88, 88) TODO see if 88 -> 316 made a difference
            'Brain_sagittal': (316, 316),  # initial size (88, 88)
            'Brain_transverse': (316, 316),  # initial size (88, 88)
            'Carotids_shortaxis': (337, 291),  # initial size (505, 436)
            'Carotids_longaxis': (337, 291),  # initial size (505, 436)
            'Carotids_CIMT120': (337, 291),  # initial size (505, 436)
            'Carotids_CIMT150': (337, 291),  # initial size (505, 436)
            'Carotids_mixed': (337, 291),  # initial size (505, 436)
            'Eyes_fundus': (316, 316),  # initial size (1388, 1388)
            'Eyes_OCT': (312, 320),  # initial size (500, 512)
            'Heart_2chambers': (316, 316),  # initial size (200, 200) TODO consider 200 -> 316
            'Heart_3chambers': (316, 316),  # initial size (200, 200)
            'Heart_4chambers': (316, 316),  # initial size (200, 200)
            'Liver_main': (288, 364),  # initial size (364, 288)
            'Pancreas_main': (288, 350),  # initial size (350, 288)
            'FullBody_figure': (541, 181),  # initial size (811, 272)
            'FullBody_skeleton': (541, 181),  # initial size (811, 272)
            'FullBody_flesh': (541, 181),  # initial size (811, 272)
            'FullBody_mixed': (541, 181),  # initial size (811, 272)
            'Spine_sagittal': (466, 211),  # initial size (1513, 684)
            'Spine_coronal': (315, 313),  # initial size (724, 720)
            'Hips_main': (329, 303),  # initial size (626, 680)
            'Knees_main': (347, 286)  # initial size (851, 700)
        }
        self.dict_architecture_to_image_size = {'MobileNet': (224, 224), 'MobileNetV2': (224, 224),
                                                'NASNetMobile': (224, 224), 'NASNetLarge': (331, 331)}
        if self.architecture in ['MobileNet', 'MobileNetV2', 'NASNetMobile', 'NASNetLarge']:
            self.image_width, self.image_height = self.dict_architecture_to_image_size[self.architecture]
        else:
            self.image_width, self.image_height = self.dict_organ_view_to_image_size[self.organ + '_' + self.view]
        
        # define dictionary of batch sizes to fit as many samples as the model's architecture allows
        self.dict_batch_sizes = {
            # Default, applies to all images with resized input ~100,000 pixels
            'Default': {'VGG16': 32, 'VGG19': 32, 'DenseNet121': 16, 'DenseNet169': 16, 'DenseNet201': 16,
                        'Xception': 32, 'InceptionV3': 64, 'InceptionResNetV2': 16, 'ResNet50': 32, 'ResNet101': 16,
                        'ResNet152': 16, 'ResNet50V2': 32, 'ResNet101V2': 16, 'ResNet152V2': 16, 'ResNeXt50': 4,
                        'ResNeXt101': 8, 'EfficientNetB7': 4,
                        'MobileNet': 128, 'MobileNetV2': 64, 'NASNetMobile': 64, 'NASNetLarge': 4}}
        # Define batch size
        if self.organ + '_' + self.view in self.dict_batch_sizes.keys():
            self.batch_size = self.dict_batch_sizes[self.organ + '_' + self.view][self.architecture]
        else:
            self.batch_size = self.dict_batch_sizes['Default'][self.architecture]
        # double the batch size for the teslaM40 cores that have bigger memory
        if len(GPUtil.getGPUs()) > 0:  # make sure GPUs are available (not truesometimes for debugging)
            if GPUtil.getGPUs()[0].memoryTotal > 20000:
                self.batch_size *= 2
        # Define number of ids per batch (twice fewer for paired organs, because left and right samples)
        self.n_ids_batch = self.batch_size
        if self.organ in self.left_right_organs:
            self.n_ids_batch //= 2
        
        # Define number of samples per subepoch
        if self.debug_mode:
            self.n_samples_per_subepoch = self.batch_size * 4
        else:
            self.n_samples_per_subepoch = 32768
        if self.organ in self.left_right_organs:
            self.n_samples_per_subepoch //= 2
        
        # dict to decide which field is used to generate the ids when several targets share the same ids
        self.dict_target_to_ids = dict.fromkeys(['Age', 'Sex'], 'Age')
        
        # Metrics
        self.prediction_type = self.dict_prediction_types[self.target]
        
        # Model
        self.model = None
        
        # Note: R-Squared and F1-Score are not available, because their batch based values are misleading.
        # For some reason, Sensitivity and Specificity are not available either. Might implement later.
        self.dict_losses_K = {'MSE': MeanSquaredError(name='MSE'),
                              'Binary-Crossentropy': BinaryCrossentropy(name='Binary-Crossentropy')}
        self.dict_metrics_K = {'R-Squared': RSquare(name='R-Squared', y_shape=(1,)),
                               'RMSE': RootMeanSquaredError(name='RMSE'),
                               'F1-Score': F1Score(name='F1-Score', num_classes=1, dtype=tf.float32),
                               'ROC-AUC': AUC(curve='ROC', name='ROC-AUC'),
                               'PR-AUC': AUC(curve='PR', name='PR-AUC'),
                               'Binary-Accuracy': BinaryAccuracy(name='Binary-Accuracy'),
                               'Precision': Precision(name='Precision'),
                               'Recall': Recall(name='Recall'),
                               'True-Positives': TruePositives(name='True-Positives'),
                               'False-Positives': FalsePositives(name='False-Positives'),
                               'False-Negatives': FalseNegatives(name='False-Negatives'),
                               'True-Negatives': TrueNegatives(name='True-Negatives')}
    
    @staticmethod
    def _append_ext(fn):
        return fn + ".jpg"
    
    def _load_data_features(self):
        for fold in self.folds:
            self.DATA_FEATURES[fold] = pd.read_csv(
                self.path_store + 'data-features_' + self.organ + '_' + self.view + '_' +
                self.dict_target_to_ids[self.target] + '_' + fold + '_' + self.outer_fold + '.csv')
            for col_name in self.id_vars + ['outer_fold']:
                self.DATA_FEATURES[fold][col_name] = self.DATA_FEATURES[fold][col_name].astype(str)
            self.DATA_FEATURES[fold].set_index('id', drop=False, inplace=True)
    
    def _take_subset_to_debug(self):
        for fold in self.folds:
            # use +1 or +2 to test the leftovers pipeline
            leftovers_extra = {'train': 0, 'val': 1, 'test': 2}
            n_batches = 2  # math.ceil(len(self.DATA_FEATURES[fold].index) / self.batch_size * self.debug_fraction)
            n_limit_fold = leftovers_extra[fold] + self.batch_size * n_batches
            self.DATA_FEATURES[fold] = self.DATA_FEATURES[fold].iloc[:n_limit_fold, :]
    
    def _generate_generators(self, DATA_FEATURES):
        GENERATORS = {}
        for fold in self.folds:
            # do not generate a generator if there are no samples (can happen for leftovers generators)
            if fold not in DATA_FEATURES.keys():
                continue
            # parameters
            training_mode = True if self.mode == 'model_training' else False
            if (fold == 'train') & (self.mode == 'model_training') & (self.organ in self.organs_to_augment):
                data_augmentation = True
            else:
                data_augmentation = False
            # define batch size for testing: data is split between a part that fits in batches, and leftovers
            if self.mode == 'model_testing':
                if self.organ in self.left_right_organs:
                    n_samples = len(DATA_FEATURES[fold].index) * 2
                else:
                    n_samples = len(DATA_FEATURES[fold].index)
                batch_size_fold = min(self.batch_size, n_samples)
            else:
                batch_size_fold = self.batch_size
            if (fold == 'train') & (self.mode == 'model_training'):
                n_samples_per_subepoch = self.n_samples_per_subepoch
            else:
                n_samples_per_subepoch = None
            # generator
            GENERATORS[fold] = MyImageDataGenerator(target=self.target, organ=self.organ,
                                                    data_features=DATA_FEATURES[fold],
                                                    n_samples_per_subepoch=n_samples_per_subepoch,
                                                    batch_size=batch_size_fold, training_mode=training_mode,
                                                    seed=self.seed, side_predictors=self.side_predictors,
                                                    dir_images=self.dir_images, images_width=self.image_width,
                                                    images_height=self.image_height,
                                                    data_augmentation=data_augmentation)
        return GENERATORS
    
    def _generate_class_weights(self):
        if self.dict_prediction_types[self.target] == 'binary':
            self.class_weights = {}
            counts = self.DATA_FEATURES['train'][self.target + '_raw'].value_counts()
            n_total = counts.sum()
            # weighting the samples for each class inversely proportional to their prevalence, with order of magnitude 1
            for i in counts.index.values:
                self.class_weights[i] = n_total / (counts.loc[i] * len(counts.index))
    
    def _generate_cnn(self):
        # define the arguments
        # take special initial weights for EfficientNetB7 (better)
        if (self.architecture == 'EfficientNetB7') & (self.keras_weights == 'imagenet'):
            w = 'noisy-student'
        else:
            w = self.keras_weights
        kwargs = {"include_top": False, "weights": w, "input_shape": (self.image_width, self.image_height, 3)}
        if self.architecture in ['ResNet50', 'ResNet101', 'ResNet152', 'ResNet50V2', 'ResNet101V2', 'ResNet152V2',
                                 'ResNeXt50', 'ResNeXt101']:
            import tensorflow.keras
            kwargs.update(
                {"backend": tensorflow.keras.backend, "layers": tensorflow.keras.layers,
                 "models": tensorflow.keras.models, "utils": tensorflow.keras.utils})
        
        # load the architecture builder
        if self.architecture == 'VGG16':
            from tensorflow.keras.applications.vgg16 import VGG16 as ModelBuilder
        elif self.architecture == 'VGG19':
            from tensorflow.keras.applications.vgg19 import VGG19 as ModelBuilder
        elif self.architecture == 'DenseNet121':
            from tensorflow.keras.applications.densenet import DenseNet121 as ModelBuilder
        elif self.architecture == 'DenseNet169':
            from tensorflow.keras.applications.densenet import DenseNet169 as ModelBuilder
        elif self.architecture == 'DenseNet201':
            from tensorflow.keras.applications.densenet import DenseNet201 as ModelBuilder
        elif self.architecture == 'Xception':
            from tensorflow.keras.applications.xception import Xception as ModelBuilder
        elif self.architecture == 'InceptionV3':
            from tensorflow.keras.applications.inception_v3 import InceptionV3 as ModelBuilder
        elif self.architecture == 'InceptionResNetV2':
            from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2 as ModelBuilder
        elif self.architecture == 'ResNet50':
            from keras_applications.resnet import ResNet50 as ModelBuilder
        elif self.architecture == 'ResNet101':
            from keras_applications.resnet import ResNet101 as ModelBuilder
        elif self.architecture == 'ResNet152':
            from keras_applications.resnet import ResNet152 as ModelBuilder
        elif self.architecture == 'ResNet50V2':
            from keras_applications.resnet_v2 import ResNet50V2 as ModelBuilder
        elif self.architecture == 'ResNet101V2':
            from keras_applications.resnet_v2 import ResNet101V2 as ModelBuilder
        elif self.architecture == 'ResNet152V2':
            from keras_applications.resnet_v2 import ResNet152V2 as ModelBuilder
        elif self.architecture == 'ResNeXt50':
            from keras_applications.resnext import ResNeXt50 as ModelBuilder
        elif self.architecture == 'ResNeXt101':
            from keras_applications.resnext import ResNeXt101 as ModelBuilder
        elif self.architecture == 'EfficientNetB7':
            from efficientnet.tfkeras import EfficientNetB7 as ModelBuilder
        # The following model have a fixed input size requirement
        elif self.architecture == 'NASNetMobile':
            from tensorflow.keras.applications.nasnet import NASNetMobile as ModelBuilder
        elif self.architecture == 'NASNetLarge':
            from tensorflow.keras.applications.nasnet import NASNetLarge as ModelBuilder
        elif self.architecture == 'MobileNet':
            from tensorflow.keras.applications.mobilenet import MobileNet as ModelBuilder
        elif self.architecture == 'MobileNetV2':
            from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as ModelBuilder
        else:
            print('Architecture does not exist.')
            sys.exit(1)
        
        # build the model's base
        cnn = ModelBuilder(**kwargs)
        x = cnn.output
        # complete the model's base
        if self.architecture in ['VGG16', 'VGG19']:
            x = Flatten()(x)
            x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
            x = Dropout(self.dropout_rate)(x)
            x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
            x = Dropout(self.dropout_rate)(x)
        else:
            x = GlobalAveragePooling2D()(x)
            if self.architecture == 'EfficientNetB7':
                x = Dropout(self.dropout_rate)(x)
        x = Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        cnn_output = Dropout(self.dropout_rate)(x)
        return cnn.input, cnn_output
    
    def _generate_side_nn(self):
        side_nn = Sequential()
        side_nn.add(Dense(128, input_dim=len(self.side_predictors), activation="relu"))
        side_nn.add(Dense(64, activation="relu"))
        side_nn.add(Dense(24, activation="relu"))
        return side_nn.input, side_nn.output
    
    def _complete_architecture(self, cnn_input, cnn_output, side_nn_input, side_nn_output):
        x = concatenate([cnn_output, side_nn_output])
        for n in [int(2 ** (10 - i)) for i in range(7)]:
            x = Dense(n, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
            # scale the dropout proportionally to the number of nodes in a layer. No dropout for the last layers
            if n > 64:
                x = Dropout(self.dropout_rate * n / 1024)(x)
        predictions = Dense(1, activation=self.dict_final_activations[self.prediction_type])(x)
        self.model = Model(inputs=[cnn_input, side_nn_input], outputs=predictions)
    
    def _generate_architecture(self):
        cnn_input, cnn_output = self._generate_cnn()
        side_nn_input, side_nn_output = self._generate_side_nn()
        self._complete_architecture(cnn_input=cnn_input, cnn_output=cnn_output, side_nn_input=side_nn_input,
                                    side_nn_output=side_nn_output)
    
    def _load_model_weights(self):
        try:
            self.model.load_weights(self.path_load_weights)
        except (FileNotFoundError, TypeError):
            # load backup weights if the main weights are corrupted
            try:
                self.model.load_weights(self.path_load_weights.replace('model-weights', 'backup-model-weights'))
            except FileNotFoundError:
                print('Error. No file was found. imagenet weights should have been used. Bug somewhere.')
                sys.exit(1)
    
    @staticmethod
    def clean_exit():
        # exit
        print('\nDone.\n')
        print('Killing JOB PID with kill...')
        os.system('touch ../eo/' + os.environ['SLURM_JOBID'])
        os.system('kill ' + str(os.getpid()))
        time.sleep(60)
        print('Escalating to kill JOB PID with kill -9...')
        os.system('kill -9 ' + str(os.getpid()))
        time.sleep(60)
        print('Escalating to kill JOB ID')
        os.system('scancel ' + os.environ['SLURM_JOBID'])
        time.sleep(60)
        print('Everything failed to kill the job. Hanging there until hitting walltime...')


class Training(DeepLearning):
    """
    Train models
    """
    
    def __init__(self, target=None, organ=None, view=None, transformation=None, architecture=None, optimizer=None,
                 learning_rate=None, weight_decay=None, dropout_rate=None, outer_fold=None, debug_mode=False,
                 max_transfer_learning=False, continue_training=True, display_full_metrics=True):
        # parameters
        DeepLearning.__init__(self, target, organ, view, transformation, architecture, optimizer, learning_rate,
                              weight_decay, dropout_rate, debug_mode)
        self.outer_fold = outer_fold
        self.version = self.version + '_' + str(self.outer_fold)
        # NNet's architecture's weights
        self.continue_training = continue_training
        self.max_transfer_learning = max_transfer_learning
        self.list_parameters_to_match = ['organ', 'transformation', 'view']
        # dict to decide in which order targets should be used when trying to transfer weight from a similar model
        self.dict_alternative_targets_for_transfer_learning = {'Age': ['Age', 'Sex'], 'Sex': ['Sex', 'Age']}
        
        # Generators
        self.folds = ['train', 'val']
        self.mode = 'model_training'
        self.class_weights = None
        self.GENERATORS = None
        
        # Metrics
        self.loss_name = self.dict_losses_names[self.prediction_type]
        self.loss_function = self.dict_losses_K[self.loss_name]
        self.main_metric_name = self.dict_main_metrics_names_K[self.target]
        self.main_metric_mode = self.main_metrics_modes[self.main_metric_name]
        self.main_metric = self.dict_metrics_K[self.main_metric_name]
        self.display_full_metrics = display_full_metrics
        if self.display_full_metrics:
            self.metrics_names = self.dict_metrics_names_K[self.prediction_type]
        else:
            self.metrics_names = [self.main_metric_name]
        self.metrics = [self.dict_metrics_K[metric_name] for metric_name in self.metrics_names]
        self.baseline_performance = None
        
        # Model
        self.path_load_weights = self.path_store + 'model-weights_' + self.version + '.h5'
        if self.debug_mode:
            self.path_save_weights = self.path_store + 'model-weights-debug.h5'
        else:
            self.path_save_weights = self.path_store + 'model-weights_' + self.version + '.h5'
        self.n_epochs_max = 100000
        self.callbacks = None
        self.optimizers = {'Adam': Adam, 'RMSprop': RMSprop, 'Adadelta': Adadelta}
    
    # Load and preprocess the data, build the generators
    def data_preprocessing(self):
        self._load_data_features()
        if self.debug_mode:
            self._take_subset_to_debug()
        self._generate_class_weights()
        self.GENERATORS = self._generate_generators(self.DATA_FEATURES)
    
    # Determine which weights to load, if any.
    def _weights_for_transfer_learning(self):
        print('Looking for models to transfer weights from...')
        
        # define parameters
        parameters = self._version_to_parameters(self.version)
        
        # continue training if possible
        if self.continue_training and os.path.exists(self.path_load_weights):
            print('Loading the weights from the model\'s previous training iteration.')
            return
        
        # Look for similar models, starting from very similar to less similar
        if self.max_transfer_learning:
            while True:
                # print('Matching models for the following criterias:');
                # print(['architecture', 'target'] + list_parameters_to_match)
                # start by looking for models trained on the same target, then move to other targets
                for target_to_load in self.dict_alternative_targets_for_transfer_learning[parameters['target']]:
                    # print('Target used: ' + target_to_load)
                    parameters_to_match = parameters.copy()
                    parameters_to_match['target'] = target_to_load
                    # load the ranked performances table to select the best performing model among the similar
                    # models available
                    path_performances_to_load = self.path_store + 'PERFORMANCES_ranked_' + parameters_to_match[
                        'target'] + '_' + 'val' + '.csv'
                    try:
                        Performances = pd.read_csv(path_performances_to_load)
                        Performances['organ'] = Performances['organ'].astype(str)
                    except FileNotFoundError:
                        # print("Could not load the file: " + path_performances_to_load)
                        break
                    # iteratively get rid of models that are not similar enough, based on the list
                    for parameter in ['architecture', 'target'] + self.list_parameters_to_match:
                        Performances = Performances[Performances[parameter] == parameters_to_match[parameter]]
                    # if at least one model is similar enough, load weights from the best of them
                    if len(Performances.index) != 0:
                        self.path_load_weights = self.path_store + 'model-weights_' + Performances['version'][0] + '.h5'
                        self.keras_weights = None
                        print('transfering the weights from: ' + self.path_load_weights)
                        return
                
                # if no similar model was found, try again after getting rid of the last selection criteria
                if len(self.list_parameters_to_match) == 0:
                    print('No model found for transfer learning.')
                    break
                self.list_parameters_to_match.pop()
        
        # Otherwise use imagenet weights to initialize
        print('Using imagenet weights.')
        # using string instead of None for path to not ge
        self.path_load_weights = None
        self.keras_weights = 'imagenet'
    
    def _compile_model(self):
        # if learning rate was reduced with success according to logger, start with this reduced learning rate
        path_logger = self.path_store + 'logger_' + self.version + '.csv'
        if os.path.exists(path_logger):
            try:
                logger = pd.read_csv(path_logger)
                best_log = \
                    logger[logger['val_' + self.main_metric_name] == logger['val_' + self.main_metric_name].max()]
                lr = best_log['learning_rate'].values[0]
            except pd.errors.EmptyDataError:
                os.remove(path_logger)
                lr = self.learning_rate
        else:
            lr = self.learning_rate
        self.model.compile(optimizer=self.optimizers[self.optimizer](lr=lr, clipnorm=1.0),
                           loss=self.loss_function, metrics=self.metrics)
    
    def _compute_baseline_performance(self):
        # calculate initial val_loss value
        if self.continue_training:
            idx_metric_name = ([self.loss_name] + self.metrics_names).index(self.main_metric_name)
            baseline_perfs = self.model.evaluate(self.GENERATORS['val'], steps=self.GENERATORS['val'].steps)
            self.baseline_performance = baseline_perfs[idx_metric_name]
        elif self.main_metric_mode == 'min':
            self.baseline_performance = np.Inf
        else:
            self.baseline_performance = -np.Inf
        print('Baseline validation ' + self.main_metric_name + ' = ' + str(self.baseline_performance))
    
    def _define_callbacks(self):
        if self.debug_mode:
            path_logger = self.path_store + 'logger-debug.csv'
            append = False
        else:
            path_logger = self.path_store + 'logger_' + self.version + '.csv'
            append = self.continue_training
        csv_logger = MyCSVLogger(path_logger, separator=',', append=append)
        model_checkpoint_backup = MyModelCheckpoint(self.path_save_weights.replace('model-weights',
                                                                                   'backup-model-weights'),
                                                    monitor='val_' + self.main_metric.name,
                                                    baseline=self.baseline_performance, verbose=1, save_best_only=True,
                                                    save_weights_only=True, mode=self.main_metric_mode,
                                                    save_freq='epoch')
        model_checkpoint = MyModelCheckpoint(self.path_save_weights,
                                             monitor='val_' + self.main_metric.name, baseline=self.baseline_performance,
                                             verbose=1, save_best_only=True, save_weights_only=True,
                                             mode=self.main_metric_mode, save_freq='epoch')
        patience_reduce_lr = 2 + self.GENERATORS['train'].steps
        reduce_lr_on_plateau = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=patience_reduce_lr, verbose=1,
                                                 mode='min', min_delta=0, cooldown=0, min_lr=0)
        early_stopping = EarlyStopping(monitor='val_' + self.main_metric.name, min_delta=0, patience=4, verbose=0,
                                       mode=self.main_metric_mode, baseline=self.baseline_performance)
        self.callbacks = [csv_logger, model_checkpoint_backup, model_checkpoint, early_stopping, reduce_lr_on_plateau]
    
    def build_model(self):
        self._weights_for_transfer_learning()
        self._generate_architecture()
        if self.keras_weights is None:
            self._load_model_weights()
        else:
            # save imagenet weights as default, in case no better weights are found
            self.model.save_weights(self.path_save_weights.replace('model-weights', 'backup-model-weights'))
            self.model.save_weights(self.path_save_weights)
        self._compile_model()
        self._compute_baseline_performance()
        self._define_callbacks()
    
    def train_model(self):
        # garbage collector
        _ = gc.collect()
        
        # train the model
        self.model.fit(self.GENERATORS['train'], steps_per_epoch=self.GENERATORS['train'].steps,
                       validation_data=self.GENERATORS['val'], validation_steps=self.GENERATORS['val'].steps,
                       shuffle=False, use_multiprocessing=False, workers=self.n_cpus, epochs=self.n_epochs_max,
                       class_weight=self.class_weights, callbacks=self.callbacks, verbose=1)


class PredictionsGenerate(DeepLearning):
    
    def __init__(self, target=None, organ=None, view=None, transformation=None, architecture=None, optimizer=None,
                 learning_rate=None, weight_decay=None, dropout_rate=None, debug_mode=False):
        # Initialize parameters
        DeepLearning.__init__(self, target, organ, view, transformation, architecture, optimizer, learning_rate,
                              weight_decay, dropout_rate, debug_mode)
        self.mode = 'model_testing'
        # Define dictionaries attributes for data, generators and predictions
        self.DATA_FEATURES_BATCH = {}
        self.DATA_FEATURES_LEFTOVERS = {}
        self.GENERATORS_BATCH = None
        self.GENERATORS_LEFTOVERS = None
        self.PREDICTIONS = {}
    
    def _split_batch_leftovers(self):
        # split the samples into two groups: what can fit into the batch size, and the leftovers.
        for fold in self.folds:
            n_leftovers = len(self.DATA_FEATURES[fold].index) % self.n_ids_batch
            if n_leftovers > 0:
                self.DATA_FEATURES_BATCH[fold] = self.DATA_FEATURES[fold].iloc[:-n_leftovers]
                self.DATA_FEATURES_LEFTOVERS[fold] = self.DATA_FEATURES[fold].tail(n_leftovers)
            else:
                self.DATA_FEATURES_BATCH[fold] = self.DATA_FEATURES[fold]  # special case for syntax if no leftovers
                if fold in self.DATA_FEATURES_LEFTOVERS.keys():
                    del self.DATA_FEATURES_LEFTOVERS[fold]
    
    def _generate_outerfolds_predictions(self):
        # prepare unscaling
        if self.target in self.targets_regression:
            mean_train = self.DATA_FEATURES['train'][self.target + '_raw'].mean()
            std_train = self.DATA_FEATURES['train'][self.target + '_raw'].std()
        else:
            mean_train, std_train = None, None
        # Generate predictions
        for fold in self.folds:
            print('Predicting samples from fold ' + fold + '.')
            print(str(len(self.DATA_FEATURES[fold].index)) + ' samples to predict.')
            print('Predicting batches: ' + str(len(self.DATA_FEATURES_BATCH[fold].index)) + ' samples.')
            pred_batch = self.model.predict(self.GENERATORS_BATCH[fold], steps=self.GENERATORS_BATCH[fold].steps, verbose=1)
            if fold in self.GENERATORS_LEFTOVERS.keys():
                print('Predicting leftovers: ' + str(len(self.DATA_FEATURES_LEFTOVERS[fold].index)) + ' samples.')
                pred_leftovers = self.model.predict(self.GENERATORS_LEFTOVERS[fold],
                                                    steps=self.GENERATORS_LEFTOVERS[fold].steps, verbose=1)
                pred_full = np.concatenate((pred_batch, pred_leftovers)).squeeze()
            else:
                pred_full = pred_batch.squeeze()
            print('Predicted a total of ' + str(len(pred_full)) + ' samples.')
            # take the average between left and right predictions for paired organs
            if self.organ in self.left_right_organs:
                pred_full = np.mean(pred_full.reshape(-1, 2), axis=1)
            # unscale predictions
            if self.target in self.targets_regression:
                pred_full = pred_full * std_train + mean_train
            # merge the predictions
            self.DATA_FEATURES[fold]['pred'] = pred_full
            if fold in self.PREDICTIONS.keys():
                self.PREDICTIONS[fold] = pd.concat([self.PREDICTIONS[fold], self.DATA_FEATURES[fold]])
            else:
                self.PREDICTIONS[fold] = self.DATA_FEATURES[fold]
            # format the dataframe
            self.PREDICTIONS[fold]['id'] = [ID.replace('.jpg', '') for ID in self.PREDICTIONS[fold]['id']]
    
    def _generate_and_concatenate_predictions(self):
        for outer_fold in self.outer_folds:
            self.outer_fold = outer_fold
            print('Predicting samples for the outer_fold = ' + self.outer_fold)
            self.path_load_weights = self.path_store + 'model-weights_' + self.version + '_' + self.outer_fold + '.h5'
            self._load_data_features()
            if self.debug_mode:
                self._take_subset_to_debug()
            self._load_model_weights()
            self._split_batch_leftovers()
            # generate the generators
            self.GENERATORS_BATCH = self._generate_generators(DATA_FEATURES=self.DATA_FEATURES_BATCH)
            if self.DATA_FEATURES_LEFTOVERS is not None:
                self.GENERATORS_LEFTOVERS = self._generate_generators(DATA_FEATURES=self.DATA_FEATURES_LEFTOVERS)
            self._generate_outerfolds_predictions()
    
    def _format_predictions(self):
        for fold in self.folds:
            # print the performance TODO check if is working
            perf_fun = self.dict_metrics_sklearn[self.dict_main_metrics_names[self.target]]
            perf = perf_fun(self.PREDICTIONS[fold][self.target + '_raw'], self.PREDICTIONS[fold]['pred'])
            print('The ' + fold + ' performance is: ' + str(perf))
            # format the predictions
            self.PREDICTIONS[fold].index.name = 'column_names'
            self.PREDICTIONS[fold] = self.PREDICTIONS[fold][['id', 'outer_fold', 'pred']]
    
    def generate_predictions(self):
        self._generate_architecture()
        self._generate_and_concatenate_predictions()
        self._format_predictions()
    
    def save_predictions(self):
        for fold in self.folds:
            self.PREDICTIONS[fold].to_csv(self.path_store + 'Predictions_instances_' + self.version + '_' + fold +
                                          '.csv', index=False)


class PredictionsMerge(Hyperparameters):
    
    def __init__(self, target=None, fold=None):
        
        Hyperparameters.__init__(self)
        
        # Define dictionaries attributes for data, generators and predictions
        self.target = target
        self.fold = fold
        self.data_features = None
        self.list_models = None
        self.Predictions_df = None
    
    def _load_data_features(self):
        self.data_features = pd.read_csv(self.path_store + 'data-features_instances.csv',
                                         usecols=self.id_vars + self.demographic_vars)
        for var in self.id_vars:
            self.data_features[var] = self.data_features[var].astype(str)
        self.data_features.set_index('id', drop=False, inplace=True)
        self.data_features.index.name = 'column_names'
    
    def _preprocess_data_features(self):
        # For the training set, each sample is predicted n_CV_outer_folds times, so prepare a larger dataframe
        if self.fold == 'train':
            df_all_folds = None
            for outer_fold in self.outer_folds:
                df_fold = self.data_features.copy()
                df_fold['outer_fold'] = outer_fold
                df_all_folds = df_fold if outer_fold == self.outer_folds[0] else df_all_folds.append(df_fold)
            self.data_features = df_all_folds
    
    def _list_models(self):
        # generate list of predictions that will be integrated in the Predictions dataframe
        self.list_models = glob.glob(self.path_store + 'Predictions_instances_' + self.target + '_*_' + self.fold +
                                     '.csv')
        # get rid of ensemble models
        self.list_models = [model for model in self.list_models
                            if not (('*' in model) | ('?' in model) | (',' in model))]
        self.list_models.sort()
    
    def preprocessing(self):
        self._load_data_features()
        self._preprocess_data_features()
        self._list_models()
    
    def merge_predictions(self):
        # merge the predictions
        print('There are ' + str(len(self.list_models)) + ' models to merge.')
        i = 0
        # define subgroups to accelerate merging process
        list_subgroups = list(set(['_'.join(model.split('_')[3:7]) for model in self.list_models]))
        for subgroup in list_subgroups:
            print('Merging models from the subgroup ' + subgroup)
            models_subgroup = [model for model in self.list_models if subgroup in model]
            Predictions_subgroup = None
            # merge the models one by one
            for file_name in models_subgroup:
                i += 1
                print('Merging the ' + str(i) + 'th model: ' +
                      file_name.replace(self.path_store + 'Predictions_instances_', '').replace('.csv', ''))
                # load csv and format the predictions
                prediction = pd.read_csv(self.path_store + file_name)
                print('raw prediction\'s shape: ' + str(prediction.shape))
                for var in ['id', 'outer_fold']:
                    prediction[var] = prediction[var].apply(str)
                version = '_'.join(file_name.split('_')[2:-1])
                prediction['outer_fold_' + version] = prediction['outer_fold']  # extra col to merge for fold == 'train'
                prediction.rename(columns={'pred': 'pred_' + version}, inplace=True)
                # merge data frames
                if Predictions_subgroup is None:
                    Predictions_subgroup = prediction
                elif self.fold == 'train':
                    Predictions_subgroup = Predictions_subgroup.merge(prediction, how='outer', on=['id', 'outer_fold'])
                else:
                    prediction = prediction.drop(['outer_fold'], axis=1)
                    # not supported for panda version > 0.23.4 for now
                    Predictions_subgroup = Predictions_subgroup.merge(prediction, how='outer', on=['id'])
                # print('prediction\'s shape: ' + str(prediction.shape))
            
            # merge group predictions data frames
            if self.Predictions_df is None:
                self.Predictions_df = Predictions_subgroup
            elif self.fold == 'train':
                self.Predictions_df = self.Predictions_df.merge(Predictions_subgroup, how='outer',
                                                                on=['id', 'outer_fold'])
            else:
                Predictions_subgroup = Predictions_subgroup.drop(['outer_fold'], axis=1)
                # not supported for panda version > 0.23.4 for now
                self.Predictions_df = self.Predictions_df.merge(Predictions_subgroup, how='outer', on=['id'])
            print('Predictions_df\'s shape: ' + str(self.Predictions_df.shape))
            # garbage collector
            gc.collect()
    
    def postprocessing(self):
        # get rid of useless rows in data_features before merging to keep the memory requirements as low as possible
        self.data_features = self.data_features[self.data_features['id'].isin(self.Predictions_df['id'].values)]
        # merge data_features and predictions
        if self.fold == 'train':
            self.Predictions_df = self.data_features.merge(self.Predictions_df, how='outer', on=['id', 'outer_fold'])
        else:
            # not supported for panda version > 0.23.4 for now
            self.Predictions_df = self.data_features.merge(self.Predictions_df, how='outer', on=['id'])
        
        # remove rows for which no prediction is available (should be none)
        subset_cols = [col for col in self.Predictions_df.columns if 'pred_' in col]
        self.Predictions_df.dropna(subset=subset_cols, how='all', inplace=True)
        
        # Format the dataframe
        self.Predictions_df.drop(['outer_fold'], axis=1, inplace=True)
        
        # Print the squared correlations between the target and the predictions
        ps = [p for p in self.Predictions_df.columns.values if 'pred' in p or p == self.target]
        perfs = (self.Predictions_df[ps].corr()[self.target] ** 2).sort_values(ascending=False)
        print('Squared correlations between the target and the predictions:')
        print(perfs)
    
    def save_merged_predictions(self):
        self.Predictions_df.to_csv(self.path_store + 'PREDICTIONS_withoutEnsembles_instances_' + self.target + '_' +
                                   self.fold + '.csv', index=False)


class PredictionsEids(Hyperparameters):
    
    def __init__(self, target=None, fold=None, ensemble_models=None, n_cpus=None, debug_mode=None):
        Hyperparameters.__init__(self)
        
        # Define dictionaries attributes for data, generators and predictions
        self.target = target
        self.fold = fold
        self.ensemble_models = self.convert_string_to_boolean(ensemble_models)
        self.n_cpus = int(n_cpus)
        self.debug_mode = debug_mode
        self.Predictions = None
        self.pred_versions = None
        self.res_versions = None
        self.outer_fold_versions = None
        self.target_0s = None
        self.Predictions_eids = None
    
    def preprocessing(self):
        # Load predictions
        if self.ensemble_models:
            self.Predictions = pd.read_csv(
                self.path_store + 'PREDICTIONS_withEnsembles_instances_' + self.target + '_' + self.fold + '.csv')
            cols_to_drop = [col for col in self.Predictions.columns.values
                            if any(s in col for s in ['pred_', 'outer_fold_']) &
                            (not any(c in col for c in self.ensemble_types))]
            self.Predictions.drop(cols_to_drop, axis=1, inplace=True)
        else:
            self.Predictions = pd.read_csv(
                self.path_store + 'PREDICTIONS_withoutEnsembles_instances_' + self.target + '_' + self.fold + '.csv')
        self.outer_fold_versions = [of for of in self.Predictions.columns.values if 'outer_fold_' in of]
        for col in self.id_vars:
            self.Predictions[col] = self.Predictions[col].astype(str)
        self.Predictions.set_index('id', drop=False, inplace=True)
        self.Predictions.index.name = 'column_names'
        self.pred_versions = [of.replace('outer_fold_', 'pred_') for of in self.outer_fold_versions]
        self.res_versions = [of.replace('outer_fold_', 'res_') for of in self.outer_fold_versions]
        
        # Prepare target values on instance 0 as a reference
        target_0s = pd.read_csv(self.path_store + 'data-features_eids.csv', usecols=['eid', self.target])
        target_0s.eid = target_0s.eid.astype(str)
        target_0s.set_index('eid', inplace=True)
        self.target_0s = target_0s[self.target]
        
        # Generate an empty dataframe to store the mean prediction value for each eid (corrected for the target)
        row_names = self.Predictions['eid'].unique()
        col_names = self.Predictions.columns.values
        Predictions_eids = np.empty((len(row_names), len(col_names),))
        Predictions_eids.fill(np.nan)
        Predictions_eids = pd.DataFrame(Predictions_eids)
        Predictions_eids.index = row_names + '_*'
        Predictions_eids.columns = col_names
        Predictions_eids['id'] = row_names + '_*'
        Predictions_eids['eid'] = row_names
        Predictions_eids['instance'] = '*'
        self.Predictions_eids = Predictions_eids
        
        # Compute residuals
        for pred in [pred for pred in self.Predictions.columns.values if 'pred_' in pred]:
            self.Predictions[pred.replace('pred_', 'res_')] = self.Predictions['Age'] - self.Predictions[pred]
    
    def average_predictions_parallel(self):
        PA_split = np.array_split(self.Predictions_eids, self.n_cpus)
        pool = Pool(self.n_cpus)
        print('Parallelizing the processing over ' + str(self.n_cpus) + ' CPUs. ' +
              str(len(self.Predictions_eids.index)) + ' eids must be processed.')
        self.Predictions_eids = pd.concat(pool.map(self.average_predictions, PA_split))
        pool.close()
        pool.join()
    
    def average_predictions(self, PA_fold):
        eids = PA_fold['eid'].unique()
        print('Starting... In one of the CPUs ' + str(len(eids)) + ' eids are to be processed.')
        for i, eid in enumerate(eids):
            if i % 1000 == 0:
                print('In one of the CPUs, ' + str(i) + ' eids have been processed.')
            Preds_eid = self.Predictions[self.Predictions['eid'] == eid]
            PA_fold.loc[eid + '_*', self.demographic_vars] = \
                Preds_eid[self.demographic_vars].mean().values
            target_eid = self.target_0s[eid]
            PA_fold.loc[eid + '_*', self.target] = target_eid
            Mean_res = Preds_eid[self.res_versions].mean(skipna=True)
            PA_fold.loc[eid + '_*', self.pred_versions] = target_eid - Mean_res.values
            PA_fold.loc[eid + '_*', self.outer_fold_versions] = Preds_eid[self.outer_fold_versions].mean(skipna=True)
        print('Completed. In one of the CPUs, all the ' + str(len(eids)) + ' eids have been processed.')
        return PA_fold
    
    def postprocessing(self):
        # For ensemble models, append the new models to the non ensemble models
        if self.ensemble_models:
            # Only keep columns that are not already in the previous dataframe
            cols_to_keep = [col for col in self.Predictions.columns.values
                            if any(s in col for s in ['pred_', 'outer_fold_'])]
            self.Predictions_eids = self.Predictions_eids[cols_to_keep]
            Predictions_withoutEnsembles = pd.read_csv(
                self.path_store + 'PREDICTIONS_withoutEnsembles_eids_' + self.target + '_' + self.fold + '.csv')
            for var in self.id_vars:
                Predictions_withoutEnsembles[var] = Predictions_withoutEnsembles[var].astype(str)
            Predictions_withoutEnsembles.set_index('id', drop=False, inplace=True)
            # Reorder the rows
            self.Predictions_eids = self.Predictions_eids.loc[Predictions_withoutEnsembles.index.values, :]
            self.Predictions_eids = pd.concat([Predictions_withoutEnsembles, self.Predictions_eids], axis=1)
        
        # Print the squared correlations between the target and the predictions
        ps = [p for p in self.Predictions_eids.columns.values if 'pred' in p or p == self.target]
        perfs = (self.Predictions_eids[ps].corr()[self.target] ** 2).sort_values(ascending=False)
        print('Squared correlations between the target and the predictions:')
        print(perfs)
    
    def _generate_single_model_predictions(self):
        for pred_version in self.pred_versions:
            outer_version = pred_version.replace('pred_', 'outer_fold_')
            Predictions_version = self.Predictions_eids[['id', outer_version, pred_version]]
            Predictions_version.rename(columns={outer_version: 'outer_fold', pred_version: 'pred'}, inplace=True)
            Predictions_version.dropna(inplace=True)
            Predictions_version.to_csv(self.path_store + 'Predictions_eids_' + '_'.join(pred_version.split('_')[2:]) +
                                       '_' + self.fold + '.csv', index=False)
    
    def save_predictions(self):
        mode = 'withEnsembles' if self.ensemble_models else 'withoutEnsembles'
        self.Predictions_eids.to_csv(self.path_store + 'PREDICTIONS_' + mode + '_eids_' + self.target + '_'
                                     + self.fold + '.csv', index=False)
        # Generate and save files for every single model
        self._generate_single_model_predictions()


class PerformancesGenerate(Metrics):
    
    def __init__(self, target=None, organ=None, view=None, transformation=None, architecture=None, optimizer=None,
                 learning_rate=None, weight_decay=None, dropout_rate=None, fold=None, pred_type=None, debug_mode=False):
        
        Metrics.__init__(self)
        
        self.target = target
        self.organ = organ
        self.view = view
        self.transformation = transformation
        self.architecture = architecture
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.fold = fold
        self.pred_type = pred_type
        
        if debug_mode:
            self.n_bootstrap_iterations = 10
        else:
            self.n_bootstrap_iterations = 1000
        
        if type(learning_rate) == str:
            learning_rate_version = learning_rate
        else:
            learning_rate_version = np.format_float_positional(learning_rate)
        weight_decay_version = weight_decay if type(weight_decay) == str else str(weight_decay)
        dropout_rate_version = dropout_rate if type(dropout_rate) == str else str(dropout_rate)
        self.version = self.target + '_' + self.organ + '_' + self.view + '_' + self.transformation + '_' + \
                       self.architecture + '_' + self.optimizer + '_' + learning_rate_version + '_' + \
                       weight_decay_version + '_' + dropout_rate_version
        self.names_metrics = self.dict_metrics_names[self.dict_prediction_types[target]]
        self.data_features = None
        self.Predictions = None
        self.PERFORMANCES = None
    
    def _preprocess_data_features_predictions_for_performances(self):
        # load dataset
        data_features = pd.read_csv(self.path_store + 'data-features_' + self.pred_type + '.csv',
                                    usecols=['id', 'Sex', 'Age'])
        # format data_features to extract y
        data_features.rename(columns={self.target: 'y'}, inplace=True)
        data_features = data_features[['id', 'y']]
        data_features['id'] = data_features['id'].astype(str)
        data_features['id'] = data_features['id']
        data_features.set_index('id', drop=False, inplace=True)
        data_features.index.name = 'columns_names'
        self.data_features = data_features
    
    def _preprocess_predictions_for_performances(self):
        Predictions = pd.read_csv(self.path_store + 'Predictions_' + self.pred_type + '_' + self.version + '_' +
                                  self.fold + '.csv')
        Predictions['id'] = Predictions['id'].astype(str)
        #Predictions.rename(columns={'Pred_' + self.version: 'pred'}, inplace=True) TODO delete
        self.Predictions = Predictions.merge(self.data_features, how='inner', on=['id'])
    
    # Initialize performances dataframes and compute sample sizes
    def _initiate_empty_performances_df(self):
        # Define an empty performances dataframe to store the performances computed
        row_names = ['all'] + self.outer_folds
        col_names_sample_sizes = ['N']
        if self.target in self.targets_binary:
            col_names_sample_sizes.extend(['N_0', 'N_1'])
        col_names = ['outer_fold'] + col_names_sample_sizes
        col_names.extend(self.names_metrics)
        performances = np.empty((len(row_names), len(col_names),))
        performances.fill(np.nan)
        performances = pd.DataFrame(performances)
        performances.index = row_names
        performances.columns = col_names
        performances['outer_fold'] = row_names
        # Convert float to int for sample sizes and some metrics.
        for col_name in col_names_sample_sizes:
            # need recent version of pandas to use type below. Otherwise nan cannot be int
            performances[col_name] = performances[col_name].astype('Int64')
        
        # compute sample sizes for the data frame
        performances.loc['all', 'N'] = len(self.Predictions.index)
        if self.target in self.targets_binary:
            performances.loc['all', 'N_0'] = len(self.Predictions.loc[self.Predictions['y'] == 0].index)
            performances.loc['all', 'N_1'] = len(self.Predictions.loc[self.Predictions['y'] == 1].index)
        for outer_fold in self.outer_folds:
            performances.loc[outer_fold, 'N'] = len(
                self.Predictions.loc[self.Predictions['outer_fold'] == int(outer_fold)].index)
            if self.target in self.targets_binary:
                performances.loc[outer_fold, 'N_0'] = len(
                    self.Predictions.loc[
                        (self.Predictions['outer_fold'] == int(outer_fold)) & (self.Predictions['y'] == 0)].index)
                performances.loc[outer_fold, 'N_1'] = len(
                    self.Predictions.loc[
                        (self.Predictions['outer_fold'] == int(outer_fold)) & (self.Predictions['y'] == 1)].index)
        
        # initialize the dataframes
        self.PERFORMANCES = {}
        for mode in self.modes:
            self.PERFORMANCES[mode] = performances.copy()
        
        # Convert float to int for sample sizes and some metrics.
        for col_name in self.PERFORMANCES[''].columns.values:
            if any(metric in col_name for metric in self.metrics_displayed_in_int):
                # need recent version of pandas to use type below. Otherwise nan cannot be int
                self.PERFORMANCES[''][col_name] = self.PERFORMANCES[''][col_name].astype('Int64')
    
    def preprocessing(self):
        self._preprocess_data_features_predictions_for_performances()
        self._preprocess_predictions_for_performances()
        self._initiate_empty_performances_df()
    
    def _bootstrap(self, data, function):
        results = []
        for i in range(self.n_bootstrap_iterations):
            data_i = resample(data, replace=True, n_samples=len(data.index))
            results.append(function(data_i['y'], data_i['pred']))
        return np.mean(results), np.std(results)
    
    # Fill the columns for this model, outer_fold by outer_fold
    def compute_performances(self):
        
        # fill it outer_fold by outer_fold
        for outer_fold in ['all'] + self.outer_folds:
            print('Calculating the performances for the outer fold ' + outer_fold)
            # Generate a subdataframe from the predictions table for each outerfold
            if outer_fold == 'all':
                predictions_fold = self.Predictions.copy()
            else:
                predictions_fold = self.Predictions.loc[self.Predictions['outer_fold'] == int(outer_fold), :]
            
            # if no samples are available for this fold, fill columns with nans
            if len(predictions_fold.index) == 0:
                print('NO SAMPLES AVAILABLE FOR MODEL ' + self.version + ' IN OUTER_FOLD ' + outer_fold)
            else:
                # For binary classification, generate class prediction
                if self.target in self.targets_binary:
                    predictions_fold_class = predictions_fold.copy()
                    predictions_fold_class['pred'] = predictions_fold_class['pred'].round()
                else:
                    predictions_fold_class = None
                
                # Fill the Performances dataframe metric by metric
                for name_metric in self.names_metrics:
                    # print('Calculating the performance using the metric ' + name_metric)
                    if name_metric in self.metrics_needing_classpred:
                        predictions_metric = predictions_fold_class
                    else:
                        predictions_metric = predictions_fold
                    metric_function = self.dict_metrics_sklearn[name_metric]
                    self.PERFORMANCES[''].loc[outer_fold, name_metric] = metric_function(predictions_metric['y'],
                                                                                         predictions_metric['pred'])
                    self.PERFORMANCES['_sd'].loc[outer_fold, name_metric] = \
                        self._bootstrap(predictions_metric, metric_function)[1]
                    self.PERFORMANCES['_str'].loc[outer_fold, name_metric] = "{:.3f}".format(
                        self.PERFORMANCES[''].loc[outer_fold, name_metric]) + '+-' + "{:.3f}".format(
                        self.PERFORMANCES['_sd'].loc[outer_fold, name_metric])
        
        # calculate the fold sd (variance between the metrics values obtained on the different folds)
        folds_sd = self.PERFORMANCES[''].iloc[1:, :].std(axis=0)
        for name_metric in self.names_metrics:
            self.PERFORMANCES['_str'].loc['all', name_metric] = "{:.3f}".format(
                self.PERFORMANCES[''].loc['all', name_metric]) + '+-' + "{:.3f}".format(
                folds_sd[name_metric]) + '+-' + "{:.3f}".format(self.PERFORMANCES['_sd'].loc['all', name_metric])
        
        # print the performances
        print('Performances for model ' + self.version + ': ')
        print(self.PERFORMANCES['_str'])
    
    def save_performances(self):
        for mode in self.modes:
            path_save = self.path_store + 'Performances_' + self.pred_type + '_' + self.version + '_' + self.fold + \
                        mode + '.csv'
            self.PERFORMANCES[mode].to_csv(path_save, index=False)


class PerformancesMerge(Metrics):
    
    def __init__(self, target=None, fold=None, pred_type=None, ensemble_models=None):
        
        # Parameters
        Metrics.__init__(self)
        self.target = target
        self.fold = fold
        self.pred_type = pred_type
        self.ensemble_models = self.convert_string_to_boolean(ensemble_models)
        self.names_metrics = self.dict_metrics_names[self.dict_prediction_types[self.target]]
        # list the models that need to be merged
        self.list_models = glob.glob(self.path_store + 'Performances_' + self.pred_type + '_' + self.target + '_*_' +
                                     self.fold + '_str.csv')
        # get rid of ensemble models
        if self.ensemble_models:
            self.list_models = [model for model in self.list_models
                                if (('*' in model) | ('?' in model) | (',' in model))]
        else:
            self.list_models = [model for model in self.list_models
                                if not (('*' in model) | ('?' in model) | (',' in model))]
        self.Performances = None
        self.Performances_alphabetical = None
        self.Performances_ranked = None
    
    def _initiate_empty_performances_summary_df(self):
        # Define the columns of the Performances dataframe
        # columns for sample sizes
        names_sample_sizes = ['N']
        if self.target in self.targets_binary:
            names_sample_sizes.extend(['N_0', 'N_1'])
        
        # columns for metrics
        names_metrics = self.dict_metrics_names[self.dict_prediction_types[self.target]]
        # for normal folds, keep track of metric and bootstrapped metric's sd
        names_metrics_with_sd = []
        for name_metric in names_metrics:
            names_metrics_with_sd.extend([name_metric, name_metric + '_sd', name_metric + '_str'])
        
        # for the 'all' fold, also keep track of the 'folds_sd' (metric's sd calculated using the folds' results)
        names_metrics_with_folds_sd_and_sd = []
        for name_metric in names_metrics:
            names_metrics_with_folds_sd_and_sd.extend(
                [name_metric, name_metric + '_folds_sd', name_metric + '_sd', name_metric + '_str'])
        
        # merge all the columns together. First description of the model, then sample sizes and metrics for each fold
        names_col_Performances = ['version'] + self.names_model_parameters  # .copy() TODO remove
        # special outer fold 'all'
        names_col_Performances.extend(
            ['_'.join([name, 'all']) for name in names_sample_sizes + names_metrics_with_folds_sd_and_sd])
        # other outer_folds
        for outer_fold in self.outer_folds:
            names_col_Performances.extend(
                ['_'.join([name, outer_fold]) for name in names_sample_sizes + names_metrics_with_sd])
        
        # Generate the empty Performance table from the rows and columns.
        Performances = np.empty((len(self.list_models), len(names_col_Performances),))
        Performances.fill(np.nan)
        Performances = pd.DataFrame(Performances)
        Performances.columns = names_col_Performances
        # Format the types of the columns
        for colname in Performances.columns.values:
            if (colname in self.names_model_parameters) | ('_str' in colname):
                col_type = str
            else:
                col_type = float
            Performances[colname] = Performances[colname].astype(col_type)
        self.Performances = Performances
    
    def merge_performances(self):
        # define parameters
        names_metrics = self.dict_metrics_names[self.dict_prediction_types[self.target]]
        
        # initiate dataframe
        self._initiate_empty_performances_summary_df()
        
        # Fill the Performance table row by row
        for i, model in enumerate(self.list_models):
            # load the performances subdataframe
            PERFORMANCES = {}
            for mode in self.modes:
                PERFORMANCES[mode] = pd.read_csv(model.replace('_str', mode))
                PERFORMANCES[mode].set_index('outer_fold', drop=False, inplace=True)
            
            # Fill the columns corresponding to the model's parameters
            version = '_'.join(model.split('_')[2:-2])
            parameters = self._version_to_parameters(version)
            
            # fill the columns for model parameters
            self.Performances['version'][i] = version
            for parameter_name in self.names_model_parameters:
                self.Performances[parameter_name][i] = parameters[parameter_name]
            
            # Fill the columns for this model, outer_fold by outer_fold
            for outer_fold in ['all'] + self.outer_folds:
                # Generate a subdataframe from the predictions table for each outerfold
                
                # Fill sample size columns
                self.Performances['N_' + outer_fold][i] = PERFORMANCES[''].loc[outer_fold, 'N']
                
                # For binary classification, calculate sample sizes for each class and generate class prediction
                if self.target in self.targets_binary:
                    self.Performances['N_0_' + outer_fold][i] = PERFORMANCES[''].loc[outer_fold, 'N_0']
                    self.Performances['N_1_' + outer_fold][i] = PERFORMANCES[''].loc[outer_fold, 'N_1']
                
                # Fill the Performances dataframe metric by metric
                for name_metric in names_metrics:
                    for mode in self.modes:
                        self.Performances[name_metric + mode + '_' + outer_fold][i] = PERFORMANCES[mode].loc[
                            outer_fold, name_metric]
                
                # calculate the fold sd (variance between the metrics values obtained on the different folds)
                folds_sd = PERFORMANCES[''].iloc[1:, :].std(axis=0)
                for name_metric in names_metrics:
                    self.Performances[name_metric + '_folds_sd_all'] = folds_sd[name_metric]
        
        # Convert float to int for sample sizes and some metrics.
        for name_col in self.Performances.columns.values:
            cond1 = name_col.startswith('N_')
            cond2 = any(metric in name_col for metric in self.metrics_displayed_in_int)
            cond3 = '_sd' not in name_col
            cond4 = '_str' not in name_col
            if cond1 | cond2 & cond3 & cond4:
                self.Performances[name_col] = self.Performances[name_col].astype('Int64')
                # need recent version of pandas to use this type. Otherwise nan cannot be int
        
        # For ensemble models, merge the new performances with the previously computed performances
        if self.ensemble_models:
            Performances_withoutEnsembles = pd.read_csv(self.path_store + 'PERFORMANCES_tuned_alphabetical_' +
                                                        self.pred_type + '_' + self.target + '_' + self.fold + '.csv')
            self.Performances = Performances_withoutEnsembles.append(self.Performances)
            # reorder the columns (weird: automatic alphabetical re-ordering happened when append was called for 'val')
            self.Performances = self.Performances[Performances_withoutEnsembles.columns]
        
        # Ranking, printing and saving
        self.Performances_alphabetical = self.Performances.sort_values(by='version')
        print('Performances of the models ranked by models\'names:')
        print(self.Performances_alphabetical)
        sort_by = self.dict_main_metrics_names[self.target] + '_all'
        sort_ascending = self.main_metrics_modes[self.dict_main_metrics_names[self.target]] == 'min'
        self.Performances_ranked = self.Performances.sort_values(by=sort_by, ascending=sort_ascending)
        print('Performances of the models ranked by the performance on the main metric on all the samples:')
        print(self.Performances_ranked)
    
    def save_performances(self):
        name_extension = 'withEnsembles' if self.ensemble_models else 'withoutEnsembles'
        path = self.path_store + 'PERFORMANCES_' + name_extension + '_alphabetical_' + self.pred_type + '_' + \
               self.target + '_' + self.fold + '.csv'
        self.Performances_alphabetical.to_csv(path, index=False)
        self.Performances_ranked.to_csv(path.replace('_alphabetical_', '_ranked_'), index=False)


class PerformancesTuning(Metrics):
    
    def __init__(self, target=None, pred_type=None):
        
        Metrics.__init__(self)
        self.target = target
        self.pred_type = pred_type
        self.PERFORMANCES = {}
        self.PREDICTIONS = {}
        self.Performances = None
        self.models = None
    
    def load_data(self):
        for fold in self.folds:
            path = self.path_store + 'PERFORMANCES_withoutEnsembles_ranked_' + self.pred_type + '_' + self.target + '_'\
                   + fold + '.csv'
            self.PERFORMANCES[fold] = pd.read_csv(path).set_index('version', drop=False)
            self.PERFORMANCES[fold]['organ'] = self.PERFORMANCES[fold]['organ'].astype(str)
            self.PERFORMANCES[fold].index.name = 'columns_names'
            self.PREDICTIONS[fold] = pd.read_csv(path.replace('PERFORMANCES', 'PREDICTIONS').replace('_ranked', ''))
    
    def preprocess_data(self):
        # Get list of distinct models without taking into account hyperparameters tuning
        self.Performances = self.PERFORMANCES['val']
        self.Performances['model'] = self.Performances['organ'] + '_' + self.Performances['view'] + '_' + \
                                     self.Performances['transformation'] + '_' + self.Performances['architecture']
        self.models = self.Performances['model'].unique()
    
    def select_models(self):
        main_metric_name = self.dict_main_metrics_names[self.target]
        Perf_col_name = main_metric_name + '_all'
        for model in self.models:
            Performances_model = self.Performances[self.Performances['model'] == model]
            best_version = Performances_model['version'][
                Performances_model[Perf_col_name] == Performances_model[Perf_col_name].max()].values[0]
            versions_to_drop = [version for version in Performances_model['version'].values if
                                not version == best_version]
            # define columns from predictions to drop
            cols_to_drop = ['pred_' + version for version in versions_to_drop] + ['outer_fold_' + version for version in
                                                                                  versions_to_drop]
            for fold in self.folds:
                self.PERFORMANCES[fold].drop(versions_to_drop, inplace=True)
                self.PREDICTIONS[fold].drop(cols_to_drop, inplace=True, axis=1)
        
        # drop 'model' column
        self.Performances.drop(['model'], inplace=True, axis=1)

        # Display results
        for fold in self.folds:
            print('The tuned ' + fold + ' performances are:')
            print(self.PERFORMANCES[fold])
    
    def save_data(self):
        # Save the files
        for fold in self.folds:
            path_pred = self.path_store + 'PREDICTIONS_tuned_' + self.pred_type + '_' + self.target + '_' + fold + \
                        '.csv'
            path_perf = self.path_store + 'PERFORMANCES_tuned_ranked_' + self.pred_type + '_' + self.target + '_' + \
                        fold + '.csv'
            self.PREDICTIONS[fold].to_csv(path_pred, index=False)
            self.PERFORMANCES[fold].to_csv(path_perf, index=False)
            Performances_alphabetical = self.PERFORMANCES[fold].sort_values(by='version')
            Performances_alphabetical.to_csv(path_perf.replace('ranked', 'alphabetical'), index=False)


class EnsemblesPredictions(Metrics):
    
    def __init__(self, target=None, pred_type=None):
        # Parameters
        Metrics.__init__(self)
        self.target = target
        self.pred_type = pred_type
        self.ensembles_performance_cutoff_percent = 0
        self.parameters = {'target': self.target, 'organ': '*', 'view': '*', 'transformation': '*',
                           'architecture': '*', 'optimizer': '*', 'learning_rate': '*', 'weight_decay': '*',
                           'dropout_rate': '*'}
        self.version = self._parameters_to_version(self.parameters)
        self.main_metric_name = self.dict_main_metrics_names[target]
        self.init_perf = -np.Inf if self.main_metrics_modes[self.main_metric_name] == 'max' else np.Inf
        path_perf = self.path_store + 'PERFORMANCES_tuned_ranked_' + self.pred_type + '_' + self.target + '_val.csv'
        self.Performances = pd.read_csv(path_perf).set_index('version', drop=False)
        self.Performances['organ'] = self.Performances['organ'].astype(str)
        self.list_ensemble_levels = ['transformation', 'view', 'organ']
        self.PREDICTIONS = {}
        self.weights_by_category = None
        self.weights_by_ensembles = None
    
    # Get rid of columns and rows for the versions for which all samples as NANs
    def _drop_na_pred_versions(self, PREDS, Performances):
        # Select the versions for which only NAs are available
        pred_versions = [col for col in PREDS['train'].columns.values if 'pred_' in col]
        to_drop = []
        for pv in pred_versions:
            for fold in PREDS.keys():
                if PREDS[fold][pv].notna().sum() == 0:
                    to_drop.append(pv)
                    break
        # Drop the corresponding columns from preds, and rows from performances
        of_to_drop = [p.replace('pred_', 'outer_fold_') for p in to_drop]
        index_to_drop = [p.replace('pred_', '') for p in to_drop
                         if not np.any([c in p for c in self.ensemble_types])]
        for fold in PREDS.keys():
            PREDS[fold].drop(to_drop + of_to_drop, axis=1, inplace=True)
        return Performances.drop(index_to_drop)
    
    def load_data(self):
        for fold in self.folds:
            self.PREDICTIONS[fold] = pd.read_csv(
                self.path_store + 'PREDICTIONS_tuned_' + self.pred_type + '_' + self.target + '_' + fold + '.csv')
    
    def _weighted_weights_by_category(self, weights, Performances, ensemble_level):
        weights_names = weights.index.values
        for category in Performances[ensemble_level].unique():
            n_category = len([name for name in weights_names if category in name])
            for weight_name in weights.index.values:
                if category in weight_name:
                    weights[weight_name] = weights[weight_name] / n_category
        self.weights_by_category = weights.values / weights.values.sum()
    
    def _weighted_weights_by_ensembles(self, Predictions, Performances, parameters, ensemble_level):
        sub_levels = Performances[ensemble_level].unique()
        self.sub_ensemble_cols = []
        weights = []
        for sub in sub_levels:
            parameters_sub = parameters.copy()
            parameters_sub[ensemble_level] = sub
            version_sub = self._parameters_to_version(parameters_sub)
            self.sub_ensemble_cols.append('pred_' + version_sub)
            df_score = Predictions[[parameters['target'], 'pred_' + version_sub]]
            df_score.dropna(inplace=True)
            weight = self.dict_metrics_sklearn[self.main_metric_name](df_score[parameters['target']],
                                                                      df_score['pred_' + version_sub])
            weights.append(weight)
        weights = np.array(weights)
        self.weights_by_ensembles = weights / weights.sum()
    
    def _build_single_ensemble(self, PREDICTIONS, Performances, parameters, version, list_ensemble_levels,
                               ensemble_level):
        # define which models should be integrated into the ensemble model, and how they should be weighted
        Predictions = PREDICTIONS['val']
        performance_cutoff = np.max(Performances[self.main_metric_name + '_all']) \
                             * self.ensembles_performance_cutoff_percent
        ensemble_namecols = ['pred_' + model_name for model_name in
                             Performances['version'][Performances[self.main_metric_name + '_all'] > performance_cutoff]]
        
        # calculate the ensemble model using three different kinds of weights
        # weighted by performance
        weights_with_names = Performances[self.main_metric_name + '_all'][
            Performances[self.main_metric_name + '_all'] > performance_cutoff]
        weights = weights_with_names.values / weights_with_names.values.sum()
        if len(list_ensemble_levels) > 0:
            # weighted by both performance and subcategories
            self._weighted_weights_by_category(weights_with_names, Performances, ensemble_level)
            # weighted by the performance of the ensemble models right below it
            self._weighted_weights_by_ensembles(Predictions, Performances, parameters, ensemble_level)
        
        # for each fold, build the ensemble model
        for fold in self.folds:
            Ensemble_predictions = PREDICTIONS[fold][ensemble_namecols] * weights
            PREDICTIONS[fold]['pred_' + version] = Ensemble_predictions.sum(axis=1, skipna=False)
            if len(list_ensemble_levels) > 0:
                Ensemble_predictions_weighted_by_category = \
                    PREDICTIONS[fold][ensemble_namecols] * self.weights_by_category
                Ensemble_predictions_weighted_by_ensembles = \
                    PREDICTIONS[fold][self.sub_ensemble_cols] * self.weights_by_ensembles
                PREDICTIONS[fold]['pred_' + version.replace('*', '?')] = \
                    Ensemble_predictions_weighted_by_category.sum(axis=1, skipna=False)
                PREDICTIONS[fold]['pred_' + version.replace('*', ',')] = \
                    Ensemble_predictions_weighted_by_ensembles.sum(axis=1, skipna=False)
    
    def _build_single_ensemble_wrapper(self, Performances, parameters, version, list_ensemble_levels, ensemble_level):
        Predictions = self.PREDICTIONS['val']
        # Select the outerfolds columns for the model
        ensemble_outerfolds_cols = [name_col for name_col in Predictions.columns.values if
                                    bool(re.compile('outer_fold_' + version).match(name_col))]
        Ensemble_outerfolds = Predictions[ensemble_outerfolds_cols]
        
        # Evaluate if the ensemble model should be built
        # 1 - separately on instance 0-1 and 2-3 (for ensemble at the top level, since overlap between is 0 otherwise)
        # 2 - piece by piece on each outer_fold
        # 3 - on all the folds at once (if the folds are not shared)
        if (ensemble_level == 'organ') & (self.pred_type == 'instances'):  # 1-Compute instances 0-1 and 2-3 separately
            PREDICTIONS_01 = {}
            PREDICTIONS_23 = {}
            for fold in self.folds:
                PREDICTIONS_01[fold] = self.PREDICTIONS[fold][self.PREDICTIONS[fold].instance.isin(['0', '1'])]
                PREDICTIONS_23[fold] = self.PREDICTIONS[fold][self.PREDICTIONS[fold].instance.isin(['2', '3'])]
            Performances_01 = self._drop_na_pred_versions(PREDICTIONS_01, Performances)
            Performances_23 = self._drop_na_pred_versions(PREDICTIONS_23, Performances)
            self._build_single_ensemble(PREDICTIONS_01, Performances_01, parameters, version, list_ensemble_levels,
                                        ensemble_level)
            self._build_single_ensemble(PREDICTIONS_23, Performances_23, parameters, version, list_ensemble_levels,
                                        ensemble_level)
            for fold in self.folds:
                for ensemble_type in self.ensemble_types:
                    self.PREDICTIONS[fold]['outer_fold_' + version.replace('*', ensemble_type)] = np.nan
                    pred_version = 'pred_' + version.replace('*', ensemble_type)
                    self.PREDICTIONS[fold][pred_version] = np.nan
                    self.PREDICTIONS[fold][pred_version][self.PREDICTIONS[fold].instance.isin(['0', '1'])] = \
                        PREDICTIONS_01[fold][pred_version]
                    self.PREDICTIONS[fold][pred_version][self.PREDICTIONS[fold].instance.isin(['2', '3'])] = \
                        PREDICTIONS_23[fold][pred_version]
            
        elif Ensemble_outerfolds.transpose().nunique().max() > 1:  # 2-Compute on all folds at once
            self._build_single_ensemble(self.PREDICTIONS, Performances, parameters, version, list_ensemble_levels,
                                        ensemble_level)
            for fold in self.folds:
                for ensemble_type in self.ensemble_types:
                    self.PREDICTIONS[fold]['outer_fold_' + version.replace('*', ensemble_type)] = np.nan
            
        else:  # 3-Compute fold by fold
            PREDICTIONS_ENSEMBLE = {}
            for outer_fold in self.outer_folds:
                # take the subset of the rows that correspond to the outer_fold
                col_outer_fold = ensemble_outerfolds_cols[0]
                PREDICTIONS_outerfold = {}
                for fold in self.folds:
                    self.PREDICTIONS[fold]['outer_fold_' + version] = self.PREDICTIONS[fold][col_outer_fold]
                    PREDICTIONS_outerfold[fold] = self.PREDICTIONS[fold][
                        self.PREDICTIONS[fold]['outer_fold_' + version] == float(outer_fold)]
                
                # build the ensemble model
                self._build_single_ensemble(PREDICTIONS_outerfold, Performances, parameters,
                                            version, list_ensemble_levels, ensemble_level)
                
                # merge the predictions on each outer_fold
                for fold in self.folds:
                    PREDICTIONS_outerfold[fold]['outer_fold_' + version] = float(outer_fold)
                    PREDICTIONS_outerfold[fold]['outer_fold_' + version.replace('*', ',')] = float(outer_fold)
                    PREDICTIONS_outerfold[fold]['outer_fold_' + version.replace('*', '?')] = float(outer_fold)
                    
                    # Save all the ensemble models if available
                    if ensemble_level is None:
                        df_outer_fold = PREDICTIONS_outerfold[fold][['id', 'outer_fold_' + version, 'pred_' + version]]
                    else:
                        df_outer_fold = PREDICTIONS_outerfold[fold][
                            ['id', 'outer_fold_' + version, 'pred_' + version,
                             'outer_fold_' + version.replace('*', ','), 'pred_' + version.replace('*', ','),
                             'outer_fold_' + version.replace('*', '?'), 'pred_' + version.replace('*', '?')]]
                    
                    # Initiate, or append if some previous outerfolds have already been concatenated
                    if fold not in PREDICTIONS_ENSEMBLE.keys():
                        PREDICTIONS_ENSEMBLE[fold] = df_outer_fold
                    else:
                        PREDICTIONS_ENSEMBLE[fold] = PREDICTIONS_ENSEMBLE[fold].append(df_outer_fold)
            
            # Add the ensemble predictions to the dataframe
            for fold in self.folds:
                if fold == 'train':
                    self.PREDICTIONS[fold] = self.PREDICTIONS[fold].merge(PREDICTIONS_ENSEMBLE[fold], how='outer',
                                                                          on=['id', 'outer_fold_' + version])
                else:
                    PREDICTIONS_ENSEMBLE[fold].drop('outer_fold_' + version, axis=1, inplace=True)
                    self.PREDICTIONS[fold] = self.PREDICTIONS[fold].merge(PREDICTIONS_ENSEMBLE[fold], how='outer',
                                                                          on=['id'])
        
        # build and save a dataset for this specific ensemble model
        for ensemble_type in ['*', ',', '?']:
            version_type = version.replace('*', ensemble_type)
            if 'pred_' + version_type in self.PREDICTIONS['test'].columns.values:
                for fold in self.folds:
                    df_single_ensemble = self.PREDICTIONS[fold][
                        ['id', 'outer_fold_' + version, 'pred_' + version_type]]
                    df_single_ensemble.rename(
                        columns={'outer_fold_' + version: 'outer_fold', 'pred_' + version_type: 'pred'}, inplace=True)
                    df_single_ensemble.dropna(inplace=True, subset=['pred'])
                    df_single_ensemble.to_csv(self.path_store + 'Predictions_' + self.pred_type + '_' + version_type +
                                              '_' + fold + '.csv', index=False)
    
    def _recursive_ensemble_builder(self, Performances_grandparent, parameters_parent, version_parent,
                                    list_ensemble_levels_parent):
        # Compute the ensemble models for the children first, so that they can be used for the parent model
        Performances_parent = Performances_grandparent[
            Performances_grandparent['version'].isin(
                fnmatch.filter(Performances_grandparent['version'], version_parent))]
        # if the last ensemble level has not been reached, go down one level and create a branch for each child.
        # Otherwise the leaf has been reached
        if len(list_ensemble_levels_parent) > 0:
            list_ensemble_levels_child = list_ensemble_levels_parent.copy()
            ensemble_level = list_ensemble_levels_child.pop()
            list_children = Performances_parent[ensemble_level].unique()
            for child in list_children:
                parameters_child = parameters_parent.copy()
                parameters_child[ensemble_level] = child
                version_child = self._parameters_to_version(parameters_child)
                # recursive call to the function
                self._recursive_ensemble_builder(Performances_parent, parameters_child, version_child,
                                                 list_ensemble_levels_child)
        else:
            ensemble_level = None
        
        # compute the ensemble model for the parent
        print('Building the ensemble model ' + version_parent)
        self._build_single_ensemble_wrapper(Performances_parent, parameters_parent, version_parent,
                                            list_ensemble_levels_parent, ensemble_level)
    
    def generate_ensemble_predictions(self):
        self._recursive_ensemble_builder(self.Performances, self.parameters, self.version, self.list_ensemble_levels)
        # Print the squared correlations between the target and the predictions
        for fold in self.folds:
            cols = [col for col in self.PREDICTIONS[fold].columns.values if 'pred' in col or col == self.target]
            corrs = (self.PREDICTIONS[fold][cols].corr()[self.target] ** 2).sort_values(ascending=False)
            print('Squared correlations between the target and the predictions for ' + fold + ': ')
            print(corrs)
    
    def save_predictions(self):
        for fold in self.folds:
            self.PREDICTIONS[fold].to_csv(self.path_store + 'PREDICTIONS_withEnsembles_' + self.pred_type + '_' +
                                          self.target + '_' + fold + '.csv', index=False)


class ResidualsGenerate(Hyperparameters):
    
    def __init__(self, target=None, fold=None, pred_type=None, debug_mode=False):
        # Parameters
        Hyperparameters.__init__(self)
        self.target = target
        self.fold = fold
        self.pred_type = pred_type
        self.debug_mode = debug_mode
        self.Residuals = pd.read_csv(self.path_store + 'PREDICTIONS_withEnsembles_' + self.pred_type + '_' +
                                     self.target + '_' + self.fold + '.csv')
        self.list_models = [col_name.replace('pred_', '') for col_name in self.Residuals.columns.values
                            if 'pred_' in col_name]
    
    def generate_residuals(self):
        list_models = [col_name.replace('pred_', '') for col_name in self.Residuals.columns.values
                       if 'pred_' in col_name]
        for model in list_models:
            print('Generating residuals for model ' + model)
            df_model = self.Residuals[['Age', 'pred_' + model]]
            no_na_indices = [not b for b in df_model['pred_' + model].isna()]
            df_model = df_model.dropna()
            if(len(df_model.index)) > 0:
                age = df_model.loc[:, ['Age']]
                res = df_model['Age'] - df_model['pred_' + model]
                regr = linear_model.LinearRegression()
                regr.fit(age, res)
                res_correction = regr.predict(age)
                res_corrected = res - res_correction
                self.Residuals.loc[no_na_indices, 'pred_' + model] = res_corrected
            # debug plot
            if self.debug_mode:
                print('Bias for the residuals ' + model, regr.coef_)
                plt.scatter(age, res)
                plt.scatter(age, res_corrected)
                regr2 = linear_model.LinearRegression()
                regr2.fit(age, res_corrected)
                print('Coefficients after: \n', regr2.coef_)
        self.Residuals.rename(columns=lambda x: x.replace('pred_', 'res_'), inplace=True)
    
    def save_residuals(self):
        self.Residuals.to_csv(self.path_store + 'RESIDUALS_' + self.pred_type + '_' + self.target + '_' + self.fold +
                              '.csv', index=False)


class ResidualsCorrelations(Hyperparameters):
    
    def __init__(self, target=None, fold=None, pred_type=None, debug_mode=False):
        Hyperparameters.__init__(self)
        self.target = target
        self.fold = fold
        self.pred_type = pred_type
        self.debug_mode = debug_mode
        if debug_mode:
            self.n_bootstrap_iterations_correlations = 10
        else:
            self.n_bootstrap_iterations_correlations = 1000
        self.Residuals = None
        self.CORRELATIONS = {}
    
    def preprocessing(self):
        # load data
        Residuals = pd.read_csv(self.path_store + 'RESIDUALS_' + self.pred_type + '_' + self.target + '_' + self.fold +
                                '.csv')
        # Format the dataframe
        Residuals_only = Residuals[[col_name for col_name in Residuals.columns.values if 'res_' in col_name]]
        Residuals_only.rename(columns=lambda x: x.replace('res_' + self.target + '_', ''), inplace=True)
        # Reorder the columns to make the correlation matrix more readable
        # Need to temporarily rename '?' because its ranking differs from the '*' and ',' characters
        Residuals_only.columns = [col_name.replace('?', ',placeholder') for col_name in Residuals_only.columns.values]
        Residuals_only = Residuals_only.reindex(sorted(Residuals_only.columns), axis=1)
        Residuals_only.columns = [col_name.replace(',placeholder', '?') for col_name in Residuals_only.columns.values]
        self.Residuals = Residuals_only
    
    def _bootstrap_correlations(self):
        names = self.Residuals.columns.values
        results = []
        for i in range(self.n_bootstrap_iterations_correlations):
            if (i + 1) % 100 == 0:
                print('Bootstrap iteration ' + str(i + 1) + ' out of ' + str(self.n_bootstrap_iterations_correlations))
            data_i = resample(self.Residuals, replace=True, n_samples=len(self.Residuals.index))
            results.append(np.array(data_i.corr()))
        results = np.array(results)
        RESULTS = {}
        for op in ['mean', 'std']:
            results_op = pd.DataFrame(getattr(np, op)(results, axis=0))
            results_op.index = names
            results_op.columns = names
            RESULTS[op] = results_op
        self.CORRELATIONS['_sd'] = RESULTS['std']
    
    def generate_correlations(self):
        # Generate the correlation matrix
        self.CORRELATIONS[''] = self.Residuals.corr()
        # Gerate the std by bootstrapping
        self._bootstrap_correlations()
        # Merge both as a dataframe of strings
        self.CORRELATIONS['_str'] = self.CORRELATIONS[''].round(3).applymap(str) \
                                    + '+-' + self.CORRELATIONS['_sd'].round(3).applymap(str)
    
    def save_correlations(self):
        for mode in self.modes:
            self.CORRELATIONS[mode].to_csv(self.path_store + 'ResidualsCorrelations' + mode + '_' + self.pred_type +
                                           '_' + self.target + '_' + self.fold + '.csv', index=True)


class SelectBest(Metrics):
    
    def __init__(self, target=None, pred_type=None):
        Metrics.__init__(self)
        
        self.target = target
        self.pred_type = pred_type
        self.organs = None
        self.best_models = None
        self.PREDICTIONS = {}
        self.RESIDUALS = {}
        self.PERFORMANCES = {}
        self.CORRELATIONS = {}
    
    def _load_data(self):
        for fold in self.folds:
            path_pred = self.path_store + 'PREDICTIONS_withEnsembles_' + self.pred_type + '_' + self.target \
                        + '_' + fold + '.csv'
            path_res = self.path_store + 'RESIDUALS_' + self.pred_type + '_' + self.target + '_' + fold + '.csv'
            path_perf = self.path_store + 'PERFORMANCES_withEnsembles_ranked_' + self.pred_type + '_' + self.target + \
                        '_' + fold + '.csv'
            path_corr = self.path_store + 'ResidualsCorrelations_str_' + self.pred_type + '_' + self.target + '_' + \
                        fold + '.csv'
            self.PREDICTIONS[fold] = pd.read_csv(path_pred)
            self.RESIDUALS[fold] = pd.read_csv(path_res)
            self.PERFORMANCES[fold] = pd.read_csv(path_perf)
            self.PERFORMANCES[fold].set_index('version', drop=False, inplace=True)
            self.CORRELATIONS[fold] = {}
            for mode in self.modes:
                self.CORRELATIONS[fold][mode] = pd.read_csv(path_corr.replace('_str', mode), index_col=0)
    
    def _select_versions(self):
        Performances = self.PERFORMANCES['val']
        idx_Ensembles = Performances['organ'].isin(['*', '?', ',']).values
        idx_withoutEnsembles = [not b for b in idx_Ensembles]
        Perf_Ensembles = Performances[idx_Ensembles]
        Perf_withoutEnsembles = Performances[idx_withoutEnsembles]
        self.organs = ['*']
        self.best_models = [Perf_Ensembles['version'].values[0]]
        for organ in Perf_withoutEnsembles['organ'].unique():
            Perf_organ = Perf_withoutEnsembles[Perf_withoutEnsembles['organ'] == organ]
            self.organs.append(organ)
            self.best_models.append(Perf_organ['version'].values[0])
    
    def _take_subsets(self):
        base_cols = self.id_vars + self.demographic_vars
        best_models_pred = ['pred_' + model for model in self.best_models]
        best_models_outer_fold = ['outer_fold_' + model for model in self.best_models]
        best_models_res = ['res_' + model for model in self.best_models]
        best_models_corr = ['_'.join(model.split('_')[1:]) for model in self.best_models]
        outer_fold_colnames = ['outer_fold_' + col for col in self.organs]
        for fold in self.folds:
            self.PREDICTIONS[fold] = self.PREDICTIONS[fold].loc[:, base_cols + best_models_pred +
                                                                   best_models_outer_fold]
            self.PREDICTIONS[fold].columns = base_cols + self.organs + outer_fold_colnames
            self.RESIDUALS[fold] = self.RESIDUALS[fold].loc[:, base_cols + best_models_res + best_models_outer_fold]
            self.RESIDUALS[fold].columns = base_cols + self.organs + outer_fold_colnames
            self.PERFORMANCES[fold] = self.PERFORMANCES[fold].loc[self.best_models, :]
            self.PERFORMANCES[fold].index = self.organs
            for mode in self.modes:
                self.CORRELATIONS[fold][mode] = self.CORRELATIONS[fold][mode].loc[best_models_corr, best_models_corr]
                self.CORRELATIONS[fold][mode].index = self.organs
                self.CORRELATIONS[fold][mode].columns = self.organs
    
    def select_models(self):
        self._load_data()
        self._select_versions()
        self._take_subsets()
    
    def save_data(self):
        for fold in self.folds:
            path_pred = self.path_store + 'PREDICTIONS_bestmodels_' + self.pred_type + '_' + self.target + '_' + fold \
                        + '.csv'
            path_res = self.path_store + 'RESIDUALS_bestmodels_' + self.pred_type + '_' + self.target + '_' + fold + \
                       '.csv'
            path_corr = self.path_store + 'ResidualsCorrelations_bestmodels_str_' + self.pred_type + '_' + self.target + \
                        '_' + fold + '.csv'
            path_perf = self.path_store + 'PERFORMANCES_bestmodels_ranked_' + self.pred_type + '_' + self.target + '_' \
                        + fold + '.csv'
            self.PREDICTIONS[fold].to_csv(path_pred, index=False)
            self.RESIDUALS[fold].to_csv(path_res, index=False)
            self.PERFORMANCES[fold].to_csv(path_perf, index=False)
            Performances_alphabetical = self.PERFORMANCES[fold].sort_values(by='version')
            Performances_alphabetical.to_csv(path_perf.replace('ranked', 'alphabetical'), index=False)
            for mode in self.modes:
                self.CORRELATIONS[fold][mode].to_csv(path_corr.replace('_str', mode), index=True)


class SelectCorrelationsNAs(Hyperparameters):
    
    def __init__(self, target=None):
        Hyperparameters.__init__(self)
        self.target = target
        self.CORRELATIONS = {'*': {'': {}, '_sd': {}, '_str': {}}}
    
    def load_data(self):
        for models_type in ['', '_bestmodels']:
            self.CORRELATIONS[models_type] = {}
            for pred_type in ['instances', 'eids', '*']:
                self.CORRELATIONS[models_type][pred_type] = {}
                for mode in self.modes:
                    self.CORRELATIONS[models_type][pred_type][mode] = {}
                    for fold in self.folds:
                        if pred_type == '*':
                            self.CORRELATIONS[models_type][pred_type][mode][fold] = \
                                pd.read_csv(self.path_store + 'ResidualsCorrelations' + models_type + mode +
                                            '_instances_' + self.target + '_' + fold + '.csv', index_col=0)
                        else:
                            self.CORRELATIONS[models_type][pred_type][mode][fold] = \
                                pd.read_csv(self.path_store + 'ResidualsCorrelations' + models_type + mode + '_' +
                                            pred_type + '_' + self.target + '_' + fold + '.csv', index_col=0)
    
    def fill_na(self):
        # Dectect NAs in the instances correlation matrix
        for models_type in ['', '_bestmodels']:
            NAs_mask = self.CORRELATIONS[models_type]['instances']['']['val'].isna()
            for mode in self.modes:
                for fold in self.folds:
                    self.CORRELATIONS[models_type]['*'][mode][fold] = \
                        self.CORRELATIONS[models_type]['instances'][mode][fold].copy()
                    self.CORRELATIONS[models_type]['*'][mode][fold][NAs_mask] = \
                        self.CORRELATIONS[models_type]['eids'][mode][fold][NAs_mask]
    
    def save_correlations(self):
        for models_type in ['', '_bestmodels']:
            for mode in self.modes:
                for fold in self.folds:
                    self.CORRELATIONS[models_type]['*'][mode][fold].to_csv(self.path_store + 'ResidualsCorrelations' +
                                                                           models_type + mode + '_*_' + self.target +
                                                                           '_' + fold + '.csv', index=True)


class PlotsCorrelations(Hyperparameters):
    
    def __init__(self, target=None, fold=None, pred_type=None, save_figures=True):
        Hyperparameters.__init__(self)
        self.target = target
        self.fold = fold
        self.pred_type = pred_type
        self.save_figures = save_figures
        self.fig_xsize = 23.4
        self.fig_ysize = 16.54
        self.Correlations = None
        self.Correlations_bestmodels = None
    
    def preprocessing(self):
        Correlations = pd.read_csv(self.path_store + 'ResidualsCorrelations_' + self.pred_type + '_' + self.target +
                                   '_' + self.fold + '.csv', index_col='Unnamed: 0')
        # Crop the names to make the reading of the labels easier
        idx_to_print = [self.names_model_parameters[1:].index(i) for i in ['organ', 'view', 'architecture']]
        Correlations.index = ['_'.join(np.array(idx.split('_'))[idx_to_print]) for idx in Correlations.index.values]
        Correlations.columns = ['_'.join(np.array(idx.split('_'))[idx_to_print]) for idx in Correlations.columns.values]
        self.Correlations = Correlations
        self.Correlations_bestmodels = pd.read_csv(self.path_store + 'ResidualsCorrelations_bestmodels_' +
                                                   self.pred_type + '_' + self.target + '_' + self.fold + '.csv',
                                                   index_col='Unnamed: 0')
    
    def _plot_correlations(self, data, title_save):
        
        # insert nan on diagonal
        data.values[tuple([np.arange(data.shape[0])]) * 2] = np.nan
        
        # set parameters
        plt.clf()
        sns.set(font_scale=1, rc={'figure.figsize': (self.fig_xsize, self.fig_ysize)})
        
        # plot
        annot = (data * 100).round().astype(str).applymap(lambda x: ''.join(x.split('.')[:1]))
        cor_plot = sns.heatmap(data=data, xticklabels=1, yticklabels=1, annot=annot, fmt='s',
                               annot_kws={"size": 10}, vmin=0, vmax=1, center=0, square=True)
        # optional: inclined x labels
        # cor_plot.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right');
        
        # Save figure
        if self.save_figures:
            fig = cor_plot.get_figure()
            fig.set_size_inches(self.fig_xsize, self.fig_ysize)
            fig.savefig('../figures/Correlations/' + title_save + '.png', dpi='figure')
    
    def generate_plots(self):
        title = 'Correlations_AllModels_' + self.pred_type + '_' + self.target + '_' + self.fold
        print(title)
        self._plot_correlations(data=self.Correlations, title_save=title)
        title_bestmodels = title.replace('AllModels', 'BestModels')
        self._plot_correlations(data=self.Correlations_bestmodels, title_save=title_bestmodels)
        
        # Plot the "ensemble models only" correlation plots
        for ensemble_type in self.ensemble_types:
            index_ensembles_only = [idx for idx in self.Correlations.columns.values if ensemble_type in idx]
            Correlations_Ensembles_only = self.Correlations.loc[index_ensembles_only, index_ensembles_only]
            title_e = title.replace('AllModels', 'Ensembles' + ensemble_type + 'Only')
            self._plot_correlations(data=Correlations_Ensembles_only, title_save=title_e)


class PlotsLoggers(Hyperparameters):
    
    def __init__(self, target=None, display_learning_rate=None):
        Hyperparameters.__init__(self)
        self.target = target
        self.display_learning_rate = display_learning_rate
        Predictions = pd.read_csv(self.path_store + 'PREDICTIONS_withoutEnsembles_instances_' + self.target +
                                  '_val.csv')
        self.list_versions = [col_name.replace('pred_', '') for col_name in Predictions.columns.values
                              if 'pred_' in col_name]
    
    def _plot_logger(self, version):
        try:
            logger = pd.read_csv(self.path_store + 'logger_' + version + '.csv')
        except FileNotFoundError:
            print('ERROR: THE FILE logger_' + version + '.csv'
                  + ' WAS NOT FOUND OR WAS EMPTY/CORRUPTED. SKIPPING PLOTTING OF THE TRAINING FOR THIS MODEL.')
            return
        # Amend column names for consistency
        logger.columns = [name[:-2] if name.endswith('_K') else name for name in logger.columns]
        metrics_names = [metric[4:] for metric in logger.columns.values if metric.startswith('val_')]
        logger.columns = ['train_' + name if name in metrics_names else name for name in logger.columns]
        # rewrite epochs numbers based on nrows. several loggers might have been appended if model has been retrained.
        logger['epoch'] = [i + 1 for i in range(len(logger.index))]
        # multiplot layout
        n_rows = 3
        n_metrics = len(metrics_names)
        fig, axs = plt.subplots(math.ceil(n_metrics / n_rows), min(n_metrics, n_rows), sharey=False, sharex=True,
                                squeeze=False)
        fig.set_figwidth(5 * n_metrics)
        fig.set_figheight(5)
        
        # plot evolution of each metric during training, train and val values
        for m, metric in enumerate(metrics_names):
            i = int(m / n_rows)
            j = m % n_rows
            for fold in ['train', 'val']:
                axs[i, j].plot(logger['epoch'], logger[fold + '_' + metric])
            axs[i, j].legend(['Training', 'Validation'], loc='upper left')
            axs[i, j].set_title(metric + ' = f(Epoch)')
            axs[i, j].set_xlabel('Epoch')
            axs[i, j].set_ylabel(metric)
            if metric not in ['true_positives', 'false_positives', 'false_negatives', 'true_negatives']:
                axs[i, j].set_ylim((-0.2, 1.1))
            # use second axis for learning rate
            if self.display_learning_rate & ('lr' in logger.columns):
                ax2 = axs[i, j].twinx()
                ax2.plot(logger['epoch'], logger['lr'], color='green')
                ax2.tick_params(axis='y', labelcolor='green')
                ax2.legend(['Learning Rate'], loc='upper right')
        fig.tight_layout()
        # save figure as pdf before closing
        fig.savefig("../figures/Loggers/Logger_" + version + '.pdf', bbox_inches='tight')
        plt.close('all')
    
    def generate_plots(self):
        for version in self.list_versions:
            for outer_fold in self.outer_folds:
                self._plot_logger(version=version + '_' + outer_fold)


class PlotsScatter(Hyperparameters):
    
    def __init__(self, target=None, pred_type=None):
        Hyperparameters.__init__(self)
        self.target = target
        self.pred_type = pred_type
        # Load the predictions
        self.PREDICTIONS = {}
        for fold in self.folds:
            self.PREDICTIONS[fold] = pd.read_csv(self.path_store + 'PREDICTIONS_withEnsembles_' + self.pred_type + '_'
                                                 + self.target + '_' + fold + '.csv')
        # print scatter plot for each model
        self.list_versions = [col_name.replace('pred_', '') for col_name in self.PREDICTIONS['test'].columns.values
                              if 'pred_' in col_name]
        # define dictionaries to format the text
        self.dict_folds_names = {'train': 'Training', 'val': 'Validation', 'test': 'Testing'}
    
    def generate_plots(self):
        for version in self.list_versions[:1]:
            # concatenate the predictions, format the data before plotting
            df_allsets = None
            for fold in self.folds:
                df_version = self.PREDICTIONS[fold][[self.target, 'pred_' + version, 'outer_fold_' + version]]
                df_version.dropna(inplace=True)
                df_version.rename(columns={'pred_' + version: 'Prediction', 'outer_fold_' + version: 'outer_fold'},
                                  inplace=True)
                df_version['outer_fold'] = df_version['outer_fold'].astype(int).astype(str)
                df_version['set'] = self.dict_folds_names[fold]
                
                # Generate single plot and save it
                single_plot = sns.lmplot(x=self.target, y='Prediction', data=df_version, fit_reg=False,
                                         hue='outer_fold', scatter_kws={'alpha': 0.3})
                single_plot.savefig('../figures/ScatterPlot_' + version + '_' + fold + '.png')
                
                # concatenate data for the multiplot
                if fold == 'train':
                    df_allsets = df_version
                else:
                    df_allsets = df_allsets.append(df_version)
            
            # generate the multi plot and save it
            multi_plot = sns.FacetGrid(df_allsets, col='set', hue='outer_fold')
            multi_plot.map(plt.scatter, 'Age', 'Prediction', alpha=.1)
            multi_plot.add_legend()
            multi_plot.savefig('../figures/Scatter_Plots/ScatterPlots_' + self.pred_type + '_' + version + '.png')


class PlotsAttentionMaps(DeepLearning):
    
    def __init__(self, target=None, organ=None, view=None, transformation=None, fold=None, debug_mode=False):
        # Partial initialization with placeholders to get access to parameters and functions
        DeepLearning.__init__(self, target, organ, view, transformation, 'VGG16', 'Adam', 0, 0, 0, False)
        
        # Parameters
        self.fold = fold
        self.parameters = None
        self.image_width = None
        self.image_height = None
        self.batch_size = None
        self.N_samples_attentionmaps = 10  # needs to be > 1 for the script to work
        if debug_mode:
            self.N_samples_attentionmaps = 2
        
        # Pick the best model based on the performances
        path_perf = self.path_store + 'PERFORMANCES_withoutEnsembles_ranked_instances_' + self.target + '_' + \
                    self.fold + '.csv'
        Performances = pd.read_csv(path_perf).set_index('version', drop=False)
        Performances = Performances[(Performances['organ'] == organ)
                                    & (Performances['view'] == self.view)
                                    & (Performances['transformation'] == self.transformation)]
        version = Performances['version'].values[0]
        del Performances
        
        # other parameters
        self.parameters = self._version_to_parameters(version)
        DeepLearning.__init__(self, target, organ, view, transformation, self.parameters['architecture'],
                              self.parameters['optimizer'], self.parameters['learning_rate'],
                              self.parameters['weight_decay'], self.parameters['dropout_rate'], False)
        self.dir_images = '../images/' + self.organ + '/' + self.view + '/' + self.transformation + '/'
        self.prediction_type = self.dict_prediction_types[self.target]
        self.Residuals = None
        self.df_to_plot = None
        self.df_outer_fold = None
        self.penultimate_layer_idx = None
        self.images = None
        self.VISUALIZATION_FILTERS = {}
        self.plot_title = None
        self.class_mode = None
        self.image = None
        self.saliency_analyzer = None
        self.guided_backprop_analyzer = None
        self.generator = None
        self.dict_map_types_to_names = {'saliency': 'Saliency', 'grad_cam': 'Gradcam',
                                        'guided_backprop': 'GuidedBackprop'}
        self.dict_architecture_to_last_conv_layer_name = {'VGG16': 'block5_conv3', 'VGG19': 'block5_conv4',
                                                          'MobileNet': 'conv_pw_13_relu', 'MobileNetV2': 'out_relu',
                                                          'DenseNet121': 'relu', 'DenseNet169': 'relu',
                                                          'DenseNet201': 'relu', 'NASNetMobile': 'activation_1136',
                                                          'NASNetLarge': 'activation_1396',
                                                          'Xception': 'block14_sepconv2_act', 'InceptionV3': 'mixed10',
                                                          'InceptionResNetV2': 'conv_7b_ac',
                                                          'EfficientNetB7': 'top_activation'}
    
    def _format_residuals(self):
        # Format the residuals
        Residuals_full = pd.read_csv(self.path_store + 'RESIDUALS_instances_' + self.target + '_' + self.fold + '.csv')
        Residuals = Residuals_full[['id'] + self.demographic_vars +
                                   ['res_' + self.version, 'outer_fold_' + self.version]]
        del Residuals_full
        Residuals.dropna(inplace=True)
        Residuals.rename(columns={'res_' + self.version: 'res', 'outer_fold_' + self.version: 'outer_fold'},
                         inplace=True)
        Residuals.set_index('id', drop=False, inplace=True)
        #Residuals['id'] = Residuals['id'].astype(str).apply(self._append_ext) TODO
        Residuals['outer_fold'] = Residuals['outer_fold'].astype(int).astype(str)
        Residuals['res_abs'] = Residuals['res'].abs()
        self.Residuals = Residuals  # [['id', 'outer_fold', 'Sex', 'Age', 'res', 'res_abs']] TODO
    
    def _select_representative_samples(self):
        # Select with samples to plot
        print('Selecting representative samples...')
        Sexes = ['Male', 'Female']
        dict_sexes_to_values = {'Male': 0, 'Female': 1}
        df_to_plot = None
        for sex in Sexes:
            print('Sex: ' + sex)
            Residuals_sex = self.Residuals[self.Residuals['Sex'] == dict_sexes_to_values[sex]]
            Residuals_sex['sex'] = sex
            for age_category in ['young', 'middle', 'old']:
                print('Age category: ' + age_category)
                if age_category == 'young':
                    Residuals_age = Residuals_sex[Residuals_sex['Age'] <= Residuals_sex['Age'].min() + 1]
                elif age_category == 'middle':
                    Residuals_age = Residuals_sex[Residuals_sex['Age'] == int(Residuals_sex['Age'].median())]
                else:
                    Residuals_age = Residuals_sex[Residuals_sex['Age'] >= Residuals_sex['Age'].max() - 1]
                Residuals_age['age_category'] = age_category
                if len(Residuals_age.index) < 3 * self.N_samples_attentionmaps:
                    print(f"Warning! Less than {3 * self.N_samples_attentionmaps} samples ({len(Residuals_age.index)})"
                          f" for sex = {sex} and age category = {age_category}")
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
        pred_age = (df_to_plot['Age'] - df_to_plot['res']).round().astype(str)
        df_to_plot['plot_title'] = 'Age = ' + df_to_plot['Age'].astype(str) + ', Predicted Age = ' + pred_age + \
                                   ', Sex = ' + df_to_plot['sex'] + ', sample ' + df_to_plot['sample'].astype(str)
        df_to_plot['save_title'] = self.target + '_' + self.organ + '_' + self.view + '_' + self.transformation + '_' \
                                   + df_to_plot['sex'] + '_' + df_to_plot['age_category'] + '_' \
                                   + df_to_plot['aging_rate'] + '_' + df_to_plot['sample'].astype(str)
        path_save = self.path_store + 'AttentionMaps-samples_' + self.target + '_' + self.organ + '_' + self.view + \
                    '_' + self.transformation + '.csv'
        df_to_plot.to_csv(path_save, index=False)
        self.df_to_plot = df_to_plot
    
    def preprocessing(self):
        self._generate_architecture()
        self.penultimate_layer_idx = utils.find_layer_idx(
            self.model, self.dict_architecture_to_last_conv_layer_name[self.parameters['architecture']])
        self._format_residuals()
        self._select_representative_samples()
    
    def _preprocess_for_outer_fold(self, outer_fold):
        self.df_outer_fold = self.df_to_plot[self.df_to_plot['outer_fold'] == outer_fold]
        
        # generate the data generators
        self.generator = MyImageDataGenerator(target=self.target, organ=self.organ, data_features=self.df_outer_fold,
                                              n_samples_per_subepoch=self.n_samples_per_subepoch,
                                              batch_size=self.batch_size, training_mode=False, seed=self.seed,
                                              side_predictors=self.side_predictors, dir_images=self.dir_images,
                                              images_width=self.image_width, images_height=self.image_height,
                                              data_augmentation=False)
        
        # load the weights for the fold
        self.model.load_weights(self.path_store + 'model-weights_' + self.version + '_' + outer_fold + '.h5')
        
        # Generate analyzers
        self.saliency_analyzer = innvestigate.create_analyzer("gradient", self.model, allow_lambda_layers=True)
        self.guided_backprop_analyzer = innvestigate.create_analyzer("guided_backprop", self.model,
                                                                     allow_lambda_layers=True)
        
        # Generate the saliency maps
        self.n_images = len(self.df_outer_fold.index)
    
    # generate the saliency map transparent filter
    def _generate_saliency_map(self, saliency):
        saliency = saliency.sum(axis=2)
        saliency *= 255 / np.max(np.abs(saliency))
        saliency = saliency.astype(int)
        r_ch = saliency.copy()
        r_ch[r_ch < 0] = 0
        b_ch = -saliency.copy()
        b_ch[b_ch < 0] = 0
        g_ch = saliency.copy() * 0
        a_ch = np.maximum(b_ch, r_ch) * 5
        self.saliency_filter = np.dstack((r_ch, g_ch, b_ch, a_ch))
    
    # generate the gradcam map transparent filter
    def _generate_gradcam_map(self):
        grad_cam = visualize_cam(model=self.model, layer_idx=-1, filter_indices=0, seed_input=self.image,
                                 penultimate_layer_idx=self.penultimate_layer_idx)
        r_ch = grad_cam[:, :, 0]
        g_ch = grad_cam[:, :, 1]
        b_ch = grad_cam[:, :, 2]
        a_ch = ((255 - b_ch) * .5).astype(int)
        b_ch = b_ch
        self.grad_cam_filter = np.dstack((r_ch, g_ch, b_ch, a_ch))
    
    # generate the guidedbackprop map transparent filter
    def _generate_guidedbackprop_map(self, guided_backprop):
        guided_backprop = np.linalg.norm(guided_backprop, axis=2)
        guided_backprop = (guided_backprop * 255 / guided_backprop.max()).astype(int)
        r_ch = guided_backprop.copy()
        g_ch = guided_backprop.copy() * 0
        b_ch = guided_backprop.copy() * 0
        a_ch = guided_backprop * 15
        self.guided_backprop_filter = np.dstack((r_ch, g_ch, b_ch, a_ch))
    
    def _plot_attention_map(self, filter_map, save_title):
        plt.clf()
        plt.imshow(self.image)
        plt.imshow(filter_map)
        plt.axis('off')
        plt.title(self.plot_title)
        fig = plt.gcf()
        fig.savefig('../figures/Attention_Maps/' + save_title + '.png')
        plt.show()
    
    def _plot_attention_maps(self, save_title):
        # format the grid of plots
        plt.clf()
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        subtitles = {0: {0: 'Original Image', 1: 'Saliency'}, 1: {0: 'Grad-CAM', 1: 'Guided Backpropagation'}}
        for i in [0, 1]:
            for j in [0, 1]:
                axes[i, j].imshow(self.image)
                axes[i, j].axis('off')
                axes[i, j].set_title(subtitles[i][j], {'fontsize': 15})
        
        # fill the plot array
        axes[0, 1].imshow(self.saliency_filter)
        axes[1, 0].imshow(self.grad_cam_filter)
        axes[1, 1].imshow(self.guided_backprop_filter)
        
        plt.suptitle(self.plot_title, fontsize=20)
        fig = plt.gcf()
        fig.savefig('../figures/Attention_Maps/Summary_' + save_title + '.png')
        plt.show()
    
    def _generate_maps_for_one_batch(self, i):
        print('Generating maps for batch ' + str(i))
        n_images_batch = np.min([self.batch_size, self.n_images - i * self.batch_size])
        images = self.generator.__getitem__(i)[0][0][:n_images_batch, :, :, :]
        '''
        #saliencies = self.saliency_analyzer.analyze(images)
        #guided_backprops = self.guided_backprop_analyzer.analyze(images)
        for j in range(saliencies.shape[0]):
        '''
        for j in range(n_images_batch):
            # select sample
            self.image = images[j]
            self.plot_title = self.df_outer_fold['plot_title'].values[i * self.batch_size + j]
            save_title = self.df_outer_fold['save_title'].values[i * self.batch_size + j]
            
            # generate the transparent filters for saliency, grad-cam and guided-backprop maps
            #self._generate_saliency_map(saliencies[j])
            self._generate_gradcam_map()
            #self._generate_guidedbackprop_map(guided_backprops[j])

            self._plot_attention_map(filter_map=getattr(self, map_type + '_filter'), save_title=self.dict_map_types_to_names[map_type] + '_' + save_title)
            
            '''
            # plot the three maps individually
            for map_type in self.dict_map_types_to_names.keys():
                self._plot_attention_map(filter_map=getattr(self, map_type + '_filter'),
                                         save_title=self.dict_map_types_to_names[map_type] + '_' + save_title)
            
            # Generate summary plot
            self._plot_attention_maps(save_title=save_title)
            '''
    
    def generate_plots(self):
        for outer_fold in self.outer_folds:
            print('Generate attention maps for outer_fold ' + outer_fold)
            gc.collect()
            self._preprocess_for_outer_fold(outer_fold)
            for i in range(math.ceil(self.n_images / self.batch_size)):
                self._generate_maps_for_one_batch(i)
