from MI_Libraries import *


# CLASSES
class Hyperparameters:
    
    def __init__(self):
        # seeds for reproducibility
        self.seed = 0
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        set_random_seed(self.seed)
        
        # other parameters
        self.path_store = '../data/'
        self.folds = ['train', 'val', 'test']
        self.n_CV_outer_folds = 10
        self.outer_folds = [str(x) for x in list(range(self.n_CV_outer_folds))]
        self.ensemble_types = ['*', ',', '?']
        self.modes = ['', '_sd', '_str']
        self.list_field_ids_in_instance_2 = ['20204', '20208', '20205', '20227']
        self.names_model_parameters = ['target', 'organ', 'field_id', 'view', 'transformation', 'architecture',
                                       'optimizer', 'learning_rate', 'weight_decay', 'dropout_rate']
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
        if len(parameters_list) > 10:
            parameters['outer_fold'] = parameters_list[10]
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
        Hyperparameters.__init__(self)
        
        self.targets_regression = ['Age']
        self.targets_binary = ['Sex']
        self.dict_prediction_types = {'Age': 'regression', 'Sex': 'binary'}
        self.metrics_displayed_in_int = ['True-Positives', 'True-Negatives', 'False-Positives', 'False-Negatives']
        self.metrics_needing_classpred = ['F1-Score', 'Binary-Accuracy', 'Precision', 'Recall']
        self.images_field_ids = ['20204', '20208', '20227', '210156']
        self.dict_metrics_names = {'regression': ['RMSE', 'R-Squared'],
                                   'binary': ['ROC-AUC', 'F1-Score', 'PR-AUC', 'Binary-Accuracy', 'Sensitivity',
                                              'Specificity', 'Precision', 'Recall', 'True-Positives', 'False-Positives',
                                              'False-Negatives', 'True-Negatives'],
                                   'multiclass': ['Categorical-Accuracy']}
        self.dict_losses_names = {'regression': 'MSE', 'binary': 'Binary-Crossentropy',
                                  'multiclass': 'categorical_crossentropy'}
        self.dict_main_metrics_names = {'Age': 'R-Squared', 'Sex': 'ROC-AUC',
                                        'imbalanced_binary_placeholder': 'F1-Score'}
        self.main_metrics_modes = {'loss': 'min', 'R-Squared': 'max', 'ROC-AUC': 'max'}
        
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
    
    def generate_data(self):
        dict_UKB_fields_to_names = {'31-0.0': 'Sex', '21003-0.0': 'Age', '21003-2.0': 'Age_Imaging',
                                    '22414-2.0': 'Liver_images_quality'}
        # extract the relevant columns from the main UKB dataset
        data_features = pd.read_csv('/n/groups/patel/uk_biobank/main_data_52887/ukb37397.csv',
                                    usecols=['eid', '31-0.0', '21003-0.0', '21003-2.0', '22414-2.0'])
        data_features.rename(columns=dict_UKB_fields_to_names, inplace=True)
        data_features['eid'] = data_features['eid'].astype(str)
        data_features = data_features.set_index('eid', drop=False)
        data_features.to_csv(self.path_store + 'data-features.csv', index=False)


class PreprocessingFolds(Metrics):
    """
    Gather all the hyperparameters of the algorithm
    """
    def __init__(self, target, image_field):
        Metrics.__init__(self)
        
        self.target = target
        self.image_field = image_field
        self.organ = self.image_field.split('_')[0]
        self.field_id = self.image_field.split('_')[1]
        self.image_quality_ids = {'Liver': '22414-2.0'}
        self.image_quality_ids.update(
            dict.fromkeys(['Heart', 'Brain', 'DXA', 'Pancreas', 'Carotid', 'ECG', 'ArterialStiffness', 'EyeFundus'],
                          None))
        self.image_quality_id = self.image_quality_ids[self.organ]
        self.list_available_ids = None
        self.data = None
        self.IDS = None
        
        # dictionary of dir_images used to generate the IDs split during preprocessing
        self.dict_default_dir_images = {'Liver_20204': '../images/Liver/20204/main/raw/',
                                        'Heart_20208': '../images/Heart/20208/4chambers/raw/',
                                        'Brain_20227': '../images/Brain/20227/sagittal/raw/',
                                        'EyeFundus_210156': '../images/EyeFundus/210156/right/raw/',
                                        'PhysicalActivity_90001': '../images/PhysicalActivity/90001/main/raw/'}
        self.dict_field_id_to_age_instance = dict.fromkeys(['Placeholder', '6025', '4205', '210156'], 'Age')
        self.dict_field_id_to_age_instance.update(dict.fromkeys(['20204', '20208', '20205', '20227'],
                                                                'Age_Imaging'))
        self.dict_field_id_to_age_instance.update(dict.fromkeys(['90001'], 'Age_Accelerometer'))
    
    def _get_list_available_ids(self):
        # get the list of the ids available for the field_id
        if self.field_id in self.images_field_ids:
            list_images = os.listdir(self.dict_default_dir_images[self.image_field])
            self.list_available_ids = [e.replace('.jpg', '') for e in list_images]
        else:
            list_available_ids_raw = pd.read_csv(self.path_store + 'IDs_' + self.field_id + '.csv')
            self.list_available_ids = list_available_ids_raw.values.squeeze().astype(str)
    
    def _filter_and_format_data(self):
        """
        Clean the data before it can be split between the rows
        """
        cols_data = ['eid', 'Sex', self.dict_field_id_to_age_instance[self.field_id]]
        dict_rename_cols = {self.dict_field_id_to_age_instance[self.field_id]: 'Age'}
        
        if self.image_quality_id is not None:
            cols_data.append(self.organ + '_images_quality')
            dict_rename_cols[self.organ + '_images_quality'] = 'Data_quality'
        data = pd.read_csv(self.path_store + 'data-features.csv', usecols=cols_data)
        data.rename(columns=dict_rename_cols, inplace=True)
        data['eid'] = data['eid'].astype(str)  # .apply(append_ext)
        data = data.set_index('eid', drop=False)
        if self.image_quality_id is not None:
            data = data[data['Data_quality'] != np.nan]
            data = data.drop('Data_quality', axis=1)
        # get rid of samples with NAs
        data.dropna(inplace=True)
        # list the samples' ids for which images are available
        data = data.loc[self.list_available_ids]
        self.data = data
    
    def _split_ids(self):
        # distribute the ids between the different outer and inner folds
        ids = self.data.index.values.copy()
        n_samples = len(ids)
        n_samples_by_fold = n_samples / self.n_CV_outer_folds
        FOLDS_IDS = {}
        for outer_fold in self.outer_folds:
            FOLDS_IDS[outer_fold] = np.ndarray.tolist(
                ids[int((int(outer_fold)) * n_samples_by_fold):int((int(outer_fold) + 1) * n_samples_by_fold)])
        TRAINING_IDS = {}
        VALIDATION_IDS = {}
        TEST_IDS = {}
        for i in self.outer_folds:
            TRAINING_IDS[i] = []
            VALIDATION_IDS[i] = []
            TEST_IDS[i] = []
            for j in self.outer_folds:
                if j == i:
                    VALIDATION_IDS[i].extend(FOLDS_IDS[j])
                elif ((int(i) + 1) % self.n_CV_outer_folds) == int(j):
                    TEST_IDS[i].extend(FOLDS_IDS[j])
                else:
                    TRAINING_IDS[i].extend(FOLDS_IDS[j])
        self.IDS = {'train': TRAINING_IDS, 'val': VALIDATION_IDS, 'test': TEST_IDS}
    
    def _split_data(self):
        # split ids
        self._split_ids()
        # generate inner fold split for each outer fold
        for outer_fold in self.outer_folds:
            print('Splitting data for outer fold ' + outer_fold)
            # compute values for scaling of regression targets
            target_mean, target_std = np.nan, np.nan
            if self.target in self.targets_regression:
                data_train = self.data.loc[self.IDS['train'][outer_fold], :]
                target_mean = data_train[self.target].mean()
                target_std = data_train[self.target].std()
            # generate folds
            for fold in self.folds:
                data_fold = self.data.loc[self.IDS[fold][outer_fold], :]
                data_fold['outer_fold'] = outer_fold
                data_fold = data_fold[['eid', 'outer_fold', 'Sex', 'Age']]
                if self.target in self.targets_regression:
                    data_fold[self.target + '_raw'] = data_fold[self.target]
                    data_fold[self.target] = (data_fold[self.target] - target_mean) / target_std
                data_fold.to_csv(
                    self.path_store + 'data-features_' + self.image_field + '_' + self.target + '_' + fold + '_' +
                    outer_fold + '.csv', index=False)
    
    def generate_folds(self):
        self._get_list_available_ids()
        self._filter_and_format_data()
        self._split_ids()
        self._split_data()


class MyImageDataGenerator(ImageDataGenerator, Sequence):
    
    def __init__(self, target=None, field_id=None, data_features=None, batch_size=None, shuffle=None, dir_images=None,
                 images_width=None, images_height=None, data_augmentation=False, seed=None):
        # parameters
        self.target = target
        self.labels = data_features[self.target]
        self.field_id = field_id
        self.data_features = data_features
        self.list_ids = data_features.index.values
        self.batch_size = batch_size
        self.steps = math.ceil(len(self.list_ids)/self.batch_size)
        self.shuffle = shuffle
        self.indices = None
        self.on_epoch_end()  # initiate the indexes and shuffles the ids
        self.dir_images = dir_images
        self.images_width = images_width
        self.images_height = images_height
        # Data augmentation
        self.data_augmentation = data_augmentation
        self.seed = seed
        self.dict_rotation_ranges = {'20227': 0, '210156': 0, '20204': 20, '20208': 20}
        self.dict_width_shift_ranges = {'20227': 0, '210156': 0, '20204': 0.1, '20208': 0.1}
        self.dict_height_shift_ranges = {'20227': 0, '210156': 0, '20204': 0.1, '20208': 0.1}
        ImageDataGenerator.__init__(self, rotation_range=self.dict_rotation_ranges[self.field_id],
                                    width_shift_range=self.dict_width_shift_ranges[self.field_id],
                                    height_shift_range=self.dict_height_shift_ranges[self.field_id], rescale=1. / 255.)
    
    def __len__(self):
        return int(np.floor(len(self.list_ids) / self.batch_size))
    
    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        list_ids_batch = [self.list_ids[i] for i in indices]
        X, y = self._data_generation(list_ids_batch)
        return X, y
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _data_generation(self, list_ids_batch):
        # Initialization
        n_samples_batch = min(len(list_ids_batch), self.batch_size)
        X = np.empty((n_samples_batch, self.images_width, self.images_height, 3))
        y = np.empty(n_samples_batch)
        # Generate data
        for i, ID in enumerate(list_ids_batch):
            path_image = self.dir_images + ID  # TODO  + '.jpg'
            img = load_img(path_image, target_size=(self.images_width, self.images_height), color_mode='rgb')
            x = img_to_array(img)
            if hasattr(img, 'close'):
                img.close()
            if self.data_augmentation:
                params = self.get_random_transform(x.shape, seed=self.seed)
                x = self.apply_transform(x, params)
                x = self.standardize(x)
            X[i, ] = x
            y[i] = self.labels[ID]
        return X, y


class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', baseline=-np.Inf, verbose=0, save_best_only=False,
                 save_weights_only=False, mode='auto', period=1):
        ModelCheckpoint.__init__(self, filepath, monitor=monitor, verbose=verbose, save_best_only=save_best_only,
                                 save_weights_only=save_weights_only, mode=mode, period=period)
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
    def __init__(self, target=None, organ_id_view=None, transformation=None, architecture=None, optimizer=None,
                 learning_rate=None, weight_decay=None, dropout_rate=None, debug_mode=False):
        
        Metrics.__init__(self)
        
        # Model's version
        self.target = target
        self.organ_id_view = organ_id_view
        self.organ = self.organ_id_view.split('_')[0]
        self.field_id = self.organ_id_view.split('_')[1]
        self.view = self.organ_id_view.split('_')[2]
        self.transformation = transformation
        self.architecture = architecture
        self.optimizer = optimizer
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.dropout_rate = float(dropout_rate)
        self.outer_fold = None
        self.version = self.target + '_' + self.organ_id_view + '_' + self.transformation + '_' + self.architecture \
                       + '_' + self.optimizer + '_' + np.format_float_positional(self.learning_rate) + '_' \
                       + str(self.weight_decay) + '_' + str(self.dropout_rate)
        
        # NNet's architecture and weights
        self.dict_final_activations = {'regression': 'linear', 'binary': 'sigmoid', 'multiclass': 'softmax',
                                       'saliency': 'linear'}
        self.path_load_weights = None
        self.keras_weights = None
        
        # Generators
        self.debug_mode = debug_mode
        self.debug_fraction = 0.02
        self.DATA_FEATURES = {}
        self.mode = None
        self.n_cpus = len(os.sched_getaffinity(0))
        self.dir_images = '../images/' + self.organ + '/' + self.field_id + '/' + self.view + '/' \
                                + self.transformation + '/'
        # define dictionary to resize the images to the right size depending on the model
        self.input_size_models = dict.fromkeys(
            ['VGG16', 'VGG19', 'MobileNet', 'MobileNetV2', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'NASNetMobile',
             'ResNet50', 'ResNext50'], 224)
        self.input_size_models.update(dict.fromkeys(['Xception', 'InceptionV3', 'InceptionResNetV2'], 299))
        self.input_size_models.update(dict.fromkeys(['NASNetLarge'], 331))
        self.input_size_models.update(dict.fromkeys(['EfficientNetB7'], 300))
        # define dictionary to fit the architecture's input size to the images sizes (take min (height, width))
        self.dict_field_id_to_image_size = {'20204': 288, '20208': 200, '20227': 88, '210156': 300}
        # for future v2: self.image_size = self.dict_field_id_to_image_size[self.field_id]
        
        self.image_size = self.input_size_models[self.architecture]
        # define dictionary of batch sizes to fit as many samples as the model's architecture allows
        self.dict_batch_sizes = dict.fromkeys(['NASNetMobile'], 128)
        self.dict_batch_sizes.update(dict.fromkeys(['MobileNet', 'MobileNetV2', 'ResNet50', 'ResNext50'], 64))
        self.dict_batch_sizes.update(dict.fromkeys(['InceptionV3', 'VGG19', 'DenseNet121', 'DenseNet169'], 32))
        self.dict_batch_sizes.update(dict.fromkeys(['DenseNet201', 'VGG16', 'Xception'], 16))
        self.dict_batch_sizes.update(dict.fromkeys(['InceptionResNetV2'], 8))
        self.dict_batch_sizes.update(dict.fromkeys(['NASNetLarge', 'EfficientNetB7'], 4))
        self.dict_rotation_ranges = {'Brain': 0, 'EyeFundus': 0, 'Liver': 20, 'Heart': 20}
        self.dict_shift_ranges = {'Brain': 0, 'EyeFundus': 0, 'Liver': 0.2, 'Heart': 0.2}
        self.batch_size = self.dict_batch_sizes[self.architecture]
        # double the batch size for the teslaM40 cores that have bigger memory
        if len(GPUtil.getGPUs()) > 0:  # make sure GPUs are available (not truesometimes for debugging)
            if GPUtil.getGPUs()[0].memoryTotal > 20000:
                self.batch_size *= 2
        # dict to decide which field is used to generate the ids when several targets share the same ids
        # (e.g Age and Sex)
        self.dict_target_to_ids = dict.fromkeys(['Age', 'Sex'], 'Age')
        # dict to decide which field is used to generate the ids when several organs/fields share the same ids
        # (e.g Liver_20204 and Heart_20208)
        self.dict_image_field_to_ids = dict.fromkeys(['PhysicalActivity_90001'], 'PhysicalActivity_90001')
        self.dict_image_field_to_ids.update(dict.fromkeys(['Liver_20204'], 'Liver_20204'))
        self.dict_image_field_to_ids.update(dict.fromkeys(['Heart_20208'], 'Heart_20208'))
        self.dict_image_field_to_ids.update(dict.fromkeys(['Brain_20227'], 'Brain_20227'))
        self.dict_image_field_to_ids.update(dict.fromkeys(['EyeFundus_210156'], 'EyeFundus_210156'))
        
        # Metrics
        self.prediction_type = self.dict_prediction_types[self.target]
        
        # Model
        self.model = None
        
        # Configure gpu(s)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.gpu_session = tf.Session(config=config)
        k.set_session(session=self.gpu_session)
        k.tensorflow_backend._get_available_gpus()
        
        def r2_k(y_true, y_pred):
            SS_res = k.sum(k.square(y_true - y_pred))
            SS_tot = k.sum(k.square(y_true - k.mean(y_true)))
            return 1 - SS_res / (SS_tot + k.epsilon())
        
        def rmse_k(y_true, y_pred):
            return k.sqrt(k.mean(k.square(y_pred - y_true)))
        
        def sensitivity_k(y_true, y_pred):
            true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
            possible_positives = k.sum(k.round(k.clip(y_true, 0, 1)))
            return true_positives / (possible_positives + k.epsilon())
        
        def specificity_k(y_true, y_pred):
            true_negatives = k.sum(k.round(k.clip((1 - y_true) * (1 - y_pred), 0, 1)))
            possible_negatives = k.sum(k.round(k.clip(1 - y_true, 0, 1)))
            return true_negatives / (possible_negatives + k.epsilon())
        
        def roc_auc_k(y_true, y_pred):
            auc = tf.metrics.auc(y_true, y_pred, curve='ROC')[1]
            k.get_session().run(tf.local_variables_initializer())
            return auc
        
        def recall_k(y_true, y_pred):
            true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
            possible_positives = k.sum(k.round(k.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + k.epsilon())
            return recall
        
        def precision_k(y_true, y_pred):
            true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
            predicted_positives = k.sum(k.round(k.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + k.epsilon())
            return precision
        
        def pr_auc_k(y_true, y_pred):
            auc = tf.metrics.auc(y_true, y_pred, curve='PR', summation_method='careful_interpolation')[1]
            k.get_session().run(tf.local_variables_initializer())
            return auc
        
        def f1_k(y_true, y_pred):
            precision = precision_k(y_true, y_pred)
            recall = recall_k(y_true, y_pred)
            return 2 * ((precision * recall) / (precision + recall + k.epsilon()))
        
        def true_positives_k(y_true, y_pred):
            return k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
        
        def false_positives_k(y_true, y_pred):
            return k.sum(k.round(k.clip((1 - y_true) * y_pred, 0, 1)))
        
        def false_negatives_k(y_true, y_pred):
            return k.sum(k.round(k.clip(y_true * (1 - y_pred), 0, 1)))
        
        def true_negatives_k(y_true, y_pred):
            return k.sum(k.round(k.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        
        self.dict_metrics_K = {'MSE': 'mean_squared_error',
                               'RMSE': rmse_k,
                               'R-Squared': r2_k,
                               'Binary-Crossentropy': 'binary_crossentropy',
                               'ROC-AUC': roc_auc_k,
                               'F1-Score': f1_k,
                               'PR-AUC': pr_auc_k,
                               'Binary-Accuracy': 'binary_accuracy',
                               'Sensitivity': sensitivity_k,
                               'Specificity': specificity_k,
                               'Precision': precision_k,
                               'Recall': recall_k,
                               'True-Positives': true_positives_k,
                               'False-Positives': false_positives_k,
                               'False-Negatives': false_negatives_k,
                               'True-Negatives': true_negatives_k}
    
    @staticmethod
    def _append_ext(fn):
        return fn + ".jpg"
    
    def _load_data_features(self):
        for fold in self.folds:
            self.DATA_FEATURES[fold] = pd.read_csv(
                self.path_store + 'data-features_' + self.dict_image_field_to_ids[self.organ + '_' + self.field_id]
                + '_' + self.dict_target_to_ids[self.target] + '_' + fold + '_' + self.outer_fold + '.csv')
            self.DATA_FEATURES[fold]['eid'] = self.DATA_FEATURES[fold]['eid'].astype(str).apply(self._append_ext)
            self.DATA_FEATURES[fold] = self.DATA_FEATURES[fold].set_index('eid', drop=False)
    
    def _take_subset_to_debug(self):
        for fold in self.folds:
            # use +1 or +2 to test the leftovers pipeline
            leftovers_extra = {'train': 0, 'val': 1, 'test': 2} #TODO fails with val = 1
            n_batches = math.ceil(len(self.DATA_FEATURES[fold].index) / self.batch_size * self.debug_fraction)
            n_limit_fold = leftovers_extra[fold] + self.batch_size * n_batches
            self.DATA_FEATURES[fold] = self.DATA_FEATURES[fold].iloc[:n_limit_fold, :]
    
    def _generate_generators(self, DATA_FEATURES):
        GENERATORS = {}
        for fold in self.folds:
            # do not generate a generator if there are no samples (can happen for leftovers generators)
            if fold not in DATA_FEATURES.keys():
                continue
            
            # define image generators
            if (fold == 'train') & (self.mode == 'model_training'):
                shuffle = True
                data_augmentation = True
            else:
                shuffle = False
                data_augmentation = False
            """
            if (fold == 'train') & (self.mode == 'model_training'):
                datagen = ImageDataGenerator(rotation_range=self.dict_rotation_ranges[self.organ],
                                             width_shift_range=self.dict_shift_ranges[self.organ],
                                             height_shift_range=self.dict_shift_ranges[self.organ],
                                             rescale=1. / 255.)
                shuffle = True
            else:
                datagen = ImageDataGenerator(rescale=1. / 255.)
                shuffle = False
            """
            # define batch size for testing: data is split between a part that fits in batches, and leftovers
            if self.mode == 'model_testing':
                batch_size_fold = min(self.batch_size, len(DATA_FEATURES[fold].index))
            else:
                batch_size_fold = self.batch_size
            """
            # define data generator
            generator_fold = datagen.flow_from_dataframe(dataframe=DATA_FEATURES[fold], directory=self.images_directory,
                                                         x_col='eid', y_col=self.target, color_mode='rgb',
                                                         batch_size=batch_size_fold, seed=self.seed, shuffle=shuffle,
                                                         class_mode='raw',
                                                         target_size=(self.image_size, self.image_size))
            """
            generator_fold = MyImageDataGenerator(target=self.target, field_id=self.field_id,
                                                  data_features=DATA_FEATURES[fold], batch_size=batch_size_fold,
                                                  shuffle=shuffle, dir_images=self.dir_images,
                                                  images_width=self.image_size, images_height=self.image_size,
                                                  data_augmentation=data_augmentation, seed=self.seed)
            
            # assign variables to their names
            GENERATORS[fold] = generator_fold
        return GENERATORS
    
    def _generate_class_weights(self):
        if self.dict_prediction_types[self.target] == 'binary':
            self.class_weights = {}
            counts = self.DATA_FEATURES['train'].value_counts()
            for i in counts.index.values:
                self.class_weights[i] = 1 / counts.loc[i]
    
    def _generate_base_model(self):
        base_model = None
        x = None
        if self.architecture in ['VGG16', 'VGG19']:
            if self.architecture == 'VGG16':
                from keras.applications.vgg16 import VGG16
                base_model = VGG16(include_top=False, weights=self.keras_weights, input_shape=(224, 224, 3))
            elif self.architecture == 'VGG19':
                from keras.applications.vgg19 import VGG19
                base_model = VGG19(include_top=False, weights=self.keras_weights, input_shape=(224, 224, 3))
            x = base_model.output
            x = Flatten()(x)
            x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
            x = Dropout(self.dropout_rate)(x)
            x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
            x = Dropout(self.dropout_rate)(x)
        elif self.architecture in ['MobileNet', 'MobileNetV2']:
            if self.architecture == 'MobileNet':
                from keras.applications.mobilenet import MobileNet
                base_model = MobileNet(include_top=False, weights=self.keras_weights, input_shape=(224, 224, 3))
            elif self.architecture == 'MobileNetV2':
                from keras.applications.mobilenet_v2 import MobileNetV2
                base_model = MobileNetV2(include_top=False, weights=self.keras_weights, input_shape=(224, 224, 3))
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
        elif self.architecture in ['DenseNet121', 'DenseNet169', 'DenseNet201']:
            if self.architecture == 'DenseNet121':
                from keras.applications.densenet import DenseNet121
                base_model = DenseNet121(include_top=True, weights=self.keras_weights, input_shape=(224, 224, 3))
            elif self.architecture == 'DenseNet169':
                from keras.applications.densenet import DenseNet169
                base_model = DenseNet169(include_top=True, weights=self.keras_weights, input_shape=(224, 224, 3))
            elif self.architecture == 'DenseNet201':
                from keras.applications.densenet import DenseNet201
                base_model = DenseNet201(include_top=True, weights=self.keras_weights, input_shape=(224, 224, 3))
            base_model = Model(base_model.inputs, base_model.layers[-2].output)
            x = base_model.output
        elif self.architecture in ['NASNetMobile', 'NASNetLarge']:
            if self.architecture == 'NASNetMobile':
                from keras.applications.nasnet import NASNetMobile
                base_model = NASNetMobile(include_top=True, weights=self.keras_weights, input_shape=(224, 224, 3))
            elif self.architecture == 'NASNetLarge':
                from keras.applications.nasnet import NASNetLarge
                base_model = NASNetLarge(include_top=True, weights=self.keras_weights, input_shape=(331, 331, 3))
            base_model = Model(base_model.inputs, base_model.layers[-2].output)
            x = base_model.output
        elif self.architecture == 'Xception':
            from keras.applications.xception import Xception
            base_model = Xception(include_top=False, weights=self.keras_weights, input_shape=(299, 299, 3))
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
        elif self.architecture == 'InceptionV3':
            from keras.applications.inception_v3 import InceptionV3
            base_model = InceptionV3(include_top=False, weights=self.keras_weights, input_shape=(299, 299, 3))
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
        elif self.architecture == 'InceptionResNetV2':
            from keras.applications.inception_resnet_v2 import InceptionResNetV2
            base_model = InceptionResNetV2(include_top=False, weights=self.keras_weights, input_shape=(299, 299, 3))
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
        elif self.architecture in ['ResNet50V2', 'ResNet152V2', 'ResNext101']:
            import keras
            kwargs = {"backend": keras.backend, "layers": keras.layers, "models": keras.models, "utils": keras.utils}
            model_builder = None
            if self.architecture == 'ResNet50V2':
                from keras_applications.resnet_v2 import ResNet50V2
                model_builder = ResNet50V2
            elif self.architecture == 'ResNet152V2':
                from keras_applications.resnet_v2 import ResNet152V2
                model_builder = ResNet152V2
            elif self.architecture == 'ResNeXt101':
                from keras_applications.resnext import ResNeXt101
                model_builder = ResNeXt101
            base_model = model_builder(include_top=False, weights=self.keras_weights, input_shape=(224, 224, 3),
                                       **kwargs)
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
        elif self.architecture == 'EfficientNetB7':
            from efficientnet.keras import EfficientNetB7
            w = 'noisy-student' if self.keras_weights == 'imagenet' else self.keras_weights
            base_model = EfficientNetB7(include_top=False, weights=w, input_shape=(300, 300, 3))
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dropout(self.dropout_rate)(x)
        return x, base_model.input
    
    def _generate_architecture_v2(self):
        # define the arguments
        # take special initial weights for EfficientNetB7 (better)
        if self.architecture == 'EfficientNetB7' & self.keras_weights == 'imagenet':
            w = 'noisy-student'
        else:
            w = self.keras_weights
        kwargs = {"include_top": False, "weights": w, "input_shape": (self.self.image_size, self.self.image_size, 3)}
        if self.architecture in ['ResNet50V2', 'ResNet152V2', 'ResNext101']:
            import keras
            kwargs.update(
                {"backend": keras.backend, "layers": keras.layers, "models": keras.models, "utils": keras.utils})
        
        # load the architecture builder
        ModelBuilder = None
        if self.architecture == 'VGG16':
            from keras.applications.vgg16 import VGG16 as ModelBuilder
        elif self.architecture == 'VGG19':
            from keras.applications.vgg19 import VGG19 as ModelBuilder
        elif self.architecture == 'MobileNet':
            from keras.applications.mobilenet import MobileNet as ModelBuilder
        elif self.architecture == 'MobileNetV2':
            from keras.applications.mobilenet_v2 import MobileNetV2 as ModelBuilder
        elif self.architecture == 'DenseNet121':
            from keras.applications.densenet import DenseNet121 as ModelBuilder
        elif self.architecture == 'DenseNet169':
            from keras.applications.densenet import DenseNet169 as ModelBuilder
        elif self.architecture == 'DenseNet201':
            from keras.applications.densenet import DenseNet201 as ModelBuilder
        if self.architecture == 'NASNetMobile':
            from keras.applications.nasnet import NASNetMobile as ModelBuilder
        elif self.architecture == 'NASNetLarge':
            from keras.applications.nasnet import NASNetLarge as ModelBuilder
        elif self.architecture == 'Xception':
            from keras.applications.xception import Xception as ModelBuilder
        elif self.architecture == 'InceptionV3':
            from keras.applications.inception_v3 import InceptionV3 as ModelBuilder
        elif self.architecture == 'InceptionResNetV2':
            from keras.applications.inception_resnet_v2 import InceptionResNetV2 as ModelBuilder
        elif self.architecture == 'ResNet50':
            from keras_applications.resnet import ResNet50V2 as ModelBuilder
        elif self.architecture == 'ResNet101':
            from keras_applications.resnet import ResNet101V2 as ModelBuilder
        elif self.architecture == 'ResNet152':
            from keras_applications.resnet import ResNet152V2 as ModelBuilder
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
            from efficientnet.keras import EfficientNetB7 as ModelBuilder
        
        # build the model's base
        base_model = ModelBuilder(kwargs)
        x = base_model.output
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
        
        return x, base_model.input
    
    def _complete_architecture_v2(self, x, input_shape):
        for n in [int(2 ** (10 - i)) for i in range(7)]:
            x = Dense(n, activation='selu', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
            x = Dropout(self.dropout_rate)(x)
        predictions = Dense(1, activation=self.dict_final_activations[self.prediction_type])(x)
        self.model = Model(inputs=input_shape, outputs=predictions)
    
    def _complete_architecture(self, x, input_shape):
        for n in [int(2 ** (10 - i)) for i in range(5)]:
            x = Dense(n, activation='selu', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
            x = Dropout(self.dropout_rate)(x)
        predictions = Dense(1, activation=self.dict_final_activations[self.prediction_type])(x)
        self.model = Model(inputs=input_shape, outputs=predictions)
    
    def _generate_architecture(self):
        x, base_model_input = self._generate_base_model()
        self._complete_architecture(x=x, input_shape=base_model_input)
    
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
    
    def clean_exit(self):
        # exit
        print('\nTHE MODEL CONVERGED!\n')
        print('Closing the GPU session before exiting...')
        self.gpu_session.close()
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
    def __init__(self, target=None, organ_id_view=None, transformation=None, architecture=None, optimizer=None,
                 learning_rate=None, weight_decay=None, dropout_rate=None, outer_fold=None, debug_mode=False,
                 max_transfer_learning=False, continue_training=True, display_full_metrics=True):
        
        # parameters
        DeepLearning.__init__(self, target, organ_id_view, transformation, architecture, optimizer, learning_rate,
                              weight_decay, dropout_rate, debug_mode)
        self.outer_fold = outer_fold
        self.version = self.version + '_' + str(self.outer_fold)
        # NNet's architecture's weights
        self.continue_training = continue_training
        self.max_transfer_learning = max_transfer_learning
        self.list_parameters_to_match = ['organ', 'transformation', 'field_id', 'view']
        # dict to decide in which order targets should be used when trying to transfer weight from a similar model
        self.dict_alternative_targets_for_transfer_learning = {'Age': ['Age', 'Sex'], 'Sex': ['Sex', 'Age']}
        
        # Generators
        self.folds = ['train', 'val']
        self.mode = 'model_training'
        self.class_weights = None
        self.GENERATORS = None
        
        # Metrics
        self.loss_name = self.dict_losses_names[self.prediction_type]
        self.loss_function = self.dict_metrics_K[self.loss_name]
        self.main_metric_name = self.dict_main_metrics_names[self.target]
        self.main_metric_mode = self.main_metrics_modes[self.main_metric_name]
        self.main_metric = self.dict_metrics_K[self.main_metric_name]
        self.display_full_metrics = display_full_metrics
        if self.display_full_metrics:
            self.metrics_names = self.dict_metrics_names[self.prediction_type]
        else:
            self.metrics_names = [self.main_metric_name]
        self.metrics = [self.dict_metrics_K[metric_name] for metric_name in self.metrics_names]
        self.baseline_performance = None
        
        # Model
        self.path_load_weights = self.path_store + 'model-weights_' + self.version + '.h5'
        if self.debug_mode:
            self.path_save_weights = self.path_store + 'mw-debug_' + self.version + '.h5'
        else:
            self.path_save_weights = self.path_store + 'model-weights_' + self.version + '.h5'
        self.n_epochs_max = 1000
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
                        Performances['field_id'] = Performances['field_id'].astype(str)
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
    
    def _set_learning_rate(self):
        opt = self.optimizers[self.optimizer](lr=self.learning_rate, clipnorm=1.0)
        self.model.compile(optimizer=opt, loss=self.loss_function, metrics=self.metrics)
    
    def _compute_baseline_performance(self):
        # calculate initial val_loss value
        if self.continue_training:
            idx_metric_name = ([self.loss_name] + self.metrics_names).index(self.main_metric_name)
            baseline_perfs = self.model.evaluate_generator(self.GENERATORS['val'], steps=self.GENERATORS['val'].steps)
            self.baseline_performance = baseline_perfs[idx_metric_name]
        elif self.main_metric_mode == 'min':
            self.baseline_performance = np.Inf
        else:
            self.baseline_performance = -np.Inf
        print('Baseline validation performance is: ' + str(self.baseline_performance))
    
    def _define_callbacks(self):
        csv_logger = CSVLogger(self.path_store + 'logger_' + self.version + '.csv', separator=',',
                               append=self.continue_training)
        model_checkpoint_backup = MyModelCheckpoint(self.path_save_weights.replace('model-weights',
                                                                                   'backup-model-weights'),
                                                    monitor='val_' + self.main_metric.__name__,
                                                    baseline=self.baseline_performance, verbose=1, save_best_only=True,
                                                    save_weights_only=True, mode=self.main_metric_mode)
        model_checkpoint = MyModelCheckpoint(self.path_save_weights,
                                             monitor='val_' + self.main_metric.__name__,
                                             baseline=self.baseline_performance, verbose=1, save_best_only=True,
                                             save_weights_only=True, mode=self.main_metric_mode)
        reduce_lr_on_plateau = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=1, mode='min',
                                                 min_delta=0, cooldown=0, min_lr=0)
        early_stopping = EarlyStopping(monitor='val_' + self.main_metric.__name__, min_delta=0, patience=10, verbose=0,
                                       mode=self.main_metric_mode, baseline=self.baseline_performance)
        self.callbacks = [csv_logger, model_checkpoint_backup, model_checkpoint, reduce_lr_on_plateau, early_stopping]
    
    def build_model(self):
        self._weights_for_transfer_learning()
        self._generate_architecture()
        if self.keras_weights is None:
            self._load_model_weights()
        else:
            # save imagenet weights as default, in case no better weights are found
            self.model.save_weights(self.path_save_weights.replace('model-weights', 'backup-model-weights'))
            self.model.save_weights(self.path_save_weights)
        self._set_learning_rate()
        self._compute_baseline_performance()
        self._define_callbacks()
    
    def train_model(self):
        # garbage collector
        gc.collect()
        
        # train the model
        verbose = 1 if self.debug_mode else 2
        self.model.fit_generator(generator=self.GENERATORS['train'], steps_per_epoch=self.GENERATORS['train'].steps,
                                 validation_data=self.GENERATORS['val'], validation_steps=self.GENERATORS['val'].steps,
                                 use_multiprocessing=True, workers=self.n_cpus, epochs=self.n_epochs_max,
                                 class_weight=self.class_weights, callbacks=self.callbacks, verbose=verbose)


class PredictionsGenerate(DeepLearning):
    
    def __init__(self, target=None, organ_id_view=None, transformation=None, architecture=None, optimizer=None,
                 learning_rate=None, weight_decay=None, dropout_rate=None, debug_mode=False):
        
        DeepLearning.__init__(self, target, organ_id_view, transformation, architecture, optimizer, learning_rate,
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
            n_leftovers = len(self.DATA_FEATURES[fold].index) % self.batch_size
            if n_leftovers > 0:
                self.DATA_FEATURES_BATCH[fold] = self.DATA_FEATURES[fold].iloc[:-n_leftovers]
                self.DATA_FEATURES_LEFTOVERS[fold] = self.DATA_FEATURES[fold].tail(n_leftovers)
            else:
                self.DATA_FEATURES_BATCH[fold] = self.DATA_FEATURES[fold]  # special case for syntax if no leftovers
    
    #def _generate_predictions_leftovers(self, ):
    
    def _generate_outerfolds_predictions(self):
        # prepare unscaling
        if self.target in self.targets_regression:
            mean_train = self.DATA_FEATURES['train'][self.target + '_raw'].mean()
            std_train = self.DATA_FEATURES['train'][self.target + '_raw'].std()
        else:
            mean_train, std_train = None, None
        
        # Generate predictions
        #for fold in self.folds: TODO
        for fold in ['val', 'test']:
            print('Predicting samples from fold ' + fold)
            print('Predicting batches: ' + str(len(self.DATA_FEATURES_BATCH[fold].index)) + ' samples.')
            pred_batch = self.model.predict_generator(self.GENERATORS_BATCH[fold],
                                                      steps=self.GENERATORS_BATCH[fold].steps, verbose=1)
            if fold in self.GENERATORS_LEFTOVERS.keys():  # TODO
                print('Predicting leftovers: ' + str(len(self.DATA_FEATURES_LEFTOVERS[fold].index)) + ' samples.')
                pred_leftovers = self.model.predict_generator(self.GENERATORS_LEFTOVERS[fold],
                                                              steps=self.GENERATORS_LEFTOVERS[fold].steps, verbose=1)
                pred_full = np.concatenate((pred_batch, pred_leftovers)).squeeze()
            else:
                pred_full = pred_batch.squeeze()
            print('Predicted a total of ' + str(len(pred_full)) + ' samples.')
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
            self.PREDICTIONS[fold]['eid'] = [eid.replace('.jpg', '') for eid in self.PREDICTIONS[fold]['eid']]
    
    def _generate_and_concatenate_predictions(self):
        for outer_fold in self.outer_folds:
            self.outer_fold = outer_fold
            print('Predicting samples for the outer_fold = ' + self.outer_fold)
            self.path_load_weights = self.path_store + 'model-weights_' + self.version + '_' + outer_fold + '.h5'
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
            self.PREDICTIONS[fold].index.name = 'column_names'
            self.PREDICTIONS[fold] = self.PREDICTIONS[fold][['eid', 'outer_fold', 'pred']]
    
    def _correct_age_instance(self):
        data_features = pd.read_csv(self.path_store + 'data-features.csv')
        data_features['eid'] = data_features['eid'].astype(str)
        data_features['correction'] = data_features['Age'] - data_features['Age_Imaging']
        data_features = data_features[['eid', 'correction']]
        data_features.set_index('eid', drop=False, inplace=True)
        data_features.index.name = 'column_names'
        
        for fold in self.folds:
            self.PREDICTIONS[fold] = self.PREDICTIONS[fold].merge(data_features, how='inner', on=['eid'])
            self.PREDICTIONS[fold]['pred'] = self.PREDICTIONS[fold]['pred'] + self.PREDICTIONS[fold]['correction']
            self.PREDICTIONS[fold] = self.PREDICTIONS[fold][['eid', 'outer_fold', 'pred']]
    
    def _postprocess_predictions(self):
        self._format_predictions()
        if (self.target == 'Age') & (self.field_id in self.list_field_ids_in_instance_2):
            self._correct_age_instance()
    
    def generate_predictions(self):
        self._generate_architecture()
        self._generate_and_concatenate_predictions()
        self._postprocess_predictions()
    
    def save_predictions(self):
        for fold in self.folds:
            self.PREDICTIONS[fold].to_csv(self.path_store + 'Predictions_' + self.version + '_' + fold + '.csv',
                                          index=False)


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
        self.data_features = pd.read_csv(self.path_store + 'data-features.csv',
                                         usecols=['eid', 'Sex', 'Age', 'Age_Imaging'])
        self.data_features['eid'] = self.data_features['eid'].astype(str)
        self.data_features = self.data_features.set_index('eid', drop=False)
        self.data_features.index.name = 'column_names'
    
    def _preprocess_data_features(self):
        # For the training set, each sample is predicted n_CV_outer_folds times, so prepare a larger dataframe
        if self.fold == 'train':
            for outer_fold in self.outer_folds:
                df_fold = self.data_features.copy()
                df_fold['outer_fold'] = outer_fold
                df_all_folds = df_fold if outer_fold == self.outer_folds[0] else df_all_folds.append(df_fold)
            self.data_features = df_all_folds
    
    def _list_models(self):
        # generate list of predictions that will be integrated in the Predictions dataframe
        self.list_models = glob.glob(self.path_store + 'Predictions_' + self.target + '_*_' + self.fold + '.csv')
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
        list_subgroups = list(set(['_'.join(model.split('_')[2:6]) for model in self.list_models]))
        for subgroup in list_subgroups:
            print('Merging models from the subgroup ' + subgroup)
            models_subgroup = [model for model in self.list_models if subgroup in model]
            Predictions_subgroup = None
            # merge the models one by one
            for file_name in models_subgroup:
                i += 1
                print('Merging the ' + str(i) + 'th model: ' + file_name.replace(self.path_store + 'Predictions_',
                                                                                 '').replace('.csv', ''))
                # load csv and format the predictions
                prediction = pd.read_csv(self.path_store + file_name)
                print('raw prediction\'s shape: ' + str(prediction.shape))
                prediction['eid'] = prediction['eid'].apply(str)
                prediction['outer_fold'] = prediction['outer_fold'].apply(str)
                version = '_'.join(file_name.split('_')[1:-1])
                prediction['outer_fold_' + version] = prediction[
                    'outer_fold']  # create an extra column for further merging purposes on fold == 'train'
                prediction.rename(columns={'pred': 'pred_' + version}, inplace=True)
                
                # merge data frames
                if Predictions_subgroup is None:
                    Predictions_subgroup = prediction
                elif self.fold == 'train':
                    Predictions_subgroup = Predictions_subgroup.merge(prediction, how='outer', on=['eid', 'outer_fold'])
                else:
                    prediction = prediction.drop(['outer_fold'], axis=1)
                    # not supported for panda version > 0.23.4 for now
                    Predictions_subgroup = Predictions_subgroup.merge(prediction, how='outer', on=['eid'])
                # print('prediction\'s shape: ' + str(prediction.shape))
            
            # merge group predictions data frames
            if self.Predictions_df is None:
                self.Predictions_df = Predictions_subgroup
            elif self.fold == 'train':
                self.Predictions_df = self.Predictions_df.merge(Predictions_subgroup, how='outer',
                                                                on=['eid', 'outer_fold'])
            else:
                Predictions_subgroup = Predictions_subgroup.drop(['outer_fold'], axis=1)
                # not supported for panda version > 0.23.4 for now
                self.Predictions_df = self.Predictions_df.merge(Predictions_subgroup, how='outer', on=['eid'])
            print('Predictions_df\'s shape: ' + str(self.Predictions_df.shape))
            # garbage collector
            gc.collect()
    
    def postprocessing(self):
        # get rid of useless rows in data_features before merging to keep the memory requirements as low as possible
        self.data_features = self.data_features[self.data_features['eid'].isin(self.Predictions_df['eid'].values)]
        # merge data_features and predictions
        if self.fold == 'train':
            self.Predictions_df = self.data_features.merge(self.Predictions_df, how='outer', on=['eid', 'outer_fold'])
        else:
            # not supported for panda version > 0.23.4 for now
            self.Predictions_df = self.data_features.merge(self.Predictions_df, how='outer', on=['eid'])
        
        # remove rows for which no prediction is available (should be none)
        subset_cols = [col for col in self.Predictions_df.columns if 'pred_' in col]
        self.Predictions_df.dropna(subset=subset_cols, how='all', inplace=True)
        
        # Format the dataframe
        self.Predictions_df.drop(['Age_Imaging', 'outer_fold'], axis=1, inplace=True)
    
    def save_merged_predictions(self):
        self.Predictions_df.to_csv(
            self.path_store + 'PREDICTIONS_withoutEnsembles_' + self.target + '_' + self.fold + '.csv', index=False)


class PerformancesGenerate(Metrics):
    
    def __init__(self, target=None, image_type=None, transformation=None, architecture=None, optimizer=None,
                 learning_rate=None, weight_decay=None, dropout_rate=None, fold=None, debug_mode=False):
        
        Metrics.__init__(self)
        
        self.target = target
        self.image_type = image_type
        self.organ = self.image_type.split('_')[0]
        self.field_id = self.image_type.split('_')[1]
        self.view = self.image_type.split('_')[2]
        self.architecture = architecture
        self.transformation = transformation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.fold = fold
        
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
        self.version = self.target + '_' + self.image_type + '_' + self.transformation + '_' + self.architecture + '_' \
                       + self.optimizer + '_' + learning_rate_version + '_' + weight_decay_version + '_' \
                       + dropout_rate_version
        self.names_metrics = self.dict_metrics_names[self.dict_prediction_types[target]]
        self.data_features = None
        self.Predictions = None
        self.PERFORMANCES = None
        
    def _preprocess_data_features_predictions_for_performances(self):
        # load dataset
        data_features = pd.read_csv(self.path_store + 'data-features.csv', usecols=['eid', 'Sex', 'Age'])
        # format data_features to extract y
        data_features.rename(columns={self.target: 'y'}, inplace=True)
        data_features = data_features[['eid', 'y']]
        data_features['eid'] = data_features['eid'].astype(str)
        data_features['eid'] = data_features['eid']
        data_features = data_features.set_index('eid', drop=False)
        data_features.index.name = 'columns_names'
        self.data_features = data_features
    
    def _preprocess_predictions_for_performances(self):
        Predictions = pd.read_csv(self.path_store + 'Predictions_' + self.version + '_' + self.fold + '.csv')
        Predictions['eid'] = Predictions['eid'].astype(str)
        Predictions.rename(columns={'Pred_' + self.version: 'pred'}, inplace=True)
        self.Predictions = Predictions.merge(self.data_features, how='inner', on=['eid'])
    
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
            performances[col_name] = performances[col_name].astype(
                'Int64')  # need recent version of pandas to use this type. Otherwise nan cannot be int
        
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
                self.PERFORMANCES[''][col_name] = self.PERFORMANCES[''][col_name].astype(
                    'Int64')  # need recent version of pandas to use this type. Otherwise nan cannot be int
    
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
    
    def save_performances(self):
        for mode in self.modes:
            path_save = self.path_store + 'Performances_' + self.version + '_' + self.fold + mode + '.csv'
            self.PERFORMANCES[mode].to_csv(path_save, index=False)


class PerformancesMerge(Metrics):
    
    def __init__(self, target=None, fold=None, ensemble_models=False):
        
        Metrics.__init__(self)
        
        self.target = target
        self.fold = fold
        self.ensemble_models = self.convert_string_to_boolean(ensemble_models)
        self.names_metrics = self.dict_metrics_names[self.dict_prediction_types[self.target]]
        
        # list the models that need to be merged
        self.list_models = glob.glob(self.path_store + 'Performances_' + self.target + '_*_' + self.fold + '_str.csv')
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
        names_col_Performances = ['version'] + self.names_model_parameters  # .copy()
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
            version = '_'.join(model.split('_')[1:-2])
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
            Performances_withoutEnsembles = pd.read_csv(
                self.path_store + 'PERFORMANCES_tuned_alphabetical_' + self.target + '_' + self.fold + '.csv')
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
        path = self.path_store + 'PERFORMANCES_' + name_extension + '_alphabetical_' + self.target + '_' + self.fold \
               + '.csv'
        self.Performances_alphabetical.to_csv(path, index=False)
        self.Performances_ranked.to_csv(path.replace('_alphabetical_', '_ranked_'), index=False)


class PerformancesTuning(Metrics):
    
    def __init__(self, target=None):
        
        Metrics.__init__(self)
        self.target = target
        self.PERFORMANCES = {}
        self.PREDICTIONS = {}
        self.Performances = None
        self.models = None
    
    def load_data(self):
        for fold in self.folds:
            path = self.path_store + 'PERFORMANCES_withoutEnsembles_ranked_' + self.target + '_' + fold + '.csv'
            self.PERFORMANCES[fold] = pd.read_csv(path).set_index('version', drop=False)
            self.PERFORMANCES[fold]['field_id'] = self.PERFORMANCES[fold]['field_id'].astype(str)
            self.PERFORMANCES[fold].index.name = 'columns_names'
            self.PREDICTIONS[fold] = pd.read_csv(path.replace('PERFORMANCES', 'PREDICTIONS').replace('_ranked', ''))
    
    def preprocess_data(self):
        # Get list of distinct models without taking into account hyperparameters tuning
        self.Performances = self.PERFORMANCES['val']
        self.Performances['model'] = self.Performances['organ'] + '_' + self.Performances['field_id'] + '_' \
                                     + self.Performances['view'] + '_' + self.Performances['transformation'] + '_' \
                                     + self.Performances['architecture']
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
    
    def save_data(self):
        # Save the files
        for fold in self.folds:
            path_pred = self.path_store + 'PREDICTIONS_tuned_' + self.target + '_' + fold + '.csv'
            path_perf = self.path_store + 'PERFORMANCES_tuned_ranked_' + self.target + '_' + fold + '.csv'
            self.PREDICTIONS[fold].to_csv(path_pred, index=False)
            self.PERFORMANCES[fold].to_csv(path_perf, index=False)
            Performances_alphabetical = self.PERFORMANCES[fold].sort_values(by='version')
            Performances_alphabetical.to_csv(path_perf.replace('ranked', 'alphabetical'), index=False)


class EnsemblesPredictions(Metrics):
    
    def __init__(self, target=None):
        
        # set parameters
        Metrics.__init__(self)
        self.target = target
        self.ensembles_performance_cutoff_percent = 0
        self.parameters = {'target': self.target, 'organ': '*', 'field_id': '*', 'view': '*', 'transformation': '*',
                           'architecture': '*', 'optimizer': '*', 'learning_rate': '*', 'weight_decay': '*',
                           'dropout_rate': '*'}
        self.version = self._parameters_to_version(self.parameters)
        self.main_metric_name = self.dict_main_metrics_names[target]
        self.init_perf = -np.Inf if self.main_metrics_modes[self.main_metric_name] == 'max' else np.Inf
        path_perf = self.path_store + 'PERFORMANCES_tuned_ranked_' + target + '_val.csv'
        self.Performances = pd.read_csv(path_perf).set_index('version', drop=False)
        self.Performances['field_id'] = self.Performances['field_id'].astype(str)
        self.list_ensemble_levels = ['transformation', 'view', 'field_id', 'organ']
        self.PREDICTIONS = {}
        self.weights_by_category = None
        self.weights_by_ensembles = None
    
    # Returns True if the dataframe is a single column duplicated.
    # Used to check if the folds are the same for the entire ensemble model
    @staticmethod
    def _is_rank_one(df):
        for i in range(len(df.columns)):
            for j in range(i + 1, len(df.columns)):
                if not df.iloc[:, i].equals(df.iloc[:, j]):
                    return False
        return True
    
    def load_data(self):
        for fold in self.folds:
            self.PREDICTIONS[fold] = pd.read_csv(
                self.path_store + 'PREDICTIONS_tuned_' + self.target + '_' + fold + '.csv')
    
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
                PREDICTIONS[fold]['pred_' + version.replace('*', ',')] = \
                    Ensemble_predictions_weighted_by_category.sum(axis=1, skipna=False)
                PREDICTIONS[fold]['pred_' + version.replace('*', '?')] = \
                    Ensemble_predictions_weighted_by_ensembles.sum(axis=1, skipna=False)
    
    def _build_single_ensemble_wrapper(self, Performances, parameters, version, list_ensemble_levels, ensemble_level):
        Predictions = self.PREDICTIONS['val']
        # Select the outerfolds columns for the model
        ensemble_outerfolds_cols = [name_col for name_col in Predictions.columns.values if
                                    bool(re.compile('outer_fold_' + version).match(name_col))]
        Ensemble_outerfolds = Predictions[ensemble_outerfolds_cols]
        
        # Evaluate if the model can be built piece by piece on each outer_fold,
        # or if the folds are not shared and the model should be built on all the folds at once
        if not self._is_rank_one(Ensemble_outerfolds):
            self._build_single_ensemble(self.PREDICTIONS, Performances, parameters, version, list_ensemble_levels,
                                        ensemble_level)
            for fold in self.folds:
                self.PREDICTIONS[fold]['outer_fold_' + version] = np.nan
                self.PREDICTIONS[fold]['outer_fold_' + version.replace('*', ',')] = np.nan
                self.PREDICTIONS[fold]['outer_fold_' + version.replace('*', '?')] = np.nan
        else:
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
                        df_outer_fold = PREDICTIONS_outerfold[fold][['eid', 'outer_fold_' + version, 'pred_' + version]]
                    else:
                        df_outer_fold = PREDICTIONS_outerfold[fold][
                            ['eid', 'outer_fold_' + version, 'pred_' + version,
                             'outer_fold_' + version.replace('*', ','),
                             'pred_' + version.replace('*', ','), 'outer_fold_' + version.replace('*', '?'),
                             'pred_' + version.replace('*', '?')]]
                    
                    # Initiate, or append if some previous outerfolds have already been concatenated
                    if fold not in PREDICTIONS_ENSEMBLE.keys():
                        PREDICTIONS_ENSEMBLE[fold] = df_outer_fold
                    else:
                        PREDICTIONS_ENSEMBLE[fold] = PREDICTIONS_ENSEMBLE[fold].append(df_outer_fold)
            
            # Add the ensemble predictions to the dataframe
            for fold in self.folds:
                if fold == 'train':
                    self.PREDICTIONS[fold] = self.PREDICTIONS[fold].merge(PREDICTIONS_ENSEMBLE[fold], how='outer',
                                                                          on=['eid', 'outer_fold_' + version])
                else:
                    PREDICTIONS_ENSEMBLE[fold].drop('outer_fold_' + version, axis=1, inplace=True)
                    self.PREDICTIONS[fold] = self.PREDICTIONS[fold].merge(PREDICTIONS_ENSEMBLE[fold], how='outer',
                                                                          on=['eid'])
        
        # build and save a dataset for this specific ensemble model
        for ensemble_type in ['*', ',', '?']:
            version_type = version.replace('*', ensemble_type)
            if 'pred_' + version_type in self.PREDICTIONS['test'].columns.values:
                for fold in self.folds:
                    df_single_ensemble = self.PREDICTIONS[fold][
                        ['eid', 'outer_fold_' + version, 'pred_' + version_type]]
                    df_single_ensemble.rename(
                        columns={'outer_fold_' + version: 'outer_fold', 'pred_' + version_type: 'pred'}, inplace=True)
                    df_single_ensemble.dropna(inplace=True, subset=['pred'])
                    df_single_ensemble.to_csv(
                        self.path_store + 'Predictions_' + version_type + '_' + fold + '.csv', index=False)
    
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
    
    def save_predictions(self):
        for fold in self.folds:
            path_perf = self.path_store + 'PREDICTIONS_withEnsembles_' + self.target + '_' + fold + '.csv'
            self.PREDICTIONS[fold].to_csv(path_perf, index=False)


class ResidualsGenerate(Hyperparameters):
    
    def __init__(self, target=None, fold=None, debug_mode=False):
        Hyperparameters.__init__(self)
        self.target = target
        self.fold = fold
        self.debug_mode = debug_mode
        self.Residuals = pd.read_csv(
            self.path_store + 'PREDICTIONS_withEnsembles_' + target + '_' + fold + '.csv')
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
        self.Residuals.to_csv(self.path_store + 'RESIDUALS_' + self.target + '_' + self.fold + '.csv',
                              index=False)


class ResidualsCorrelations(Hyperparameters):

    def __init__(self, target=None, fold=None, debug_mode=False):
        Hyperparameters.__init__(self)
        self.target = target
        self.fold = fold
        self.debug_mode = debug_mode
        if debug_mode:
            self.n_bootstrap_iterations_correlations = 10
        else:
            self.n_bootstrap_iterations_correlations = 1000
        self.Residuals = None
        self.CORRELATIONS = {}

    def preprocessing(self):
        # load data
        Residuals = pd.read_csv(self.path_store + 'RESIDUALS_' + self.target + '_' + self.fold + '.csv')
        
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
            if i % 100 == 0:
                print('Bootstrap iteration ' + str(i) + ' out of ' + str(self.n_bootstrap_iterations_correlations))
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
            self.CORRELATIONS[mode].to_csv(
                self.path_store + 'ResidualsCorrelations' + mode + '_' + self.target + '_' + self.fold + '.csv',
                index=True)


class SelectBest(Metrics):
    
    def __init__(self, target=None):
        Metrics.__init__(self)
        
        self.target = target
        self.organs = None
        self.best_models = None
        self.PREDICTIONS = {}
        self.RESIDUALS = {}
        self.PERFORMANCES = {}
        self.CORRELATIONS = {}
    
    def _load_data(self):
        for fold in self.folds:
            path_pred = self.path_store + 'PREDICTIONS_withEnsembles_' + self.target + '_' + fold + '.csv'
            path_res = self.path_store + 'RESIDUALS_' + self.target + '_' + fold + '.csv'
            path_perf = self.path_store + 'PERFORMANCES_withEnsembles_ranked_' + self.target + '_' + fold + '.csv'
            path_corr = self.path_store + 'ResidualsCorrelations_' + self.target + '_' + fold + '.csv'
            self.PREDICTIONS[fold] = pd.read_csv(path_pred)
            self.RESIDUALS[fold] = pd.read_csv(path_res)
            self.PERFORMANCES[fold] = pd.read_csv(path_perf)
            self.PERFORMANCES[fold].set_index('version', drop=False, inplace=True)
            self.CORRELATIONS[fold] = pd.read_csv(path_corr, index_col=0)
    
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
        base_cols = ['eid', 'Sex', 'Age']
        best_models_pred = ['pred_' + model for model in self.best_models]
        best_models_res = ['res_' + model for model in self.best_models]
        best_models_corr = ['_'.join(model.split('_')[1:]) for model in self.best_models]
        for fold in self.folds:
            self.PREDICTIONS[fold] = self.PREDICTIONS[fold].loc[:, base_cols + best_models_pred]
            self.PREDICTIONS[fold].columns = base_cols + self.organs
            self.RESIDUALS[fold] = self.RESIDUALS[fold].loc[:, base_cols + best_models_res]
            self.RESIDUALS[fold].columns = base_cols + self.organs
            self.PERFORMANCES[fold] = self.PERFORMANCES[fold].loc[self.best_models, :]
            self.PERFORMANCES[fold].index = self.organs
            self.CORRELATIONS[fold] = self.CORRELATIONS[fold].loc[best_models_corr, best_models_corr]
            self.CORRELATIONS[fold].index = self.organs
            self.CORRELATIONS[fold].columns = self.organs
    
    def select_models(self):
        self._load_data()
        self._select_versions()
        self._take_subsets()
    
    def save_data(self):
        for fold in self.folds:
            path_pred = self.path_store + 'PREDICTIONS_bestmodels_' + self.target + '_' + fold + '.csv'
            path_res = self.path_store + 'RESIDUALS_bestmodels_' + self.target + '_' + fold + '.csv'
            path_corr = self.path_store + 'ResidualsCorrelations_bestmodels_' + self.target + '_' + fold + '.csv'
            path_perf = self.path_store + 'PERFORMANCES_bestmodels_ranked_' + self.target + '_' + fold + '.csv'
            self.PREDICTIONS[fold].to_csv(path_pred, index=False)
            self.RESIDUALS[fold].to_csv(path_res, index=False)
            self.CORRELATIONS[fold].to_csv(path_corr, index=True)
            self.PERFORMANCES[fold].to_csv(path_perf, index=False)
            Performances_alphabetical = self.PERFORMANCES[fold].sort_values(by='version')
            Performances_alphabetical.to_csv(path_perf.replace('ranked', 'alphabetical'), index=False)


class PlotsCorrelations(Hyperparameters):
    
    def __init__(self, target=None, fold=None, save_figures=True):
        Hyperparameters.__init__(self)
        self.target = target
        self.fold = fold
        self.save_figures = save_figures
        self.fig_xsize = 23.4
        self.fig_ysize = 16.54
        self.Correlations = None
        self.Correlations_bestmodels = None
        
    def preprocessing(self):
        Correlations = pd.read_csv(
            self.path_store + 'ResidualsCorrelations' + '_' + self.target + '_' + self.fold + '.csv',
            index_col='Unnamed: 0')
        
        # Crop the names to make the reading of the labels easier
        idx_to_print = [self.names_model_parameters[1:].index(i) for i in ['organ', 'view', 'architecture']]
        Correlations.index = ['_'.join(np.array(idx.split('_'))[idx_to_print]) for idx in Correlations.index.values]
        Correlations.columns = ['_'.join(np.array(idx.split('_'))[idx_to_print]) for idx in Correlations.columns.values]
        self.Correlations = Correlations
        self.Correlations_bestmodels = pd.read_csv(self.path_store + 'ResidualsCorrelations_bestmodels_' + self.target +
                                                   '_' + self.fold + '.csv', index_col='Unnamed: 0')
    
    def _plot_correlations(self, data, title_save):
        # set parameters
        plt.clf()
        sns.set(font_scale=1, rc={'figure.figsize': (self.fig_xsize, self.fig_ysize)})
        
        # plot
        cor_plot = sns.heatmap(
            data=data, xticklabels=1, yticklabels=1, annot=(data * 100).round().astype(int), fmt='d',
            annot_kws={"size": 10},
            vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True)
        # optional: inclined x labels
        # cor_plot.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right');
        
        # Save figure
        if self.save_figures:
            fig = cor_plot.get_figure()
            fig.set_size_inches(self.fig_xsize, self.fig_ysize)
            fig.savefig('../figures/Correlations/' + title_save + '.png', dpi='figure')
    
    def generate_plots(self):
        title = 'Correlations_AllModels_' + self.target + '_' + self.fold
        self._plot_correlations(data=self.Correlations, title_save=title)
        title_bestmodels = 'Correlations_BestModels_' + self.target + '_' + self.fold
        self._plot_correlations(data=self.Correlations_bestmodels, title_save=title_bestmodels)
        
        # Plot the "ensemble models only" correlation plots
        for ensemble_type in self.ensemble_types:
            index_ensembles_only = [idx for idx in self.Correlations.columns.values if ensemble_type in idx]
            Correlations_Ensembles_only = self.Correlations.loc[index_ensembles_only, index_ensembles_only]
            title = 'Correlations_Ensembles' + ensemble_type + 'Only_' + self.target + '_' + self.fold
            self._plot_correlations(data=Correlations_Ensembles_only, title_save=title)


class PlotsLoggers(Hyperparameters):
    
    def __init__(self, target=None, display_learning_rate=None):
        Hyperparameters.__init__(self)
        self.target = target
        self.display_learning_rate = display_learning_rate
        self.PREDICTIONS = {}
        for fold in self.folds:
            self.PREDICTIONS[fold] = pd.read_csv(
                self.path_store + 'PREDICTIONS_withoutEnsembles_' + self.target + '_' + fold + '.csv')
        self.list_versions = [col_name.replace('pred_', '') for col_name in self.PREDICTIONS['test'].columns.values
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
            i = int(m/n_rows)
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
    
    def __init__(self, target=None):
        Hyperparameters.__init__(self)
        self.target = target
        # Load the predictions
        self.PREDICTIONS = {}
        for fold in self.folds:
            self.PREDICTIONS[fold] = pd.read_csv(
                self.path_store + 'PREDICTIONS_withEnsembles_' + self.target + '_' + fold + '.csv')
        # print scatter plots for each model
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
            multi_plot.savefig('../figures/Scatter_Plots/ScatterPlots_' + version + '.png')


class PlotsAttentionMaps(DeepLearning):
    
    def __init__(self, target=None, organ_id_view=None, transformation=None, fold=None):
        # partial initialization with placeholders to get access to parameters and functions
        DeepLearning.__init__(self, target, organ_id_view, transformation, 'VGG16', 'Adam', 0, 0, 0, False)
        
        self.fold = fold
        self.parameters = None
        self.image_size = None
        self.batch_size = None
        self.N_samples_attentionmaps = 10  # needs to be > 1 for the script to work
        
        # Pick the best model based on the performances
        organ, field_id, view = organ_id_view.split('_')
        path_perf = self.path_store + 'PERFORMANCES_withoutEnsembles_ranked_' + self.target + '_' + self.fold + '.csv'
        Performances = pd.read_csv(path_perf).set_index('version', drop=False)
        Performances = Performances[(Performances['organ'] == organ)
                                    & (Performances['field_id'].astype(str) == self.field_id)
                                    & (Performances['view'] == self.view)
                                    & (Performances['transformation'] == self.transformation)]
        version = Performances['version'].values[0]
        del Performances
        self.parameters = self._version_to_parameters(version)
        self.image_size = self.input_size_models[self.parameters['architecture']]
        self.batch_size = self.dict_batch_sizes[self.parameters['architecture']]
        
        DeepLearning.__init__(self, target, organ_id_view, transformation, self.parameters['architecture'],
                              self.parameters['optimizer'], self.parameters['learning_rate'],
                              self.parameters['weight_decay'], self.parameters['dropout_rate'], False)
        self.dir_images = '../images/' + self.organ + '/' + self.field_id + '/' + self.view + '/' + self.transformation\
                          + '/'
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
        Residuals_full = pd.read_csv(self.path_store + 'RESIDUALS_' + self.target + '_' + self.fold + '.csv')
        Residuals = Residuals_full[['eid', 'Age', 'Sex', 'res_' + self.version, 'outer_fold_' + self.version]]
        del Residuals_full
        Residuals.dropna(inplace=True)
        Residuals.rename(columns={'res_' + self.version: 'res', 'outer_fold_' + self.version: 'outer_fold'},
                         inplace=True)
        Residuals['eid'] = Residuals['eid'].astype(str).apply(self._append_ext)
        Residuals['outer_fold'] = Residuals['outer_fold'].astype(int).astype(str)
        Residuals['res_abs'] = Residuals['res'].abs()
        self.Residuals = Residuals[['eid', 'outer_fold', 'Sex', 'Age', 'res', 'res_abs']]
    
    def _select_representative_samples(self):
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
        df_to_plot['save_title'] = self.target + '_' + self.organ + '_' + self.field_id + '_' + self.view + '_' + \
                                   self.transformation + '_' + df_to_plot['Sex'] + '_' + df_to_plot['age_category'] + \
                                   '_' + df_to_plot['aging_rate'] + '_' + df_to_plot['sample'].astype(str)
        path_save = self.path_store + 'AttentionMaps-samples_' + self.target + '_' + self.organ_id_view + '_' \
                    + self.transformation + '.csv'
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
        self.generator = MyImageDataGenerator(target=self.target, field_id=self.field_id,
                                              data_features=self.df_outer_fold, batch_size=self.batch_size,
                                              shuffle=False, dir_images=self.dir_images,
                                              images_width=self.image_size, images_height=self.image_size,
                                              data_augmentation=False, seed=self.seed)
        
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
        saliency *= 255/np.max(np.abs(saliency))
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
        images = self.generator.__getitem__(i)[0][:n_images_batch, :, :, :]
        saliencies = self.saliency_analyzer.analyze(images)
        guided_backprops = self.guided_backprop_analyzer.analyze(images)
        for j in range(saliencies.shape[0]):
            # select sample
            self.image = images[j]
            self.plot_title = self.df_outer_fold['plot_title'].values[i * self.batch_size + j]
            save_title = self.df_outer_fold['save_title'].values[i * self.batch_size + j]
            
            # generate the transparent filters for saliency, grad-cam and guided-backprop maps
            self._generate_saliency_map(saliencies[j])
            self._generate_gradcam_map()
            self._generate_guidedbackprop_map(guided_backprops[j])
            
            # plot the three maps individually
            for map_type in self.dict_map_types_to_names.keys():
                self._plot_attention_map(filter_map=getattr(self, map_type + '_filter'),
                                         save_title=self.dict_map_types_to_names[map_type] + '_' + save_title)
            
            # Generate summary plot
            self._plot_attention_maps(save_title=save_title)
    
    def generate_plots(self):
        for outer_fold in self.outer_folds:
            print('Generate attention maps for outer_fold ' + outer_fold)
            gc.collect()
            self._preprocess_for_outer_fold(outer_fold)
            for i in range(math.ceil(self.n_images / self.batch_size)):
                self._generate_maps_for_one_batch(i)
