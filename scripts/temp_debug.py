#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 13:10:23 2019

@author: Alan
"""



class myBaseLogger(myBaseLogger):
    """take as input initial_val_loss instead of np.Inf, useful if model has already been trained
    """
    
    def __init__(self, stateful_metrics=None):
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()
    
    def on_epoch_begin(self, epoch, logs=None):
        self.seen = 0
        self.totals = {}
    
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        self.seen += batch_size
    
        for k, v in logs.items():
            print(k)
            if k in self.stateful_metrics:
                self.totals[k] = v
            else:
                if k in self.totals:
                    self.totals[k] += v * batch_size
                else:
                    self.totals[k] = v * batch_size
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for k in self.params['metrics']:
                if k in self.totals:
                    # Make value available to next callbacks.
                    if k in self.stateful_metrics:
                        logs[k] = self.totals[k]
                    else:
                        logs[k] = self.totals[k] / self.seen


class BaseLogger(Callback):
    """Callback that accumulates epoch averages of metrics.
    This callback is automatically applied to every Keras model.
    # Arguments
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over an epoch.
            Metrics in this list will be logged as-is in `on_epoch_end`.
            All others will be averaged in `on_epoch_end`.
    """
    
    def __init__(self, stateful_metrics=None):
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()
    
    def on_epoch_begin(self, epoch, logs=None):
        self.seen = 0
        self.totals = {}
    
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        self.seen += batch_size
    
        for k, v in logs.items():
            print(k)
            if k in self.stateful_metrics:
                self.totals[k] = v
            else:
                if k in self.totals:
                    self.totals[k] += v * batch_size
                else:
                    self.totals[k] = v * batch_size
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for k in self.params['metrics']:
                if k in self.totals:
                    # Make value available to next callbacks.
                    if k in self.stateful_metrics:
                        logs[k] = self.totals[k]
                    else:
                        logs[k] = self.totals[k] / self.seen

class CollectOutputAndTarget(Callback):
    def __init__(self):
        super(CollectOutputAndTarget, self).__init__()
        self.targets = []  # collect y_true batches
        self.outputs = []  # collect y_pred batches
        
        # the shape of these 2 variables will change according to batch shape
        # to handle the "last batch", specify `validate_shape=False`
        self.var_y_true = tf.Variable(0., validate_shape=False)
        self.var_y_pred = tf.Variable(0., validate_shape=False)
    
    def on_batch_end(self, batch, logs=None):
        # evaluate the variables and save them into lists
        self.targets.append(K.eval(self.var_y_true))
        self.outputs.append(K.eval(self.var_y_pred))
        print(self.targets)
        print(self.outputs)

# build a simple model
# have to compile first for model.targets and model.outputs to be prepared
model = Sequential([Dense(5, input_shape=(10,))])
model.compile(loss='mse', optimizer='adam')

# initialize the variables and the `tf.assign` ops
cbk = CollectOutputAndTarget()
fetches = [tf.assign(cbk.var_y_true, model.targets[0], validate_shape=False),
           tf.assign(cbk.var_y_pred, model.outputs[0], validate_shape=False)]
model._function_kwargs = {'fetches': fetches}  # use `model._function_kwargs` if using `Model` instead of `Sequential`

# fit the model and check results
X = np.random.rand(10, 10)
Y = np.random.rand(10, 5)
model.fit(X, Y, batch_size=8, callbacks=[cbk])




#train the model
print('TRAINING...')
history = model.fit_generator(generator=GENERATORS['train'],
                steps_per_epoch=STEP_SIZES['train'],
                validation_data=GENERATORS['val'],
                validation_steps=STEP_SIZES['val'],
                use_multiprocessing = True,
                epochs=n_epochs_max,
                class_weight=class_weights,
                callbacks=[cbk])









from sklearn.metrics import r2_score, mean_squared_error

    #TODO: save preds and truths at each batch end


class Histories_regression(keras.callbacks.Callback):
    
    def __init__(self, y_val, generator_val, step_sizes_val):
        self.y_val = y_val
        self.generator_val = generator_val
        self.step_sizes_val = step_sizes_val
        self.var_y_train = tf.Variable(0., validate_shape=False)
        self.var_pred_train = tf.Variable(0., validate_shape=False)
    
    def on_train_begin(self, logs={}):
        self.losses_train = []
        self.losses_val = []
        self.R2s_train = []
        self.R2s_val = []
        self.RMSEs_train = []
        self.RMSEs_val = []
    
    def on_epoch_begin(self, epoch, logs={}):
        self.y_train = []  # collect y_true batches
        self.pred_train = []  # collect y_pred batches
    
    def on_batch_end(self, batch, logs=None):
        # evaluate the training values and preds and save them into lists
        self.y_train.append(K.eval(self.var_y_train)) #TODO figure out why this always evaluate to [0.0]
        self.pred_train.append(K.eval(self.var_pred_train))
        print(type(K.eval(self.var_pred_train)))
        print(self.pred_train)
    
    def on_epoch_end(self, epoch, logs={}):
        print("HERE")
        self.pred_val=self.model.predict_generator(self.generator_val, self.step_sizes_val,verbose=1)
        self.R2s_train.append(r2_score(self.y_train, self.pred_train))
        self.R2s_val.append(r2_score(self.y_val, self.pred_val))
        self.RMSEs_train.append(np.sqrt(mean_squared_error(self.y_train, self.pred_train)))
        self.RMSEs_val.append(np.sqrt(mean_squared_error(self.y_val, self.pred_val)))
        self.losses_train.append(logs.get('loss'))
        self.losses_val.append(logs.get('val_loss'))
        print(self.R2s_train)
        print(self.R2s_val)
        print(self.RMSEs_train)
        print(self.RMSEs_val)
        print(self.losses_train)
        print(self.losses_val)
        print("DONE")

#define callbacks
model_checkpoint = ModelCheckpoint(path_store + 'model_weights_' + version + '.h5', monitor='val_loss', initial_val_loss=initial_val_loss, verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
reduce_lr_on_plateau = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
csv_logger = CSVLogger(path_store + 'logger_' + version + '.csv', separator=',', append=continue_training)
histories_regression = Histories_regression(y_val = DATA_FEATURES['val'][target], generator_val=GENERATORS['val'], step_sizes_val=STEP_SIZES['val'])


fetches = [tf.assign(histories_regression.var_y_train, model.targets[0], validate_shape=False), tf.assign(histories_regression.var_pred_train, model.outputs[0], validate_shape=False)]
model._function_kwargs = {'fetches': fetches}  # use `model._function_kwargs` if using `Model` instead of `Sequential`
