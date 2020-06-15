"""
    Classes for neural network models. 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .supervised_model import SupervisedModel
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.preprocessing import OneHotEncoder

class SimpleNN(SupervisedModel):
    def __init__(self, task_names,
                 n_features=1024,
                 batch_size=32,
                 epochs=200,
                 verbose=0):
        super(SimpleNN, self).__init__(task_names)
        # model params
        self.n_features = n_features
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        
        self.oh_enc = OneHotEncoder()
        self.oh_enc.fit(np.array([1,0]).reshape(-1,1))
        
        # setup model
        self.model_dict = {}
        for task_name in self.task_names:
            self.model_dict[task_name] = self._get_simple_model()
    
    def _get_simple_model(self):
        input = Input(shape=(self.n_features,))
        dense_1 = Dense(self.n_features, activation='relu')(input)
        dense_2 = Dense(self.n_features//2, activation='relu')(dense_1)
        softmax_output = Dense(2, activation='softmax')(dense_2)
        model = Model(inputs=input, outputs=softmax_output)
        
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam')
        return model
        
    def fit(self, X_train, y_train):
        if X_train.shape[0] > 0:
            # perform random shuffling of training data (including X_train)
            shuffle_p = np.random.permutation(X_train.shape[0])
            X_train = X_train[shuffle_p]
            y_train = y_train[shuffle_p]
            
            # fit model
            for ti, task_name in enumerate(self.task_names):
                y_train_ti = y_train[:,ti].flatten()
                nan_indices = np.where(np.isnan(y_train_ti))[0] # remove NaN labes
                y_train_ti = np.delete(y_train_ti, nan_indices, axis=0)
                X_task_ti = np.delete(X_train, nan_indices, axis=0)
                
                y_train_ti = self.oh_enc.transform(y_train_ti.reshape(-1,1)).toarray()
                
                self.model_dict[task_name].fit(X_task_ti, y_train_ti, 
                                               batch_size=self.batch_size,
                                               epochs=self.epochs,
                                               verbose=self.verbose)
            # set fit check
            self.check_fit = True
        
    def predict(self, X):
        y_pred = np.zeros(shape=(X.shape[0],len(self.task_names)))
        if self.check_fit:
            for ti, task_name in enumerate(self.task_names):
                y_pred[:,ti] = self.model_dict[task_name].predict(X, 
                                                                  batch_size=self.batch_size,
                                                                  verbose=self.verbose)[:,1]
        else: # if model has not been fit, then assume 0.5 for all samples
            y_pred[:] = 0.5
        return y_pred