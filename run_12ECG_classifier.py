#!/usr/bin/env python

import numpy as np, os, sys
from numpy import array
from numpy import argmax
import joblib
import numpy as np, os, sys, joblib
from scipy.io import loadmat
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from get_12ECG_features import get_12ECG_features

# Import tensorflow and keras libraries
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, LSTM, Dense, Dropout, TimeDistributed
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense, Dropout
from keras.optimizers import Adam
import keras
# conver to hot-encode
from keras.utils import to_categorical

import matplotlib.pyplot as plt

def run_12ECG_classifier(data,header_data,model):
        
    # Load Classes
    classes = ['164884008', '164889003', '164909002', '164931005', '270492004', '284470004', '426783006', '429622005', '59118001']
    imputer = joblib.load('./model/imputer.sav')
    imputer = imputer['imputer']

    features=np.asarray(get_12ECG_features(data,header_data))    
    
    # Normalize data between 0 and 1
    features = tf.keras.utils.normalize(features)
    feats_reshape = features.reshape(1, -1)
    # Evaluate if this is really necesary
    #imputer = SimpleImputer().fit(feats_reshape) # instead of this use the trained imputer
    feats_reshape = imputer.transform(feats_reshape)
    
    prediction = model.predict(feats_reshape)[0]  
    prediction = np.float64(prediction)
    prediction = np.argmax(prediction)
    encoded = to_categorical(prediction)
    current_label = encoded.astype(int)
    
    # Suppose softmax model 
    probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])  
    current_score = model.predict(feats_reshape)
    current_score = current_score.flatten()
    # convert float32 to float 64 just in case
    return current_label, current_score, classes

def load_12ECG_model(input_directory):
    # Tensorflow & keras model saving mode
    name = 'NN_1.model'
    model = tf.keras.models.load_model(input_directory)
    return model
    '''
    # load the model from disk 
    f_out='finalized_model.sav'
    
    filename = os.path.join(input_directory,f_out)

    loaded_model = joblib.load(filename)
    #model = load_model('saved_model.sav')
    return loaded_model
    '''
    
