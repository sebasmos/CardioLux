#!/usr/bin/env python

import numpy as np, os, sys
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

import matplotlib.pyplot as plt

def run_12ECG_classifier(data,header_data,model):
        
    #Load Keras model    
    # Load Classes
    classes = ['164884008', '164889003', '164909002', '164931005', '270492004', '284470004', '426783006', '429622005', '59118001']
    # Use your classifier here to obtain a label and score for each class.
   # model = loaded_model['model']
   # imputer = loaded_model['imputer']
    #classes = loaded_model['classes']

    features=np.asarray(get_12ECG_features(data,header_data))    
    
    # Normalize data between 0 and 1
    features = tf.keras.utils.normalize(features)
    feats_reshape = features.reshape(1, -1)
    # Evaluate if this is really necesary
    imputer=SimpleImputer().fit(feats_reshape)
    feats_reshape = imputer.transform(feats_reshape)
    
    current_label = model.predict(feats_reshape)[0]    
    current_label = current_label.astype(int)
    
    # Suppose softmax model 
    probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])  
    current_score = model.predict(feats_reshape)
    
    '''
    current_score = model.predict_proba(feats_reshape)
    current_score=np.asarray(current_score)
    current_score=current_score[:,0,1]
    '''
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
    
