#!/usr/bin/env python
import os, sys

import numpy as np, os, sys, joblib
from scipy.io import loadmat
import numpy
from sklearn.impute import SimpleImputer
from get_12ECG_features import get_12ECG_features
from get_12ECG_features import fun_extract_data 

import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from train_12ECG_classifier import get_classes, load_challenge_data
# Import tensorflow and keras libraries
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, LSTM, Dense, Dropout, TimeDistributed
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense, Dropout
from keras.optimizers import Adam
import keras
# cross validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


# Functions from trainer
from train_12ECG_classifier import get_classes, load_challenge_data
def cross(input_directory, output_directory):
    print("Reading files")
    header_files = []
    for f in os.listdir(input_directory):
        g = os.path.join(input_directory, f)
        if not f.lower().startswith('.') and f.lower().endswith('hea') and os.path.isfile(g):
            header_files.append(g)
    classes = get_classes(input_directory, header_files)
    num_classes = len(classes)
    
    num_files = len(header_files)
    recordings = list()
    headers = list()

    for i in range(num_files):
        recording, header = load_challenge_data(header_files[i])
        recordings.append(recording)
        headers.append(header)

    features = list()
    labels = list()
    
    # 4813 training tests 

    for i in range(num_files):
        recording = recordings[i]
        header = headers[i]
        #tmp = fun_extract_data(recording)
        tmp = get_12ECG_features(recording, header)
        features.append(tmp)
        print("extracting feature number: ", i, "/4813")
    #hot encoding for applying DL
        for l in header:
            if l.startswith('#Dx:'):
                labels_act = np.zeros(num_classes)
                arrs = l.strip().split(' ')
                for arr in arrs[1].split(','):
                    class_index = classes.index(arr.rstrip()) # Only use first positive index
                    labels_act[class_index] = 1
        labels.append(labels_act)
        
    print("features extracted succesfully..")
    
    
    labels = np.array(labels) # 4813
    
    features = np.array(features) # 4813 x 14
    
    features = tf.keras.utils.normalize(features)
    
    imputer=SimpleImputer().fit(features)
    features=imputer.transform(features)
    print(labels.shape)
    print(features.shape)
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)
    '''
    for i in range(labels.shape[0]):
        label = labels[i]
        decoded_labels = decode(label)
        label_final.append(decoded_labels)
    # Convert list to np array
    label_final = np.array(label_final)
    label_final = np.uint8(label_final)
    '''
    # create model
    model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
    # evaluate using 10-fold cross validation
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(model, features, labels, cv=kfold)
    print(results.mean())


def decode(datum):
    return np.argmax(datum)
label_final = []
    

def create_model():
    #createmodel

    model = Sequential([
    Conv1D(
        filters=1,
        kernel_size=4,
        strides=1,
        input_shape=(14, 1),
        padding="same",
        activation="relu"
    ),
    Flatten(),
    Dropout(0.5),
    Dense(
        9,
        activation="sigmoid",
        name="output",
    )
    ])
    optimizer = Adam(lr=0.001)
    # Compiling the model
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    return model
    

    
    
if __name__ == '__main__':
    # Parse arguments.
    # input_directory = sys.argv[1]
    input_directory = "../dataset"
    #output_directory = sys.argv[2]
    output_directory = "./results"

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    print('Running cross-validation code...')

    cross(input_directory, output_directory)

    print('Done.')

