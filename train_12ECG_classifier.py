#!/usr/bin/env python

import numpy as np, os, sys, joblib
from scipy.io import loadmat
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from get_12ECG_features import get_12ECG_features
from get_12ECG_features import fun_extract_data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
'''
from keras.models import Sequential
from keras.layers import Conv1D, LSTM, Dense, Dropout, TimeDistributed
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense, Dropout
from keras.optimizers import Adam
'''

import matplotlib.pyplot as plt

def train_12ECG_classifier(input_directory, output_directory):
    # Load data.
    print('Loading data...')

    header_files = []
    for f in os.listdir(input_directory):
        g = os.path.join(input_directory, f)
        if not f.lower().startswith('.') and f.lower().endswith('hea') and os.path.isfile(g):
            header_files.append(g)

    # get_classes determines 9 unique classes for the entire
    # annotated dataset
    classes = get_classes(input_directory, header_files)
    num_classes = len(classes)
    
    num_files = len(header_files)
    recordings = list()
    headers = list()

    for i in range(num_files):
        recording, header = load_challenge_data(header_files[i])
        recordings.append(recording)
        headers.append(header)
    

    # Train model.
    print('Training model...')
    

    features = list()
    labels = list()

    for i in range(num_files):
        recording = recordings[i]
        #recording_opened = load_challenge_data(header_files[i])
        header = headers[i]
        
        #tmp = fun_extract_data(recording)
        tmp = get_12ECG_featuress(recording, header)
        
        print("Extracted feature Number: ",i,"/4813")
        features.append(tmp)

    #hot encoding for applying DL
        for l in header:
            if l.startswith('#Dx:'):
                labels_act = np.zeros(num_classes)
                arrs = l.strip().split(' ')
                for arr in arrs[1].split(','):
                    class_index = classes.index(arr.rstrip()) # Only use first positive index
                    labels_act[class_index] = 1
        labels.append(labels_act)

    features = np.array(features)
    labels = np.array(labels)

    # Replace NaN values with mean values
    imputer=SimpleImputer().fit(features)
    features=imputer.transform(features)

    # Train the classifier
    model = RandomForestClassifier().fit(features,labels)

    # Save model.
    print('Saving model...')

    final_model={'model':model, 'imputer':imputer}

    filename = os.path.join(output_directory, 'finalized_model.sav')
    joblib.dump(final_model, filename, protocol=0)

# Load challenge data.
def load_challenge_data(header_file):
    with open(header_file, 'r') as f:
        header = f.readlines()
    mat_file = header_file.replace('.hea', '.mat')
    x = loadmat(mat_file)
    recording = np.asarray(x['val'], dtype=np.float64)
    # For testing: 
    # numpy.savetxt("A0001.csv",recording, delimiter=",")
    return recording, header
# Load data and convert to adequate format for encoding
def load_challenge_data_encoding(header_file):
    with open(header_file, 'r') as f:
        header = f.readlines()
    mat_file = header_file.replace('.hea', '.mat')
    x = loadmat(mat_file)
    recordings_on_arrays = list()
    recording = np.asarray(x['val'], dtype=np.float64)
    for i in range(recording.shape[0]):
        for j in range(recording.shape[1]):
            recordings_on_arrays.append(recording[i,j])
    # For testing: 
    # numpy.savetxt("A0001.csv",recording, delimiter=",")
    return recordings_on_arrays, header


# Find unique classes.
def get_classes(input_directory, filenames):
    classes = set()
    for filename in filenames:
        with open(filename, 'r') as f:
            for l in f:
                if l.startswith('#Dx'):
                    tmp = l.split(': ')[1].split(',')
                    for c in tmp:
                        classes.add(c.strip())
    return sorted(classes)
