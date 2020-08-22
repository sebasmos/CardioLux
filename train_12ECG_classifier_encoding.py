#!/usr/bin/env python

import numpy as np, os, sys, joblib
from scipy.io import loadmat
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from get_12ECG_features import get_12ECG_features

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path

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

def train_12ECG_classifier_encoding(input_directory, output_directory):
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
        header = headers[i]
        # 
        
        #tmp = get_12ECG_features_num(recording, header,n)
        tmp = get_12ECG_features(recording, header)

        # features containing set-up values: 
        
        # array([4.00003134e-03, 3.16780508e-05, ..]) 
        features.append(tmp)
        print("features extracted succesfully..")
    #hot encoding for applying DL
        for l in header:
            if l.startswith('#Dx:'):
                labels_act = np.zeros(num_classes)
                arrs = l.strip().split(' ')
                for arr in arrs[1].split(','):
                    class_index = classes.index(arr.rstrip()) # Only use first positive index
                    labels_act[class_index] = 1
        labels.append(labels_act)

    # Replace NaN values with mean values
    imputer=SimpleImputer().fit(features)
    features=imputer.transform(features)

    # Train the classifier
    sequence_size = features.shape[1]
    n_features=1
    print("training the model")
    ############# MODEL 1 - NN ######################
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    # use 128 neurons & use relu act func
    model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
    # the final output layer must be Dense and must fit # classifications and must use probability distribution instead of an activation function
    model.add(tf.keras.layers.Dense(9, activation = tf.nn.softmax))
    
    # Compile 
    model.compile(
    optimizer= 'adam' ,# NN intends to minimize losss, not maximize accuracy
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
    )
    label_final = []
    
    # Extract hot-encode format to normal format
    for i in range(labels.shape[0]):
        label = labels[i]
        decoded_labels = decode(label)
        label_final.append(decoded_labels)
    # Convert list to np array
    label_final = np.array(label_final)
    
    # Fit the model appropiatelly
    model.fit(features, label_final, epochs=3)
    val_loss, val_acc = model.evaluate(features, label_final)
    print("model is trained")
    
    '''
    
    cnn_model = Sequential([
    Conv1D(
        filters=8,
        kernel_size=4,
        strides=1,
        input_shape=(sequence_size, n_features),
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
    cnn_model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    cnn_model.summary()
    f = np.expand_dims(features, axis=2) 
    print("features shape: ", f.shape)
    print("label features: ", labels.shape)
    
    cnn_model.fit(f, labels)
    
    # Second model (throws a "can't picke RloCK Objects")
    
    model_cnn2 = keras.Sequential()
    model_cnn2.add(keras.layers.Dense(units=27, input_dim=14))
    model_cnn2.add(keras.layers.Activation('relu'))
    model_cnn2.add(keras.layers.Dense(units=9))
    model_cnn2.add(keras.layers.Activation('sigmoid'))
    opt = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
    model_cnn2.compile(loss = 'binary_crossentropy',
              optimizer = opt,
              metrics = ['accuracy'])
    model_cnn2.fit(features, labels, epochs=100, verbose=2)   
    print('Saving model...')

    final_model={'model':model_cnn2, 'imputer':imputer}

    filename = os.path.join(output_directory, 'finalized_model_cnn.sav')
    joblib.dump(final_model, filename, protocol=0)
    '''
    # Save model.
    model.save("NN_1.model")
    


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
def decode(datum):
    return np.argmax(datum)