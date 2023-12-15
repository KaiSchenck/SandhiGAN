import math
import matplotlib
import os
import re
import librosa
import blosc
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import IPython.display as ipd
import sklearn.metrics as metrics
import keras
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPool2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import to_categorical
from tqdm import tqdm
from collections import defaultdict
from matplotlib import pyplot as plt

def wavtospec(audio, max_pad):
    spec = librosa.feature.melspectrogram(y=audio, sr=44000, n_mels=64,n_fft=2048, hop_length=16, fmin=50,fmax=350) 
    pad_width = max_pad - spec.shape[1]
    spec = np.pad(spec, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return spec

def filetospec(file_path, max_pad):
    audio, sample_rate = librosa.core.load(file_path)
    return wavtospec(audio, max_pad)

def wavtof0(audio, max_pad):
    #audio, sample_rate = librosa.core.load(file_path)
    f0, voiced, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=44000, fill_na=None)
    pad_width = max_pad - f0.shape[0]
    f0 = np.pad(f0, pad_width=(0, pad_width), mode='constant')
    return f0
    
def get_tone_net(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=(5,5), strides=(3,3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(3,3), strides=(3,3), padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))
    model.add(Dropout(0.25))
    model.add(Flatten())  
    model.add(Dense(1024))  
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1024))  
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(num_classes))  
    model.add(Activation('softmax'))  
    sgd = SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model
    
def confusion_matrix(X_test, y_test):
    y_pred = model.predict(X_test).ravel()
    y_pred_ohe = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred_ohe, axis=1) # only necessary if output has one-hot-encoding, shape=(n_samples)

    y_true_labels = np.argmax(y_test, axis=1)

    confusion_matrix = metrics.confusion_matrix(y_true=y_true_labels, y_pred=y_pred_labels)
    return confusion_matrix
    
def check(path):
    X_test =  np.array([filetospec(os.path.abspath(os.path.join(path, a)), 120) for a in tqdm(os.listdir(path))])
    y_test = np.array([np.array([0, 1, 0, 0]) for i in range(X_test.shape[0])])
    trial_results = confusion_matrix(X_test, y_test)[1]
    return trial_results

if __name__ == '__main__':

    results = []

    model = get_tone_net((60, 120, 1), 4)
    checkpoint_filepath = './l_checkpoint/checkpoint'
    model.load_weights(checkpoint_filepath).expect_partial()
    
    generated = "./epoch_val/"
    
    """
    for e in [60]:
        epoch_path = generated + "e" + str(e) + "_trial/"
        for t in [f for f in os.listdir(epoch_path)]:
            trial_path = epoch_path + str(t)
            trial_results = check(trial_path)
            results.append([e, int(t), *list(trial_results)])
    
    """      
    for e in [65]:
        epoch_path = generated + "e" + str(e) + "_norm/"
        trial_results = check(epoch_path)
        results.append([e, 1, *list(trial_results)])
  

    results = pd.DataFrame(results, columns = ['Epoch', 'Trial', 'T1', 'T2', 'T3', 'T4'])
    results.to_csv("results.csv", index=False)