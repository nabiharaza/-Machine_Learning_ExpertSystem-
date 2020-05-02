# Import all the libs
import numpy as np
import os
import re
import pickle
import timeit
import glob
import cv2

from skimage import transform
import skimage
from skimage import io

import sklearn
from sklearn.model_selection import train_test_split  ### import sklearn tool

import keras
from keras.preprocessing import image as image_utils
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, TimeDistributed
from skimage import io


def load_dataset(path_to_video, total_frames):
    # Capture a video
    vidcap = cv2.VideoCapture(path_to_video)
    # Check if video frame is read correctly
    success, image = vidcap.read()
    count = 0
    img = []
    # Read all frames of video iteratively
    while success:
        success, image = vidcap.read()
        img.append(image)
        count += 1
        if count == total_frames:
            break
    # if numpy.shape(frames)==(49, 144, 256):
    #         all_frames.append(frames)
    return img


data = []
labels = []
all_frames = []
f = open("Annotations.txt", "r")
lines = f.readlines()
for line in lines[1:5]:
    video_name = line[0:6]
    y_labels = line[10:156].split(',')
    y_labels = [int(i[1]) for i in y_labels]
    total_frames = len(y_labels)

    x1 = line[159:].split(',')[2]
    x2 = line[159:].split(',')[3]
    is_ego = line[159:].split(',')[4]
    if "Yes" in is_ego:
        frame_count = 0
        for i in y_labels:
            frame_count += 1
            if i == 1:
                break
        flag = 0
        for i in y_labels:
            if i == 1 and flag == 0:
                break
        for i in range(len(y_labels)):
            if y_labels[i] != 1:
                y_labels[i] = frame_count
                frame_count -= 1
            else:
                break

        labels.extend(y_labels)
        img = load_dataset("Crash1500/" + video_name + ".mp4", total_frames)
        all_frames.extend(img)
# Check if all videos' frame are loaded in np array
print("# of videos loaded", len(all_frames))
print("# of labels", len(labels))

print(np.shape(all_frames))

all_frames = np.asarray(all_frames)

print(labels)

x_train, x_test, y_train, y_test = train_test_split(all_frames, labels, test_size=0.40, random_state=0)

print(np.shape(x_train), np.shape(y_train))

# x_train = x_train.reshape(29, 720, -1)
# x_test = x_test.reshape(20, 720, -1)

print(np.shape(x_train), np.shape(y_train), np.shape(x_test))

frame = 29
row = 720
col = 1280

row_hidden = 128
col_hidden = 128

batch_size = 2
num_classes = 45
epochs = 29

x = Input(shape=(row, col, 3))

encoded_rows = TimeDistributed(LSTM(row_hidden))(x)  ### encodes row of pixels using TimeDistributed Wrapper
encoded_columns = LSTM(col_hidden)(encoded_rows)  ### encodes columns of encoded rows using previous layer

### set up prediction and compile the model
prediction = Dense(num_classes, activation='softmax')(encoded_columns)

model = Model(x, prediction)
model.compile(loss='sparse_categorical_crossentropy',
              ### loss choice for category classification - computes probability error
              optimizer='NAdam',  ### NAdam optimization
              metrics=['accuracy'])

np.random.seed(18247)

model.fit(x_train, y_train,
          batch_size=2,
          epochs=epochs,
          validation_data=(x_test, y_test))

scores = model.evaluate(x_test, y_test, verbose=0)  ### score model
print('Test loss:', scores[0])  ### test loss
print('Test accuracy:', scores[1])  ### test accuracy (ROC later)

model.save("model.h5")
print("Saved model to disk")
