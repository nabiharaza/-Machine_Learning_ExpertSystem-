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
from keras.layers import Dense, LSTM, Input, TimeDistributed, ConvLSTM2D, BatchNormalization, Conv3D, Flatten, Conv2D, \
    MaxPooling2D, Reshape
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


def read_videos():
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
            img = load_dataset("Crash1500/" + video_name + ".mp4", 45)
            img = np.asarray(img)
            # print(np.shape(img))
            lol = np.array_split(img, 5)
            # print("L0L", np.shape(lol))

            all_frames.append(lol)
    # Check if all videos' frame are loaded in np array
    print("# of videos loaded", len(all_frames))
    # print("# of labels", len(labels))
    return all_frames, labels


all_frames, labels = read_videos()

labels = [
    [4, 3, 2, 1, 0],
    [4, 3, 2, 1, 0],
    [4, 3, 2, 1, 0],
    [4, 3, 2, 1, 0],
]
# all_frames = np.reshape(all_frames, (196, -1))
# all_frames = np.reshape(all_frames, ( 196, 2764800, 0))

print("Shape of labels", np.shape(labels))
print("Shape of all frames", np.shape(all_frames))
# Split into train and test data
x_train, x_test, y_train, y_test = train_test_split(all_frames, labels, test_size=0.5, random_state=0)
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)

# x_train = np.moveaxis(x_train, 2, 0)
# x_test = np.moveaxis(x_test, 2, 0)
# x_train = np.asarray([x_train])
# x_test = np.asarray([x_test])


# x_train = x_train.reshape((-1, 34, 720, 1280, 3))
# x_test = x_test.reshape((-1, 15, 720, 1280, 3))


# x_train = np.reshape(x_train, (5, 49, -1))
# x_test = np.reshape(x_test, (5, 49, -1))

# seq3 = np.zeros((1, 34))
# for i in range(1):
#     seq3[i] = y_train
#
# seq4 = np.zeros((1, 15))
# for i in range(1):
#     seq4[i] = y_test
#
# x_train = np.reshape(x_train, (1, x_train.shape[0], x_train.shape[1]))
# x_test = np.reshape(x_test, (1, x_test.shape[0], x_test.shape[1]))

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], -1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], -1))
print(np.shape(x_train), np.shape(y_train))

row_hidden = 128
col_hidden = 128

batch_size = 2
num_classes = 45
epochs = 5

# cnn_input = Input(shape=(34, 720, 1280, 3))
# conv1 = TimeDistributed(Conv2D(32, kernel_size=(50, 5), activation='relu'))(cnn_input)
# print(1, np.shape(conv1))
# conv2 = TimeDistributed(Conv2D(32, kernel_size=(20, 5), activation='relu'))(conv1)
# print(2, np.shape(conv2))
# pool1 = TimeDistributed(MaxPooling2D(pool_size=(4, 4)))(conv2)
# print(3, np.shape(pool1))
# flat = TimeDistributed(Flatten())(pool1)
# print(4, np.shape(flat))
# cnn_op = TimeDistributed(Dense(100))(flat)
# print(5, np.shape(cnn_op))
#
# lstm = LSTM(128, return_sequences=True, activation='tanh')(cnn_op)
# print(6, np.shape(lstm))
# op = TimeDistributed(Dense(100))(lstm)
# print(7, np.shape(op))
# op2 = Dense(1)(op)
# print(8, np.shape(op2))
# reshape = Reshape((34,))(op2)
# print(9, np.shape(reshape))
# model = Model(inputs=[cnn_input], outputs=reshape)
#######
visible = Input(shape=(9, 720 * 1280 * 3))

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(9, 720 * 1280 * 3)))
model.add(Dense(5))

# encoded_rows = LSTM(row_hidden, input_shape=(1, 34, 720*1280*3))
# encoded_columns = Dense(col_hidden)(encoded_rows)
#
# prediction = Dense(34, activation='softmax')(encoded_columns)
# reshape = Reshape((34,))(prediction)
# print(9999, np.shape(reshape))

# model = Model(visible, reshape)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='NAdam',
              metrics=['accuracy'])
#
# np.random.seed(18247)
#
for i in range(len(x_train)):
    model.fit(x_train[i], y_train[i],
              batch_size=5,
              epochs=epochs,
              validation_data=(x_test[i], y_test[i]))
#
scores = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
#
model.save("model.h5")
print("Saved model to disk")
