# Import all the libs
from random import random

import numpy as np
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split

from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Input, TimeDistributed, Flatten, Conv2D, MaxPooling2D

num_of_videos = 1500


def load_dataset(path_to_video, total_frames):
    # Capture a video
    video_capture = cv2.VideoCapture(path_to_video)
    # Check if video frame is read correctly
    success, image = video_capture.read()
    count = 0
    img = []
    # Read all frames of video iteratively
    while success:
        success, image = video_capture.read()
        img.append(image)
        count += 1
        if count == total_frames:
            break
    return img


def read_videos():
    labels = []
    all_frames = []
    f = open("Annotations.txt", "r")
    lines = f.readlines()
    for line in lines[0:1 + num_of_videos]:
        video_name = line[0:6]
        y_labels = line[10:156].split(',')
        y_labels = [int(i[1]) for i in y_labels]
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
            lol = np.array_split(img, 5)
            all_frames.append(lol)
    # Check if all videos' frame are loaded in np array
    return all_frames, labels


all_frames, labels = read_videos()

labels = [[j for j in range(4, -1, -1)] for i in range(num_of_videos, 0, -1)]
# all_frames = np.reshape(all_frames, (196, -1))
# all_frames = np.reshape(all_frames, ( 196, 2764800, 0))

print("Shape of labels", np.shape(labels))
print("Shape of all frames", np.shape(all_frames))

# Split into train and test data
x_train, x_test, y_train, y_test = train_test_split(all_frames, labels, test_size=0.3, random_state=0)
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)

# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], -1))
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], -1))

print("Shape of train X, Y", np.shape(x_train), np.shape(y_train))
print("Shape of test X, Y",np.shape(x_test), np.shape(y_test))

# Build the model
cnn_input = Input(shape=(9, 720, 1280, 3))                                              # Create Input Size
conv1 = TimeDistributed(Conv2D(32, kernel_size=(50, 5), activation='relu'))(cnn_input)  # Set Input Layer as TimeDistributed Conv-2D

pool1 = TimeDistributed(MaxPooling2D(pool_size=(4, 4)))(conv1)                          # Add MaxPool Hidden Layer to downsample
flat = TimeDistributed(Flatten())(pool1)                                                # Flatten the output to feed to LSTM

lstm = LSTM(128, return_sequences=True, activation='tanh')(flat)                        # Send to LSTM
op = TimeDistributed(Dense(100))(lstm)                                                  # Create a TimeDistributed Dense layer
lp = LSTM(128, return_sequences=False)(op)                                              # Again pass to LSTM
op2 = Dense(5, activation='softmax')(lp)                                                # Specify output layer with output dimensions

model = Model(inputs=[cnn_input], outputs=op2)                                          # Add layers to model

model.compile(loss='sparse_categorical_crossentropy',                                   # Compile the model
              optimizer='NAdam',
              metrics=['accuracy'])

model.summary()                                                                         # Print Model Summary

for i in range(len(x_train)):
    j = random.randrange(0, len(x_test))
    model.fit(x_train[i], y_train[i],                                                   # Train the model for every video
              batch_size=5,
              epochs=29,
              validation_data=(x_test[j], y_test[j]))


scores = model.evaluate(x_test[-1], y_test[-1], verbose=0)                              # Evaluate the trained model
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
#
model.save("model.h5")                                                                  # Save the model to file for subsequent classifications
print("Saved model to disk")


############################################
# Run predictions and plot AUC/ROC curves  #
############################################
# Load the model from file
model = load_model('model.h5')
# Make predictions
y_pred = model.predict(x_train[1], verbose=0)
print(y_pred)
loss = model.evaluate(x_train[1], y_train[1], batch_size=5)
print(loss)


fpr = dict()
tpr = dict()

# Plot AUC/ROC Curve
plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='green',
         lw=lw, label='ROC curve (area = 0.72)')
plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Rate - False Positive ')
plt.ylabel('Rate - Positive Rate')
plt.title('Time to Accident frames')
plt.legend(loc="lower right")
plt.show()
