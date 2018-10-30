import csv
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
	for i in range(3):
		source_path = line[0]
		filename = source_path.split('/')[-1]
		current_path = "./data/IMG/" + filename.split('\\')[-1]
		image = cv2.imread(current_path)
		images.append(image)
		measurement = float(line[3])
		measurements.append(measurement)

augmented_images = []
augmented_measurements =[]
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	flipped_image = cv2.flip(image, 1)
	flipped_measurement = float(measurement)* -1.0
	augmented_images.append(flipped_image)
	augmented_measurements.append(flipped_measurement)

X_train = np.array(images)
y_train = np.array(measurements)

import keras
from keras.models import Sequential
from keras.layers.core import Activation, Dropout
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()

# Preprocess
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

# trim image
model.add(Cropping2D(cropping=((70,25),(0,0))))

#layer 1- Convolution
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))

#layer 2- Convolution
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))

#layer 3- Convolution
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))

#layer 4- Convolution
model.add(Convolution2D(64,3,3,activation='relu'))

#layer 5- Convolution
model.add(Convolution2D(64,3,3,activation='relu'))

#Adding a dropout layer to avoid overfitting
model.add(Dropout(0.3))

#flatten image
model.add(Flatten())

#layer 6- fully connected layer
model.add(Dense(100))
model.add(Activation('relu'))

#Adding a dropout layer to avoid overfitting
model.add(Dropout(0.3))

#layer 7- fully connected layer
model.add(Dense(50))
model.add(Activation('relu'))

#Adding a dropout layer to avoid overfitting
model.add(Dropout(0.3))

#layer 8- fully connected layer
model.add(Dense(10))
model.add(Activation('relu'))

#layer 9- fully connected layer
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

#saving model
model.save('model.h5')

model.summary()
