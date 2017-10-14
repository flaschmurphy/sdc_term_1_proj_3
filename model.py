import os
import csv
import cv2
import numpy as np
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D


TRAINING_CSV_PATH = '../data/my_training_data'
TRAINING_CSV_FILE = 'driving_log.csv'
TRAINING_IMAGE_PATH = os.path.join(TRAINING_CSV_PATH, 'IMG')
MODEL_SAVE_FILE = './model.h5'


def load_data():
    """Load data from disk. While loading, convert paths in the csv text to local paths. There are three
    images in the data: left, right and center. The center image is the main one, but left and right can 
    be used to train the model to correct itself. If the model sees an image that looks like a left image
    then encourage the model to steer to the right (the simplest way is to specify a constant amount) and
    similarly if the model sees an image that looks like a right image, steer to the left.

    Returns:
        csvdata: list of the raw csv data with file paths maped to local dirs instead of source dirs
        images: list of images loaded using cv2.imread()
        measurements: list of steering measurements for each image
    """

    csvdata, images, measurements = [], [], []
    remap = lambda n: os.path.join(TRAINING_IMAGE_PATH, os.path.basename(line[n]))
    correction_factor = 0.35

    with open(os.path.join(TRAINING_CSV_PATH, TRAINING_CSV_FILE)) as f:
        reader = csv.reader(f)
        for line in reader:
            # Translate the paths in the csv data to local paths
            line[0] = remap(0)
            line[1] = remap(1)
            line[2] = remap(2)
            csvdata.append(line)

            # Load the images and measurements
            image_c = cv2.imread(line[0])            # center image
            images.append(image_c)
            measurements.append(float(line[3]))      # center steering measurement

            image_l = cv2.imread(line[1])            # left image
            images.append(image_l)
            measurements.append(correction_factor)   # left steering measurement

            image_r = cv2.imread(line[2])            # left image
            images.append(image_r)
            measurements.append(-correction_factor)  # right steering measurement
            
            # Add more data by augmenting the images. By flipping the image and 
            # steering angle we can compensate for the fact that the training data
            # was taken by driving around the track clockwise and train the model
            # to handle right turns
            images.append(cv2.flip(image_c, 1))
            measurements.append(float(line[3]))

            images.append(cv2.flip(image_l, 1))
            measurements.append(-correction_factor)

            images.append(cv2.flip(image_r, 1))
            measurements.append(correction_factor)

    return csvdata, images, measurements


def build_model(X_train, y_train):
    model = Sequential()

    # Crop the image to remove the scenery and hood of the car
#    model.add(Lambda(lambda x: x[70:140,:,:], input_shape=X_train.shape[1:]))

    # Normalize the images to the rance -1 to +1
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=X_train.shape[1:]))

    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model


def main():
    print('Loading data...')
    csvdata, images, measurements = load_data()
    X_train = np.array(images)
    y_train = np.array(measurements)

    print('Done.')
    print('X shape: {}'.format(X_train.shape))
    print('y shape: {}'.format(y_train.shape))
    print('y summary: {}'.format(pd.Series(y_train).describe()))

    print('Building & training model...')
    model = build_model(X_train, y_train)

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, nb_epoch=3, validation_split=0.2, shuffle=True)

    model.save(MODEL_SAVE_FILE)


if __name__ == '__main__':
    d = main()




