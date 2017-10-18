import os
import csv
import cv2
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, ZeroPadding2D, Convolution2D, Cropping2D, Dropout
#from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


TRAINING_CSV_PATH = '../data/my_training_data'
TRAINING_CSV_FILE = 'driving_log.csv'
TRAINING_IMAGE_PATH = os.path.join(TRAINING_CSV_PATH, 'IMG')
MODEL_SAVE_FILE = './model.h5'
TRAINING_HIST_FILE = './model_training_hist.pkle'


def load_data():
    csvdata = pd.read_csv(os.path.join(TRAINING_CSV_PATH, TRAINING_CSV_FILE), \
            names=['center_image', 'left_image', 'right_image', 'steering', 'throttle', 'break',    'speed'])
    print('\nSummary of raw csv data:\n{}'.format(csvdata.describe()))

    remap = lambda n: os.path.join(TRAINING_IMAGE_PATH, os.path.basename(n))

    csvdata['center_image'] = csvdata['center_image'].apply(remap)
    csvdata['left_image'] = csvdata['left_image'].apply(remap)
    csvdata['right_image'] = csvdata['right_image'].apply(remap)
    print('\nFirst lines of the remaped center image filenames:\n{}'.format(csvdata['center_image'].head()))

    return np.array(csvdata)


def split_data(csvdata):
    features = csvdata[:,0:3]
    labels = csvdata[:,3]
    return train_test_split(features, labels)


def randomize_brightness(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_brightness = .1 + np.random.uniform()
    image[:,:,2] = image[:,:,2] * random_brightness
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image


def data_generator(X, y, batch_size, correction_factor, training=True):
    #
    # There are three images in the data: left, right and center. The center
    # image is the main one, but left and right can be used to train the model to
    # correct itself. If the model sees an image that looks like a left image
    # then encourage the model to steer to the right (the simplest way is to
    # specify a constant amount) and similarly if the model sees an image that
    # looks like a right image, steer to the left.
    # 
    # Note that the length of the data returned will actually be 6x the batch_size
    # since it includes center, left and right images (=3x) plus a second copy of 
    # each of the 3 images, but flipped (data augmentation). Therefore if the batch 
    # size is 128, the data length returned will be 768. If new images are added with 
    # randomized brightness then the output will be even bigger.
    #

    assert len(X) == len(y), "Required to have the same number of features and labels"

    num_samples = len(X)

    while True:
        X, y = shuffle(X, y)
        for offset in range(0, num_samples, batch_size):

            X_batch = X[offset:offset+batch_size]
            y_batch = y[offset:offset+batch_size]

            images, measurements = [], []

            # To resize images here instead of in the model, do this: resized = cv2.resize(image, (80, 60))
            for idx in range(len(X_batch)):
                image_c = cv2.imread(X_batch[idx][0])            # center image
                images.append(image_c)
                measurements.append(float(y_batch[idx]))         # center steering measurement

                image_l = cv2.imread(X_batch[idx][1])            # left image
                images.append(image_l)
                measurements.append(correction_factor)           # left steering measurement

                image_r = cv2.imread(X_batch[idx][2])            # left image
                images.append(image_r)
                measurements.append(-correction_factor)          # right steering measurement

                # Add more data by augmenting the images. By flipping the image and 
                # steering angle we can compensate for the fact that the training data
                # was taken by driving around the track clockwise and train the model
                # to handle right turns. If this is the validation generator however
                # then don't do data augmentation
                if training:
                    images.append(cv2.flip(image_c, 1))
                    measurements.append(float(y_batch[idx]))
                    images.append(cv2.flip(image_l, 1))
                    measurements.append(-correction_factor)
                    images.append(cv2.flip(image_r, 1))
                    measurements.append(correction_factor)

            yield shuffle(np.array(images), np.array(measurements))


def build_model():
    model = Sequential()

    # Crop the image to remove the scenery and hood of the car
    model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160, 320, 3)))

    # Normalize each image
    model.add(Lambda(lambda n: n/255.0 - 0.5))

    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(1,1), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(1,1), activation='relu'))
    #model.add(Convolution2D(64, 3, 3, activation='relu'))
    #model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    print('Model summary:\n{}'.format(model.summary()))

    return model


def main(epochs, batch_size, correction_factor):
    print('Loading data...')
    csvdata = load_data()
    X_train, X_valid, y_train, y_valid = split_data(csvdata)
    train_gen = data_generator(X_train, y_train, batch_size, correction_factor)
    valid_gen = data_generator(X_valid, y_valid, batch_size, correction_factor, training=False)

    print('Building & training model...')
    model = build_model()

    model.compile(optimizer='adam', loss='mse')

    history_object = model.fit_generator(\
            train_gen, \
            samples_per_epoch=batch_size*6, \
            validation_data=valid_gen, \
            nb_val_samples=len(y_train), \
            nb_epoch=epochs, verbose=1)

    model.save(MODEL_SAVE_FILE)
    print('Model was saved as {}'.format(MODEL_SAVE_FILE))

    with open(TRAINING_HIST_FILE, 'wb') as f:
        pickle.dump(history_object.history, f)
    print('Training history was saved as {}'.format(TRAINING_HIST_FILE))


def plot_hist(history_pickle_file=TRAINING_HIST_FILE):
    """plot the training and validation loss for each epoch"""
    history = pickle.load(open(history_pickle_file, 'rb'))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


if __name__ == '__main__':
    epochs = 4 
    correction_factor= 0.35
    batch_size = 64
    main(epochs, batch_size, correction_factor)




