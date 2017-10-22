import os
import csv
import cv2
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


TRAINING_CSV_PATH = '../data/my_training_data'
TRAINING_CSV_FILE = 'driving_log.csv'
TRAINING_IMAGE_PATH = os.path.join(TRAINING_CSV_PATH, 'IMG')
MODEL_SAVE_FILE = './model.h5'
TRAINING_HIST_FILE = './model_training_hist.pkle'


def load_data():
    colnames = ['center_img', 'left_img', 'right_img', 'steering', 'throttle', 'break', 'speed']
    csvdata = pd.read_csv(os.path.join(TRAINING_CSV_PATH, TRAINING_CSV_FILE), names = colnames)
    remap = lambda n: os.path.join(TRAINING_IMAGE_PATH, os.path.basename(n))
    csvdata.iloc[:,0:3] = csvdata.iloc[:,0:3].applymap(remap)
    return csvdata


def explore_csv(csvdata):
    print()
    print('Summary of data:\n{}'.format(csvdata.describe()))
    print("""
  ==> The steering angles are normalized in the range -1 to +1
  ==> Most steering angles are less than zero, meaning the data is biased
      to left turns. This is expected given that the track is known to be
      dominated by left turns
  ==> The steering data is quite clustered around zero, see the 25% and 50% 
      quartiles
  """)
    print()


def split_data(csvdata):
    features = csvdata[:,0:3]
    labels = csvdata[:,3]
    return train_test_split(features, labels)


def data_generator(X, y, batch_size, correction_factor, training=True):

    # Note that the length of the data returned will actually be 6x the batch_size
    # since it includes center, left and right images (=3x) plus a second copy of 
    # each of the 3 images, but flipped (data augmentation). Therefore if the batch 
    # size is 128, the data length returned will be 768.

    assert len(X) == len(y), "Required to have the same number of features and labels"

    num_samples = len(X)

    while True:
        X, y = shuffle(X, y)
        for offset in range(0, num_samples, batch_size):

            X_batch = X[offset:offset+batch_size]
            y_batch = y[offset:offset+batch_size]

            images, measurements = [], []

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
                # to handle right turns
                if training:
                    images.append(cv2.flip(image_c, 1))
                    measurements.append(float(y_batch[idx]))
                    images.append(cv2.flip(image_l, 1))
                    measurements.append(-correction_factor)
                    images.append(cv2.flip(image_r, 1))
                    measurements.append(correction_factor)

            yield shuffle(np.array(images), np.array(measurements))


def build_model(drop_prob):
    model = Sequential()

    # Normalize the images to the rance -1 to +1
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
    # Crop the image to remove the scenery and hood of the car
    model.add(Cropping2D(cropping=((70,25),(0,0))))

    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Dropout(drop_prob))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    print()
    print(model.summary())
    print()

    return model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-l', '--local', action='store_true', default=False, dest='local', \
            help='Enable if running traing on a local machine to see plots of eg. images and training loss')

    parser.add_argument('-a', '--analyze_only', action='store_true', default=False, dest='analyze_only', \
            help="""Enable to skip training and only perform the data analytics part. Expects to find the training 
            history data locally in a pickle file called {} """.format(TRAINING_HIST_FILE))

    return parser.parse_args()


def main(epochs, batch_size, correction_factor, drop_prob):
    args = parse_args()

    print('Loading data...')
    csvdata = load_data()
    explore_csv(csvdata)

    csvdata = np.array(csvdata)

    X_train, X_valid, y_train, y_valid = split_data(csvdata)
    train_gen = data_generator(X_train, y_train, batch_size, correction_factor)
    valid_gen = data_generator(X_valid, y_valid, batch_size, correction_factor, training=False)

    print('Building model...')
    model = build_model(drop_prob)

    if not args.analyze_only:
        print('Compiling and training model...')
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

    if args.local:
        plot_history()


def plot_history(history_pickle_file=TRAINING_HIST_FILE):
    """Plot the training and validation loss for each epoch"""
    history = pickle.load(open(history_pickle_file, 'rb'))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Training vs Validation Loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


if __name__ == '__main__':
    epochs = 25 
    correction_factor= 0.425
    batch_size = 128
    drop_prob = 0.25

    main(epochs, batch_size, correction_factor, drop_prob)

