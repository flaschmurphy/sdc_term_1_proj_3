#
# Udacity Self Driving Car Nanodegree
#
# Term 1, Project 3 -- Behavioral Cloning
#
# Author: Ciaran Murphy
# Date: 27th Oct 2017
#
import os
import csv
import cv2
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from argparse import ArgumentParser

# Prevent uninteresting messages from tensorflow when launching
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Global constants
TRAINING_CSV_PATH = '../data/newdata/main'
TRAINING_CSV_FILE = 'driving_log.csv'
TRAINING_IMAGE_PATH = os.path.join(TRAINING_CSV_PATH, 'IMG')
MODEL_SAVE_FILE = './model.h5'
TRAINING_HIST_FILE = './train_hist.pkle'
#

def load_data():
    """Loads data from disk using pandas"""
    colnames = ['center_img', 'left_img', 'right_img', 'steering', 'throttle', 'break', 'speed']
    csvdata = pd.read_csv(os.path.join(TRAINING_CSV_PATH, TRAINING_CSV_FILE), names = colnames)
    remap = lambda n: os.path.join(TRAINING_IMAGE_PATH, os.path.basename(n))
    csvdata.iloc[:,0:3] = csvdata.iloc[:,0:3].applymap(remap)
    return csvdata


def split_data(csvdata):
    """Split data into training and validation sets using sklearn"""
    features = csvdata[:,0:3]
    labels = csvdata[:,3]
    return train_test_split(features, labels)


def data_generator(X, y, batch_size, correction_factor, training=True):
    """Python generator for supplying data to the training process. 

    Note that the length of the data returned will be 6 times the batch_size.
    This is because each line in the batch references center, left and right
    images (=3x) plus a second copy of each of the 3 images, but flipped as
    part of data augmentation.  Therefore if the batch size is 128, the data
    length returned will be 768.  

    """

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



def build_model(drop_prob=0.5, crop_top=60, crop_bottom=25):
    model = Sequential()

    # Normalize the images to the rance -1 to +1
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3), name='normalize'))

    # Crop the image to remove the scenery and hood of the car
    model.add(Cropping2D(cropping=((crop_top,crop_bottom),(0,0)), name='crop'))

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


def explore_images(csvdata, crop_top=60, crop_bottom=25):
    """Creates and displays a plot of images showing the three stages of image processing used."""

    def crop(X, top=crop_top, bottom=crop_bottom):
        """Embeded utility function to perform cropping"""
        height = X.shape[1]
        return X[:, top:height-bottom, :, :]

    # Choose a random number as the index for our sample image
    idx = np.random.randint(csvdata.shape[0]+1)

    # Load a center image and it's corresponding left and right images
    to_rgb = lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_ctr = to_rgb(cv2.imread(csvdata[idx][0]))
    img_lft = to_rgb(cv2.imread(csvdata[idx][1]))
    img_rht = to_rgb(cv2.imread(csvdata[idx][2]))
    images = np.array([img_lft, img_ctr, img_rht])

    # Crop
    images_crop = crop(images)

    plt.figure(figsize=(20, 15))
    
    # Plot 3 original images in 1st row
    plt.subplot(331)
    plt.imshow(images[0])
    plt.title('Original Left')
    
    plt.subplot(332)
    plt.imshow(images[1])
    plt.title('Original Center')
    
    plt.subplot(333)
    plt.imshow(images[2])
    plt.title('Original Right')

    # Plot 3 cropped images in the 2nd row
    plt.subplot(334)
    plt.imshow(images_crop[0])
    plt.title('Cropped Left')
    
    plt.subplot(335)
    plt.imshow(images_crop[1])
    plt.title('Cropped Center')
    
    plt.subplot(336)
    plt.imshow(images_crop[2])
    plt.title('Cropped Right')
    
    # Plot 3 flipped images in the 3rd row
    plt.subplot(337)
    plt.imshow(cv2.flip(images_crop[0], 1))
    plt.title('Fliped Left')
    
    plt.subplot(338)
    plt.imshow(cv2.flip(images_crop[1], 1))
    plt.title('Flipped Center')
    
    plt.subplot(339)
    plt.imshow(cv2.flip(images_crop[2], 1))
    plt.title('Flipped Right')

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_history(history_pickle_file=TRAINING_HIST_FILE):
    """Creates and displays a plot of the training vs validation data history."""

    plt.figure(figsize=(16, 15))
    history = pickle.load(open(history_pickle_file, 'rb'))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Training vs Validation Loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


def plot_steering_angles(csvdata):
    """Creates and displays a histogram of the steering angles observed in the data."""

    plt.figure(figsize=(16, 15))
    plt.hist(csvdata['steering'], bins=100)
    plt.title('Steering angles in training data')
    plt.show()


def main(epochs, batch_size, correction_factor):
    """This is the entry point into the script."""

    args = parse_args()
    if not args.train:
        print("\nNo training will be run. Call with '-t' to including training. See the help for more info.\n")

    print('Loading data...')
    csvdata = load_data()
    print('Summary of data:\n{}\n'.format(csvdata.describe()))

    if not args.train:
        plot_steering_angles(csvdata)

    csvdata = np.array(csvdata)

    X_train, X_valid, y_train, y_valid = split_data(csvdata)
    train_gen = data_generator(X_train, y_train, batch_size, correction_factor)
    valid_gen = data_generator(X_valid, y_valid, batch_size, correction_factor, training=False)

    print('Building model...')
    model = build_model()
    if args.train:
        print('Compiling and training model...')
        model.compile(optimizer='adam', loss='mse')

        # The line below generate a TensorBoard callback that is then fed into
        # the fit() method. The # result of this is that log data is collected
        # during training which can be used to visualize # and if necessary
        # debug the model training behavior. Unfortunately the project
        # environment is # defaulted to Keras version 1.2.1 which is less
        # feature-rich than the # newer v2 releases. Therefore there are some
        # limitations to what # can be accomplished with this version of the
        # callback.
        tboard = TensorBoard(log_dir='./tboard')

        history_object = model.fit_generator(\
                train_gen, \
                samples_per_epoch=batch_size*6, \
                validation_data=valid_gen, \
                nb_val_samples=len(y_train), \
                nb_epoch=epochs, verbose=1, \
                callbacks=[tboard])

        model.save(MODEL_SAVE_FILE)
        print('Model was saved as {}'.format(MODEL_SAVE_FILE))

        with open(TRAINING_HIST_FILE, 'wb') as f:
            pickle.dump(history_object.history, f)
        print('Training history was saved as {}'.format(TRAINING_HIST_FILE))

    else:
        explore_images(csvdata)
        plot_history()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true', default=False, dest='train', \
            help="""Enable to run training. If not set, assume data analytics
            only. If running only analytics, there must be a pickle file in the
            local dir called {} that contains the training history data from
            a previous training run.""".format(TRAINING_HIST_FILE))

    return parser.parse_args()


if __name__ == '__main__':
    # Hyperameters are configured in global scope below
    epochs = 20
    correction_factor= 0.3
    batch_size = 128
    #

    main(epochs, batch_size, correction_factor)

