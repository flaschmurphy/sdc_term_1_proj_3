import os
import csv
import cv2
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from argparse import ArgumentParser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Reshape
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.utils import plot_model
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


def to_gray(X):
    return  X[:, :, :, 0] * 0.299 + X[:, :, :, 1] * 0.587 + X[:, :, :, 2] * 0.114

def crop(X, top=60, bottom=25):
    if type(X) == np.ndarray:
        height = X.shape[1]
    else:
        height = X.get_shape()[1]
    return X[:, top:height-bottom, :]

def preprocess(X):
    X = to_gray(X)
    X = crop(X)
    return X

def build_model(drop_prob):
    model = Sequential()

    # Normalize the images to the rance -1 to +1
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))

    model.add(Lambda(preprocess))

    model.add(Reshape((75, 320, 1)))
 
    model.add(Convolution2D(24, (5, 5), strides=(2,2), activation='relu'))
    model.add(Dropout(drop_prob))
    model.add(Convolution2D(36, (5, 5), strides=(2,2), activation='relu'))
    model.add(Convolution2D(48, (5, 5), strides=(2,2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
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


def explore_images(csvdata):
    # Choose a random number as the index for our sample image
    idx = np.random.randint(csvdata.shape[0]+1)

    # Load a center image and it's corresponding left and right images
    to_rgb = lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_ctr = to_rgb(cv2.imread(csvdata[idx][0]))
    img_lft = to_rgb(cv2.imread(csvdata[idx][1]))
    img_rht = to_rgb(cv2.imread(csvdata[idx][2]))
    images = np.array([img_lft, img_ctr, img_rht])
    
    # Convert to grayscale
    images_gray = to_gray(images)
    
    # Crop
    images_crop = crop(images_gray)
    
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
    
    # Plot 3 gray images in the 3nd row
    plt.subplot(334)
    plt.imshow(images_gray[0], cmap='gray')
    plt.title('Gray Left')
    
    plt.subplot(335)
    plt.imshow(images_gray[1], cmap='gray')
    plt.title('Gray Center')
    
    plt.subplot(336)
    plt.imshow(images_gray[2], cmap='gray')
    plt.title('Gray Right')

    # Plot 3 cropped images in the 2nd row
    plt.subplot(337)
    plt.imshow(images_crop[0], cmap='gray')
    plt.title('Cropped Left')
    
    plt.subplot(338)
    plt.imshow(images_crop[1], cmap='gray')
    plt.title('Cropped Center')
    
    plt.subplot(339)
    plt.imshow(images_crop[2], cmap='gray')
    plt.title('Cropped Right')
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    


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
    print('Plotting model to png file...')
    plot_model(model, to_file='model.png', show_shapes=True)

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
        explore_images(csvdata)
        plot_history()



if __name__ == '__main__':
    epochs = 25 
    correction_factor= 0.425
    batch_size = 128
    drop_prob = 0.25

    main(epochs, batch_size, correction_factor, drop_prob)

