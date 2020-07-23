""" Code file to create a simple solution using keras"""

import numpy as np
import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd

from typing import List, Tuple
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import Model
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import log_loss


def get_im(path: str) -> np.ndarray:
    """ Load an image in greyscale and resize to (129,96)
        Args:
            path: Path where image is stored
        Returns:
            Image object as a numpy array
    """

    # Load as grayscale
    img = cv2.imread(path, 0)
    # Reduce size
    resized = cv2.resize(img, (128, 96))

    return resized


def load_train() -> Tuple[List[np.ndarray], List[int]]:
    """ Loads training data images into specified folder as file glob
        Returns:
            Tuple of resized images and image type
    """

    x_train: List[np.ndarray] = []
    y_train = []
    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join(os.path.dirname(__file__), '..', 'input', 'imgs', 'train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            img = get_im(fl)
            x_train.append(img)
            y_train.append(j)

    return x_train, y_train


def load_test() -> Tuple[List[np.ndarray], List[str]]:
    """ Loads test data images into specified folder as file glob
        Returns:
            Tuple of resized images and image name
    """

    print('Read test images')
    path = os.path.join(os.path.dirname(__file__), '..', 'input', 'imgs', 'test', '*.jpg')
    files = glob.glob(path)
    x_test = []
    x_test_id = []
    total = 0
    thr = math.floor(len(files) / 10)
    for fl in files:
        fl_base = os.path.basename(fl)
        img = get_im(fl)
        x_test.append(img)
        x_test_id.append(fl_base)
        total += 1
        if total % thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    return x_test, x_test_id


def cache_data(data: Tuple[List, List], path: str) -> None:
    """ Dumps data as a file in the given path
        Args:
            data: Tuple of images and image type or name
            path: Path where cached data is stored"""

    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')


def restore_data(path: str) -> Tuple[List, List]:
    """ Reads cached data from the given path
        Args:
            path: Path where cached data is stored
        Returns:
            returns data as a tuple of images and image type or name
    """
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data


def save_model(model: Model) -> None:
    """ Saves model as JSON file in the specified path
        Args:
            model: Trained keras model to be saved
    """
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    open(os.path.join('cache', 'architecture.json'), 'w').write(json_string)
    model.save_weights(os.path.join('cache', 'model_weights.h5'), overwrite=True)


def read_model() -> Model:
    """ Reads model from specified path, loads weights and return the model
        Returns:
            Returns a keras model with pre trained weights
    """
    model = model_from_json(open(os.path.join('cache', 'architecture.json')).read())
    model.load_weights(os.path.join('cache', 'model_weights.h5'))
    return model


def split_validation_set_with_hold_out(train: np.ndarray, target: np.ndarray, test_size: float)\
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Splits training data into training, validation and holdout data sets
        Args:
            train: Array of training data features
            target: Array of training data target
            test_size: Split ratio of validation and holdout data sets
        Returns:
            Training, Validation and Holdout features and targets as arrays
    """
    random_state = 51
    train, x_test, target, y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)
    x_train, x_holdout, y_train, y_holdout = train_test_split(train, target, test_size=test_size,
                                                              random_state=random_state)
    return x_train, x_test, x_holdout, y_train, y_test, y_holdout


def create_submission(predictions: np.ndarray, test_id: List[str], loss: float) -> None:
    """ Creates and stores a submission file from model predictions
        Args:
            predictions: Array of model predictions on test data
            test_id: List of test file names
            loss: Model loss used to provide a specific name to the file for future reference
    """
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = str(round(loss, 6)) + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)


def validate_holdout(model: Model, holdout: np.ndarray, target: np.ndarray) -> float:
    """ Computes log-loss score of holdout data set
        Args:
            model: Trained keras model
            holdout: Holdout data features as array
            target: Holdout targets as array
        Returns:
            Log-loss score of holdout data set
    """
    predictions = model.predict(holdout, batch_size=128, verbose=1)
    score = log_loss(target, predictions)
    print('Score log_loss: ', score)

    return score


def get_simple_keras_solution() -> None:
    """ Function to derive a simple solution from keras
    """

    np.random.seed(2016)

    # Load training data
    cache_path = os.path.join('cache', 'train.dat')
    if not os.path.isfile(cache_path):
        train_data, train_target = load_train()
        cache_data((train_data, train_target), cache_path)
    else:
        print('Restore train from cache!')
        (train_data, train_target) = restore_data(cache_path)

    batch_size = 64
    nb_classes = 10
    nb_epoch = 2
    # input image dimensions
    img_rows, img_cols = 96, 128
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3

    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)
    train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
    train_target = np_utils.to_categorical(train_target, nb_classes)
    train_data = train_data.astype('float32')
    train_data /= 255
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')

    x_train, x_test, x_holdout, y_train, y_test, y_holdout = \
        split_validation_set_with_hold_out(train_data, train_target, 0.2)

    print('Split train: ', len(x_train))
    print('Split valid: ', len(x_test))
    print('Split holdout: ', len(x_holdout))

    model_from_cache = 0
    if model_from_cache == 1:
        model = read_model()
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    else:
        model = Sequential()
        model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                                border_mode='valid',
                                input_shape=(1, img_rows, img_cols)))
        model.add(Activation('relu'))
        model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        '''
        model.fit(train_data, train_target, batch_size=batch_size, nb_epoch=nb_epoch,
                  show_accuracy=True, verbose=1, validation_split=0.1)
        '''
        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                  show_accuracy=True, verbose=1, validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Score: ', score)
    score = model.evaluate(x_holdout, y_holdout, verbose=0)
    print('Score holdout: ', score)
    validate_holdout(model, x_holdout, y_holdout)
    save_model(model)

    cache_path = os.path.join('cache', 'test.dat')
    if not os.path.isfile(cache_path):
        test_data, test_id = load_test()
        cache_data((test_data, test_id), cache_path)
    else:
        print('Restore test from cache!')
        (test_data, test_id) = restore_data(cache_path)

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols)
    test_data = test_data.astype('float32')
    test_data /= 255
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    predictions = model.predict(test_data, batch_size=128, verbose=1)

    create_submission(predictions, test_id, score)