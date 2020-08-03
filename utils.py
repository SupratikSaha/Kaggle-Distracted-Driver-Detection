""" Code file to create functions used throughout the module """

import pickle
import numpy as np
import os
import glob
import math
import cv2
import datetime
import pandas as pd
from typing import Dict, List, Tuple, Iterable

from sklearn.model_selection import train_test_split
from keras import Model
from keras.models import model_from_json


def get_im(path: str, img_rows: int, img_cols: int) -> np.ndarray:
    """ Load an image in greyscale and resize to (129,96)
        Args:
            path: Path where image is stored
            img_rows: Row pixel dimension of images
            img_cols: Column pixel dimension of images
        Returns:
            Image object as a numpy array
    """

    # Load as grayscale
    img = cv2.imread(path, 0)
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))

    return resized


def load_train(img_rows: int, img_cols: int) -> Tuple[List[np.ndarray], List[int]]:
    """ Loads training data images into specified folder as file glob
        Args:
            img_rows: Row pixel dimension of images
            img_cols: Column pixel dimension of images
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
            img = get_im(fl, img_rows, img_cols)
            x_train.append(img)
            y_train.append(j)

    return x_train, y_train


def load_test(img_rows: int, img_cols: int) -> Tuple[List[np.ndarray], List[str]]:
    """ Loads test data images into specified folder as file glob
        Args:
            img_rows: Row pixel dimension of images
            img_cols: Column pixel dimension of images
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
        img = get_im(fl, img_rows, img_cols)
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
        print('Directory doesnt exist')


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


def split_validation_set(train: np.ndarray, target: np.ndarray, test_size: float)\
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Splits training data into training, validation and holdout data sets
        Args:
            train: Array of training data features
            target: Array of training data target
            test_size: Split ratio of validation and holdout data sets
        Returns:
            Training and Validation features and targets as arrays
    """
    random_state = 51
    x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test


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


def create_submission(predictions: Iterable, test_id: List[str], loss: float) -> None:
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


def dict_to_list(d: Dict) -> List:
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret
