""" Code file to create functions used throughout the module """

import pickle
import numpy as np
import os
import glob
import math
import cv2
import datetime
import pandas as pd
from typing import Any, Dict, List, Tuple, Iterable

from sklearn.model_selection import train_test_split
from keras import Model
from keras.models import model_from_json


# color_type = 1 - gray
def get_im(path: str, img_rows: int, img_cols: int, color_type: int = 1) -> np.ndarray:
    """ Load an image in greyscale and resize to (129,96)
        Args:
            path: Path where image is stored
            img_rows: Row pixel dimension of images
            img_cols: Column pixel dimension of images
            color_type: 1 indicates gray, else RGB
        Returns:
            Image object as a numpy array
    """
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    else:
        img = cv2.imread(path)
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))

    return resized


def get_driver_data() -> Dict[str, str]:
    """ Returns a dictionary of image name mapped to subject"""
    dr = dict()
    path = os.path.join('..', 'input', 'driver_imgs_list.csv')
    print('Read drivers data')
    f = open(path, 'r')
    f.readline()
    while True:
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        dr[arr[2]] = arr[0]
    f.close()

    return dr


def load_train(img_rows: int, img_cols: int, color_type=1) -> \
        Tuple[List[np.ndarray], List[int], List[str], List[str]]:
    """ Loads training data images into specified folder as file glob
        Args:
            img_rows: Row pixel dimension of images
            img_cols: Column pixel dimension of images
            color_type: 1 indicates gray, else RGB
        Returns:
            Tuple of resized images and image type
    """
    x_train: List[np.ndarray] = []
    y_train = []
    driver_id = []

    driver_data = get_driver_data()

    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join(os.path.dirname(__file__), '..', 'input', 'imgs', 'train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            fl_base = os.path.basename(fl)
            img = get_im(fl, img_rows, img_cols, color_type)
            x_train.append(img)
            y_train.append(j)
            driver_id.append(driver_data[fl_base])

    unique_drivers = sorted(list(set(driver_id)))
    print('Unique drivers: {}'.format(len(unique_drivers)))
    print(unique_drivers)

    return x_train, y_train, driver_id, unique_drivers


def load_test(img_rows: int, img_cols: int, color_type=1) -> Tuple[List[np.ndarray], List[str]]:
    """ Loads test data images into specified folder as file glob
        Args:
            img_rows: Row pixel dimension of images
            img_cols: Column pixel dimension of images
            color_type: 1 indicates gray, else RGB
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
        img = get_im(fl, img_rows, img_cols, color_type)
        x_test.append(img)
        x_test_id.append(fl_base)
        total += 1
        if total % thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    return x_test, x_test_id


def cache_data(data: Any, path: str) -> None:
    """ Dumps data as a file in the given path
        Args:
            data: Tuple of images and image type or name
            path: Path where cached data is stored
    """
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exist')


def restore_data(path: str) -> Any:
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


def split_validation_set(train: np.ndarray, target: np.ndarray, test_size: float) \
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


def split_validation_set_with_hold_out(train: np.ndarray, target: np.ndarray, test_size: float) \
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


def create_submission(predictions: Iterable, test_id: List[str], info: Any) -> None:
    """ Creates and stores a submission file from model predictions
        Args:
            predictions: Array of model predictions on test data
            test_id: List of test file names
            info: Model loss or other info used to provide a specific name to the file
                  for future reference
    """
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    if isinstance(info, float):
        suffix = str(round(info, 6)) + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    else:
        suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)


def dict_to_list(d: Dict) -> List:
    """ Converts a dictionary to a list
        Args:
            d: Input dictionary
        Returns:
            Outputs list
    """
    ret = []
    for i in d.items():
        ret.append(i[1])

    return ret


def merge_several_folds_fast(data: List[float], n_folds: int) -> Iterable:
    """ Function to merge different cross-validation fold predictions
        Args:
            data: Predictions from different cross-validation folds
            n_folds: Number of cross-validation folds
        Returns:
            List of predictions from different cross-validation folds
    """
    a = np.array(data[0])
    for i in range(1, n_folds):
        a += np.array(data[i])
    a /= n_folds
    return a.tolist()


def merge_several_folds_geom(data: List[float], n_folds: int) -> Iterable:
    """ Function to merge different cross-validation fold predictions using a geometric mean
        Args:
            data: Predictions from different cross-validation folds
            n_folds: Number of cross-validation folds
        Returns:
            List of predictions from different cross-validation folds
    """
    a = np.array(data[0])
    for i in range(1, n_folds):
        a *= np.array(data[i])
    a = np.power(a, 1 / n_folds)

    return a.tolist()
