""" Code file to create functions used throughout the module """

import pickle
import numpy as np
import os
import glob
import math
import random
import time
import cv2
import datetime
import pandas as pd
from typing import Any, Dict, List, Tuple, Iterable
from shutil import copy2
from sklearn.model_selection import train_test_split
from keras import Model
from keras.models import model_from_json, Sequential


# color_type = 1 - gray
def get_im(path: str, img_rows: int, img_cols: int, color_type: int = 1, mode: str = 'norm') -> np.ndarray:
    """ Load an image in greyscale and resize to (129,96)
        Args:
            path: Path where image is stored
            img_rows: Row pixel dimension of images
            img_cols: Column pixel dimension of images
            color_type: 1 indicates gray, else RGB
            mode: Mode of reading image
        Returns:
            Image object as a numpy array
    """
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    else:
        img = cv2.imread(path)
    # Reduce size
    resized = ''
    if mode == 'norm':
        resized = cv2.resize(img, (img_cols, img_rows))
    elif mode == 'cv2':
        resized = cv2.resize(img, (img_cols, img_rows), cv2.INTER_LINEAR)
    elif mode == 'mod':
        rotate = random.uniform(-10, 10)
        mat = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), rotate, 1)
        img = cv2.warpAffine(img, mat, (img.shape[1], img.shape[0]))
        resized = cv2.resize(img, (img_cols, img_rows), cv2.INTER_LINEAR)

    return resized


def get_driver_data() -> Tuple[Dict[str, str], Dict[str, List[Tuple[str, str]]]]:
    """ Returns a dictionary of image name mapped to subject"""
    dr = dict()
    clss = dict()
    path = os.path.join(os.path.dirname(__file__), '..', 'input', 'driver_imgs_list.csv')
    print('Read drivers data')
    f = open(path, 'r')
    f.readline()
    while True:
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        dr[arr[2]] = arr[0]
        if arr[0] not in clss.keys():
            clss[arr[0]] = [(arr[1], arr[2])]
        else:
            clss[arr[0]].append((arr[1], arr[2]))
    f.close()

    return dr, clss


def load_train(img_rows: int, img_cols: int, color_type=1, mode='norm') -> \
        Tuple[List[np.ndarray], List[int], List[str], List[str], List[str]]:
    """ Loads training data images into specified folder as file glob
        Args:
            img_rows: Row pixel dimension of images
            img_cols: Column pixel dimension of images
            color_type: 1 indicates gray, else RGB
            mode: Mode of loading training data
        Returns:
            Tuple of resized images and image type
    """
    x_train: List[np.ndarray] = []
    x_train_id = []
    y_train = []
    driver_id = []
    start_time = time.time()
    driver_data, dr_class = get_driver_data()

    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join(os.path.dirname(__file__), '..', 'input', 'imgs', 'train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            fl_base = os.path.basename(fl)
            img = get_im(fl, img_rows, img_cols, color_type, mode)
            x_train.append(img)
            x_train_id.append(fl_base)
            y_train.append(j)
            driver_id.append(driver_data[fl_base])

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    unique_drivers = sorted(list(set(driver_id)))
    print('Unique drivers: {}'.format(len(unique_drivers)))
    print(unique_drivers)

    return x_train, y_train, x_train_id, driver_id, unique_drivers


def load_test(img_rows: int, img_cols: int, color_type=1, mode='norm') -> Tuple[List[np.ndarray], List[str]]:
    """ Loads test data images into specified folder as file glob
        Args:
            img_rows: Row pixel dimension of images
            img_cols: Column pixel dimension of images
            color_type: 1 indicates gray, else RGB
            mode: Mode of loading test data
        Returns:
            Tuple of resized images and image name
    """
    print('Read test images')
    path = os.path.join(os.path.dirname(__file__), '..', 'input', 'imgs', 'test', '*.jpg')
    files = glob.glob(path)
    x_test = []
    x_test_id = []
    total = 0
    start_time = time.time()
    thr = math.floor(len(files) / 10)
    for fl in files:
        fl_base = os.path.basename(fl)
        img = get_im(fl, img_rows, img_cols, color_type, mode)
        x_test.append(img)
        x_test_id.append(fl_base)
        total += 1
        if total % thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    print('Read test data time: {} seconds'.format(round(time.time() - start_time, 2)))

    return x_test, x_test_id


def load_test_vgg(part: int, img_rows: int, img_cols: int, color_type=1, mode='norm') \
        -> Tuple[List[np.ndarray], List[str]]:
    """ Loads test data images into specified folder as file glob
        Args:
            part: Partition number of test data
            img_rows: Row pixel dimension of images
            img_cols: Column pixel dimension of images
            color_type: 1 indicates gray, else RGB
            mode: Mode of loading test data
        Returns:
            Tuple of resized images and image name
    """
    path = os.path.join('..', 'input', 'test', '*.jpg')
    files = sorted(glob.glob(path))
    ch = split_list(files, 5)

    x_test = []
    x_test_id = []
    print('Start image: ' + str(ch[part][0]))
    print('Last image: ' + str(ch[part][-1]))
    for fl in ch[part]:
        fl_base = os.path.basename(fl)
        img = get_im(fl, img_rows, img_cols, color_type, mode)
        x_test.append(img)
        x_test_id.append(fl_base)

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


def save_model(model: Model, arch_path: str = None, weights_path: str = None,
               file_name_suffix: str = '') -> None:
    """ Saves model as JSON file in the specified path
        Args:
            model: Trained keras model to be saved
            arch_path: Path of architecture file
            weights_path: Path of weights files
            file_name_suffix: Optional file name suffix
    """
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    if arch_path and weights_path:
        open(arch_path, 'w').write(json_string)
        model.save_weights(weights_path, overwrite=True)
    else:
        open(os.path.join(os.path.dirname(__file__), '..', 'cache',
                          'architecture' + file_name_suffix + '.json'), 'w').write(json_string)
        model.save_weights(os.path.join(os.path.dirname(__file__), '..', 'cache',
                                        'model_weights' + file_name_suffix + '.h5'), overwrite=True)


def read_model(arch_path: str = None, weights_path: str = None) -> Model:
    """ Reads model from specified path, loads weights and return the model
        Args:
            arch_path: Path of architecture file
            weights_path: Path of weights files
        Returns:
            Returns a keras model with pre trained weights
    """
    if arch_path and weights_path:
        model = model_from_json(open(arch_path).read())
        model.load_weights(weights_path)
    else:
        model = model_from_json(open(os.path.join(os.path.dirname(__file__), '..', 'cache',
                                                  'architecture.json')).read())
        model.load_weights(os.path.join(os.path.dirname(__file__), '..', 'cache', 'model_weights.h5'))

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
        suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
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


def show_image(im: np.ndarray, name: str = 'image') -> None:
    """ Function to display an image and then clear it after keystroke
        Args:
            im: Image to be displayed
            name: Name of the image
    """
    cv2.imshow(name, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def copy_selected_drivers(train_data: np.ndarray, train_target: np.ndarray,
                          driver_id: List[str], driver_list: List[str]) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Function to copy data of selected drivers provided
        Args:
            train_data: Training data set
            train_target: Target values of training data set
            driver_id: Selected driver ids to be copied
            driver_list: List of all driver ids
        Returns:
            Training data, target values and index positions of selected drivers
    """
    data = []
    target = []
    index = []

    for i in range(len(driver_id)):
        if driver_id[i] in driver_list:
            data.append(train_data[i])
            target.append(train_target[i])
            index.append(i)
    data = np.array(data, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    index = np.array(index, dtype=np.uint32)

    return data, target, index


def save_useful_data(predictions_valid: List[float], valid_ids: List[str], model: Sequential,
                     info: str) -> None:
    """ Saves model predictions, model and submission file
        Args:
            predictions_valid: Model predictions
            valid_ids: Driver Ids predicted for
            model: Model used to predict distracted drivers
            info: Model loss or other info used to provide a specific name to the file
                  for future reference
    """
    result1 = pd.DataFrame(predictions_valid, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(valid_ids, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir(os.path.join('subm', 'data')):
        os.mkdir(os.path.join('subm', 'data'))
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    # Save predictions
    pred_file = os.path.join('subm', 'data', 's_' + suffix + '_train_predictions.csv')
    result1.to_csv(pred_file, index=False)
    # Save model
    json_string = model.to_json()
    model_file = os.path.join('subm', 'data', 's_' + suffix + '_model.json')
    open(model_file, 'w').write(json_string)
    # Save code
    cur_code = os.path.realpath(__file__)
    code_file = os.path.join('subm', 'data', 's_' + suffix + '_code.py')
    copy2(cur_code, code_file)


def get_validation_predictions(train_data: np.ndarray, predictions_valid: Dict[int, float]) -> List[float]:
    """ Appends the predictions into a list
        Args:
            train_data: Training data set
            predictions_valid: Predictions for valid drivers
        Returns:
            Predictions for valid drivers as a list
    """
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])

    return pv


def append_chunk(main: List[Any], part: List[Any]) -> List[Any]:
    """ Add the components of a split list to a main list
        Args:
            main: Main list
            part: A split list
        Returns:
            Appended main list
    """
    for p in part:
        main.append(p)

    return main


def split_list(lst: List[str], wanted_parts: int = 1) -> List[List[str]]:
    """ Splits a list into a list of lists of each of size 'wanted_parts'
        Args:
            lst: List to be split into smaller parts
            wanted_parts: Desired size of each smaller list
        Returns:
            A list of lists of each of size 'wanted_parts'
    """
    length = len(lst)

    return [lst[i * length // wanted_parts: (i + 1) * length // wanted_parts] for i in range(wanted_parts)]


def normalize_image(img: np.ndarray) -> np.ndarray:
    """ Normalizes an input image
        Args:
            img: Input image
        Returns:
            Normalized Image
    """
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68

    return img
