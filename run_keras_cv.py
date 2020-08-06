""" Code file to create a keras solution with cross-validation """

from typing import Union
from sklearn.model_selection import KFold
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.metrics import log_loss
from utils import *


def read_and_normalize_train_data(img_rows: int, img_cols: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Function to read and normalize training data
        Args:
            img_rows: Row pixel dimension of images
            img_cols: Column pixel dimension of images
        Returns:
            Normalized training data
    """
    cache_path = os.path.join('cache', 'train_r_' + str(img_rows) + '_c_' + str(img_cols) + '.dat')
    if not os.path.isfile(cache_path):
        train_data, train_target, _, _, _ = load_train(img_rows, img_cols)
        cache_data((train_data, train_target), cache_path)
    else:
        print('Restore train from cache!')
        (train_data, train_target) = restore_data(cache_path)

    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)
    train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
    train_target = np_utils.to_categorical(train_target, 10)
    train_data = train_data.astype('float32')
    train_data /= 255
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target


def read_and_normalize_test_data(img_rows: int, img_cols: int) -> Tuple[np.ndarray, Union[List[str], np.ndarray]]:
    """ Function to read and normalize test data
        Args:
            img_rows: Row pixel dimension of images
            img_cols: Column pixel dimension of images
        Returns:
            Normalized test data
    """
    cache_path = os.path.join('cache', 'test_r_' + str(img_rows) + '_c_' + str(img_cols) + '.dat')
    if not os.path.isfile(cache_path):
        test_data, test_id = load_test(img_rows, img_cols)
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
    return test_data, test_id


def run_cross_validation(n_folds: int = 10) -> None:
    """ Function to derive a keras solution with cross-validation
        Args:
            n_folds: Number of cross-validation folds
    """
    np.random.seed(2016)

    # input image dimensions
    img_rows, img_cols = 24, 32
    batch_size = 64
    nb_classes = 10
    nb_epoch = 1
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
    random_state = 51

    train_data, train_target = read_and_normalize_train_data(img_rows, img_cols)
    test_data, test_id = read_and_normalize_test_data(img_rows, img_cols)

    y_full_train = dict()
    y_full_test = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    num_fold = 0
    for train_index, test_index in kf:
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, n_folds))
        x_train, x_valid = train_data[train_index], train_data[test_index]
        y_train, y_valid = train_target[train_index], train_target[test_index]
        print('Split train: ', len(x_train))
        print('Split valid: ', len(x_valid))

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

        model.compile(loss='categorical_crossentropy', optimizer='adadelta')

        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                  show_accuracy=True, verbose=1, validation_data=(x_valid, y_valid))

        predictions_valid = model.predict(x_valid, batch_size=128, verbose=1)
        score = log_loss(y_valid, predictions_valid)
        print('Score log_loss: ', score)

        # Store valid predictions
        for i in range(len(test_index)):
            y_full_train[test_index[i]] = predictions_valid[i]

        # Store test predictions
        test_prediction = model.predict(test_data, batch_size=128, verbose=1)
        y_full_test.append(test_prediction)

    score = log_loss(train_target, dict_to_list(y_full_train))
    print('Final score log_loss: ', score)

    test_res = merge_several_folds_fast(y_full_test, n_folds)
    create_submission(test_res, test_id, score)
