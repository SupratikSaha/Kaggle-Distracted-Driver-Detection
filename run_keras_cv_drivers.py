""" Code file to create a keras model using cross-validation """

from typing import Union
from sklearn.model_selection import KFold
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.metrics import log_loss
from utils import *


def read_and_normalize_train_data_color(img_rows: int, img_cols: int, use_cache: int = 0,
                                        color_type: int = 1) -> \
        Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """ Function to read and normalize training images in color
        Args:
            img_rows: Row pixel dimension of images
            img_cols: Column pixel dimension of images
            use_cache: Indicates if cache is to be used
            color_type: 1 indicates gray, else RGB
        Returns:
            Normalized training data
    """
    cache_path = os.path.join('cache',
                              'train_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color_type) + '.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        train_data, train_target, _, driver_id, unique_drivers = load_train(img_rows, img_cols, color_type)
        cache_data((train_data, train_target, driver_id, unique_drivers), cache_path)
    else:
        print('Restore train from cache!')
        (train_data, train_target, driver_id, unique_drivers) = restore_data(cache_path)

    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)
    if color_type == 1:
        train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
    else:
        train_data = train_data.transpose((0, 3, 1, 2))
    train_target = np_utils.to_categorical(train_target, 10)
    train_data = train_data.astype('float32')
    train_data /= 255
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')

    return train_data, train_target, driver_id, unique_drivers


def read_and_normalize_test_data_color(img_rows: int, img_cols: int, use_cache: int = 0, color_type=1) -> \
        Tuple[np.ndarray, Union[List[str], np.ndarray]]:
    """ Function to read and normalize test data
        Args:
            img_rows: Row pixel dimension of images
            img_cols: Column pixel dimension of images
            use_cache: Indicates if cache si to be used
            color_type: 1 indicates gray, else RGB
        Returns:
            Normalized test data
    """
    cache_path = os.path.join('cache',
                              'test_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color_type) + '.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        test_data, test_id = load_test(img_rows, img_cols, color_type)
        cache_data((test_data, test_id), cache_path)
    else:
        print('Restore test from cache!')
        (test_data, test_id) = restore_data(cache_path)

    test_data = np.array(test_data, dtype=np.uint8)
    if color_type == 1:
        test_data = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols)
    else:
        test_data = test_data.transpose((0, 3, 1, 2))
    test_data = test_data.astype('float32')
    test_data /= 255
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')

    return test_data, test_id


def create_model_v1(img_rows: int, img_cols: int, color_type: int = 1) -> Sequential:
    """ Create version 1 of the predicting model
        Args:
            img_rows: Row pixel dimension of images
            img_cols: Column pixel dimension of images
            color_type: 1 indicates gray, else RGB
        Returns:
            Created Model
    """
    nb_classes = 10
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(color_type, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    return model


def run_cross_validation_cv_drivers(n_folds: int = 10):
    """ Function to derive a keras solution with cross-validation for selected drivers
        Args:
            n_folds: Number of cross-validation folds
    """
    np.random.seed(2016)
    use_cache = 1
    # color type: 1 - grey, not 1 - rgb
    color_type_global = 1
    # input image dimensions
    img_rows, img_cols = 24, 32
    batch_size = 32
    nb_epoch = 1
    random_state = 51

    train_data, train_target, driver_id, unique_drivers = \
        read_and_normalize_train_data_color(img_rows, img_cols, use_cache, color_type_global)
    test_data, test_id = read_and_normalize_test_data_color(img_rows, img_cols, use_cache,
                                                            color_type_global)

    y_full_train = dict()
    y_full_test = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    num_fold = 0
    for train_drivers, test_drivers in kf:
        unique_list_train = [unique_drivers[i] for i in train_drivers]
        x_train, y_train, train_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
        unique_list_valid = [unique_drivers[i] for i in test_drivers]
        x_valid, y_valid, test_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, n_folds))
        print('Split train: ', len(x_train), len(y_train))
        print('Split valid: ', len(x_valid), len(y_valid))
        print('Train drivers: ', unique_list_train)
        print('Test drivers: ', unique_list_valid)

        model = create_model_v1(img_rows, img_cols, color_type_global)
        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                  show_accuracy=True, verbose=1, validation_data=(x_valid, y_valid))

        # score = model.evaluate(X_valid, Y_valid, show_accuracy=True, verbose=0)
        # print('Score log_loss: ', score[0])

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
    print('Final log_loss: {}, rows: {} cols: {} n_folds: {} epoch: {}'.format(score, img_rows,
                                                                               img_cols, n_folds,
                                                                               nb_epoch))
    info_string = 'loss_' + str(score) \
                  + '_r_' + str(img_rows) \
                  + '_c_' + str(img_cols) \
                  + '_folds_' + str(n_folds) \
                  + '_ep_' + str(nb_epoch)

    test_res = merge_several_folds_fast(y_full_test, n_folds)
    create_submission(test_res, test_id, info_string)
