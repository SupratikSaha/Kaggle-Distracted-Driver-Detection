""" Code file to create a version 2 keras model using cross-validation """

import warnings
from typing import Union
from sklearn.model_selection import KFold
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from sklearn.metrics import log_loss
from utils import *


def read_and_normalize_train_data_rotated(img_rows: int, img_cols: int, use_cache: int = 0,
                                          color_type: int = 1) -> \
        Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    """ Function to read and normalize training images in color and rotated form
        Args:
            img_rows: Row pixel dimension of images
            img_cols: Column pixel dimension of images
            color_type: 1 indicates gray, else RGB
            use_cache: Indicates if cache is to be used
        Returns:
            Normalized and rotated training data
    """
    cache_path = os.path.join(os.path.dirname(__file__), '..', 'cache', 'train_r_' +
                              str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color_type) + '_rotated.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        train_data, train_target, train_id, driver_id, unique_drivers = \
            load_train(img_rows, img_cols, color_type, 'mod')
        cache_data((train_data, train_target, train_id, driver_id, unique_drivers), cache_path)
    else:
        print('Restore train from cache!')
        (train_data, train_target, train_id, driver_id, unique_drivers) = restore_data(cache_path)

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

    return train_data, train_target, train_id, driver_id, unique_drivers


def read_and_normalize_test_data_rotated(img_rows: int, img_cols: int, use_cache: int = 0,
                                         color_type: int = 1) -> \
        Tuple[np.ndarray, Union[List[str], np.ndarray]]:
    """ Function to read and normalize rotated test data
            Args:
                img_rows: Row pixel dimension of images
                img_cols: Column pixel dimension of images
                use_cache: Indicates if cache is to be used
                color_type: 1 indicates gray, else RGB
            Returns:
                Normalized and rotated test data
        """
    cache_path = os.path.join(os.path.dirname(__file__), '..', 'cache', 'test_r_' +
                              str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color_type) + '_rotated.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        test_data, test_id = load_test(img_rows, img_cols, color_type, 'mod')
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


def create_model_v2(img_rows: int, img_cols: int, color_type: int = 1) -> Sequential:
    """ Create version 2 of the predicting model
        Args:
            img_rows: Row pixel dimension of images
            img_cols: Column pixel dimension of images
            color_type: 1 indicates gray, else RGB
        Returns:
            Created Model
    """
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal',
                            input_shape=(color_type, img_rows, img_cols)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
    model.add(MaxPooling2D(pool_size=(8, 8)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(Adam(lr=1e-3), loss='categorical_crossentropy')

    return model


def run_cross_validation_v2(n_folds: int = 10) -> None:
    """ V2 of Function to derive a keras solution with cross-validation for selected drivers
        Args:
            n_folds: Number of cross-validation folds
    """
    np.random.seed(2016)
    warnings.filterwarnings("ignore")
    use_cache = 1

    # input image dimensions
    img_rows, img_cols = 64, 64
    # color type: 1 - grey, 3 - rgb
    color_type_global = 1
    batch_size = 16
    nb_epoch = 50
    random_state = 51
    restore_from_last_checkpoint = 0

    train_data, train_target, train_id, driver_id, unique_drivers = \
        read_and_normalize_train_data_rotated(img_rows, img_cols, use_cache, color_type_global)
    test_data, test_id = read_and_normalize_test_data_rotated(img_rows, img_cols, use_cache, color_type_global)
    model = create_model_v2(img_rows, img_cols, color_type_global)

    y_full_train = dict()
    y_full_test = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    num_fold = 0
    sum_score = 0
    for train_drivers, test_drivers in kf.split(unique_drivers):
        unique_list_train = [unique_drivers[i] for i in train_drivers]
        x_train, y_train, train_index = \
            copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
        unique_list_valid = [unique_drivers[i] for i in test_drivers]
        x_valid, y_valid, test_index = \
            copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, n_folds))
        print('Split train: ', len(x_train), len(y_train))
        print('Split valid: ', len(x_valid), len(y_valid))
        print('Train drivers: ', unique_list_train)
        print('Test drivers: ', unique_list_valid)

        k_fold_weights_path = os.path.join(os.path.dirname(__file__), '..', 'cache',
                                           'weights_k_fold_' + str(num_fold) + '.h5')
        if not os.path.isfile(k_fold_weights_path) or restore_from_last_checkpoint == 0:
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=1, verbose=0),
                ModelCheckpoint(k_fold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
            ]
            model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                      shuffle=True, verbose=1, validation_data=(x_valid, y_valid),
                      callbacks=callbacks)
        if os.path.isfile(k_fold_weights_path):
            model.load_weights(k_fold_weights_path)

        predictions_valid = model.predict(x_valid, batch_size=batch_size, verbose=1)
        score = log_loss(y_valid, predictions_valid)
        print('Score log_loss: ', score)
        sum_score += score * len(test_index)

        # Store valid predictions
        for i in range(len(test_index)):
            y_full_train[test_index[i]] = predictions_valid[i]

        # Store test predictions
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=1)
        y_full_test.append(test_prediction)

    score = sum_score / len(train_data)
    print("Log_loss train independent avg: ", score)

    predictions_valid = get_validation_predictions(train_data, y_full_train)
    score1 = log_loss(train_target, predictions_valid)
    if abs(score1 - score) > 0.0001:
        print('Check error: {} != {}'.format(score, score1))

    print('Final log_loss: {}, rows: {} cols: {} n_folds: {} epoch: {}'.format(
        score, img_rows, img_cols, n_folds, nb_epoch))

    test_res = merge_several_folds_fast(y_full_test, n_folds)
    create_submission(test_res, test_id, 'keras_cv_drivers_v2')
    save_useful_data(predictions_valid, train_id, model, 'keras_cv_drivers_v2')


def run_single():
    """ Function to derive a keras solution selected drivers without cross-validation"""
    np.random.seed(2016)
    warnings.filterwarnings("ignore")
    use_cache = 1
    # input image dimensions
    img_rows, img_cols = 64, 64
    color_type_global = 1
    batch_size = 32
    nb_epoch = 50

    train_data, train_target, train_id, driver_id, unique_drivers = \
        read_and_normalize_train_data_rotated(img_rows, img_cols, use_cache, color_type_global)
    test_data, test_id = read_and_normalize_test_data_rotated(img_rows, img_cols, use_cache, color_type_global)

    y_full_test = []
    unique_list_train = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p035', 'p041', 'p042', 'p045', 'p047',
                         'p049', 'p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p075', 'p081']
    x_train, y_train, train_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
    unique_list_valid = ['p024', 'p026', 'p039', 'p072']
    x_valid, y_valid, test_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)

    print('Start Single Run')
    print('Split train: ', len(x_train))
    print('Split valid: ', len(x_valid))
    print('Train drivers: ', unique_list_train)
    print('Valid drivers: ', unique_list_valid)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, verbose=0),
    ]
    model = create_model_v2(img_rows, img_cols, color_type_global)
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              shuffle=True, verbose=1, validation_data=(x_valid, y_valid),
              callbacks=callbacks)

    predictions_valid = model.predict(x_valid, batch_size=batch_size, verbose=1)
    score = log_loss(y_valid, predictions_valid)
    print('Score log_loss: ', score)

    # Store test predictions
    test_prediction = model.predict(test_data, batch_size=batch_size, verbose=1)
    y_full_test.append(test_prediction)

    print('Final log_loss: {}, rows: {} cols: {} epoch: {}'.format(score, img_rows, img_cols, nb_epoch))

    full_pred = model.predict(train_data, batch_size=batch_size, verbose=1)
    score = log_loss(train_target, full_pred)
    print('Full score log_loss: ', score)

    test_res = merge_several_folds_fast(y_full_test, 1)
    create_submission(test_res, test_id, 'keras_cv_drivers_v2_single')
    save_useful_data(full_pred, train_id, model, 'keras_cv_drivers_v2_single')
