""" Code file to create a model using pre-trained vgg-16 model """

import warnings
import h5py
from typing import Union
from numpy.random import permutation
from sklearn.model_selection import KFold
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.utils import np_utils
from sklearn.metrics import log_loss
from utils import *


def read_and_normalize_train_data(img_rows: int, img_cols: int, use_cache: int = 0, color_type: int = 1) -> \
        Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    """ Function to read and normalize training images in color and rotated form
            Args:
                img_rows: Row pixel dimension of images
                img_cols: Column pixel dimension of images
                use_cache: Indicates if cache is to be used
                color_type: 1 indicates gray, else RGB
            Returns:
                Normalized and rotated training data
        """
    cache_path = os.path.join(os.path.dirname(__file__), '..', 'cache', 'train_r_' +
                              str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color_type) + '.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        train_data, train_target, train_id, driver_id, unique_drivers = \
            load_train(img_rows, img_cols, color_type, 'cv2')
        cache_data((train_data, train_target, train_id, driver_id, unique_drivers), cache_path)
    else:
        print('Restore train from cache!')
        (train_data, train_target, train_id, driver_id, unique_drivers) = restore_data(cache_path)

    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    print('Reshape...')
    train_data = train_data.transpose((0, 3, 1, 2))

    print('Convert to float...')
    train_data = train_data.astype('float16')
    mean_pixel = [103.939, 116.779, 123.68]
    print('Subtract 0...')
    train_data[:, 0, :, :] -= mean_pixel[0]
    print('Subtract 1...')
    train_data[:, 1, :, :] -= mean_pixel[1]
    print('Subtract 2...')
    train_data[:, 2, :, :] -= mean_pixel[2]

    train_target = np_utils.to_categorical(train_target, 10)

    # Shuffle experiment START !!!
    perm = permutation(len(train_target))
    train_data = train_data[perm]
    train_target = train_target[perm]
    # Shuffle experiment END !!!

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')

    return train_data, train_target, train_id, driver_id, unique_drivers


def read_and_normalize_test_data(part: int, img_rows: int, img_cols: int, use_cache: int = 0, color_type: int = 1) -> \
        Tuple[np.ndarray, Union[List[str], np.ndarray]]:
    """ Function to read and normalize rotated test data
                Args:
                    part: Partition number of test data
                    img_rows: Row pixel dimension of images
                    img_cols: Column pixel dimension of images
                    use_cache: Indicates if cache is to be used
                    color_type: 1 indicates gray, else RGB
                Returns:
                    Normalized and rotated test data
            """
    start_time = time.time()
    cache_path = os.path.join(os.path.dirname(__file__), '..', 'cache', 'test_r_' + str(img_rows) +
                              '_c_' + str(img_cols) +
                              '_part_' + str(part) +
                              '.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        test_data, test_id = load_test_vgg(part, img_rows, img_cols, color_type,  'cv2')
        cache_data((test_data, test_id), cache_path)
    else:
        print('Restore test from cache [{}]!'.format(part))
        (test_data, test_id) = restore_data(cache_path)

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.transpose((0, 3, 1, 2))

    test_data = test_data.astype('float16')
    mean_pixel = [103.939, 116.779, 123.68]
    test_data[:, 0, :, :] -= mean_pixel[0]
    test_data[:, 1, :, :] -= mean_pixel[1]
    test_data[:, 2, :, :] -= mean_pixel[2]

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))

    return test_data, test_id


def vgg_16() -> Sequential:
    """ Create vgg model
        Returns:
            Created Model
    """
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), stride=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), stride=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), stride=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), stride=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), stride=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    f = h5py.File('weights/vgg16_weights.h5')
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the saved file
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model


class EarlyStoppingByLossVal(Callback):
    """ Class to control Early Stopping by tracking validation Loss """
    def __init__(self, monitor: str = 'val_loss', value: float = 0.00001, verbose: int = 0) -> None:
        """ init function for EarlyStoppingByLossVal class
        monitor: Parameter being tracked to measure model performance
        value: Loss threshold value
        verbose: Dictates model verbosity level
        """
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = None) -> None:
        """ Method that stops training and produces log messages on early stopping 
            epoch: Epoch number
            logs: Dictionary that holds the monitor being tracked and its value
        """
        if logs is None:
            logs = {}
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


def run_cross_validation_create_models(n_folds: int = 10) -> None:
    """ Function to build a vgg-16 model using cross-validation """
    # input image dimensions
    batch_size = 16
    nb_epoch = 25
    random_state = 51
    restore_from_last_checkpoint = 1
    model = Sequential()
    img_rows, img_cols = 224, 224
    color_type = 3
    use_cache = 1
    warnings.filterwarnings("ignore")
    np.random.seed(2016)

    train_data, train_target, train_id, driver_id, unique_drivers = \
        read_and_normalize_train_data(img_rows, img_cols, use_cache, color_type)

    y_full_train = dict()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    num_fold = 0
    sum_score = 0
    for train_drivers, test_drivers in kf.split(train_data):
        model = vgg_16()
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

        k_fold_weights_path = os.path.join(os.path.dirname(__file__), '..', 'cache',
                                           'weights_k_fold_vgg16_' + str(num_fold) + '.h5')
        if not os.path.isfile(k_fold_weights_path) or restore_from_last_checkpoint == 0:
            callbacks = [
                EarlyStoppingByLossVal(monitor='val_loss', value=0.00001, verbose=1),
                EarlyStopping(monitor='val_loss', patience=5, verbose=0),
                ModelCheckpoint(k_fold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
            ]
            model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                      shuffle=True, verbose=1, validation_data=(x_valid, y_valid),
                      callbacks=callbacks)
        if os.path.isfile(k_fold_weights_path):
            model.load_weights(k_fold_weights_path)

        predictions_valid = model.predict(x_valid.astype('float32'), batch_size=batch_size, verbose=1)
        score = log_loss(y_valid, predictions_valid)
        print('Score log_loss: ', score)
        sum_score += score * len(test_index)

        # Store valid predictions
        for i in range(len(test_index)):
            y_full_train[test_index[i]] = predictions_valid[i]

    score = sum_score / len(train_data)
    print("Log_loss train independent avg: ", score)

    predictions_valid = get_validation_predictions(train_data, y_full_train)

    print('Final log_loss: {}, n_folds: {} epoch: {}'.format(score, n_folds, nb_epoch))
    # info_string = 'loss_' + str(score) \
    #               + '_folds_' + str(n_folds) \
    #               + '_ep_' + str(nb_epoch)

    save_useful_data(predictions_valid, train_id, model, 'vgg_16')

    score1 = log_loss(train_target, predictions_valid)
    if abs(score1 - score) > 0.0001:
        print('Check error: {} != {}'.format(score, score1))


def run_cross_validation_process_test(n_folds: int = 10) -> None:
    """ Function to predict using the vgg-16 model on the test data """
    batch_size = 16
    num_fold = 0
    y_full_test = []
    test_id = []
    img_rows, img_cols = 224, 224
    color_type = 3
    use_cache = 1
    warnings.filterwarnings("ignore")
    np.random.seed(2016)

    for i in range(n_folds):
        model = vgg_16()
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, n_folds))
        k_fold_weights_path = os.path.join(os.path.dirname(__file__), '..', 'cache',
                                           'weights_k_fold_vgg16_' + str(num_fold) + '.h5')
        model.load_weights(k_fold_weights_path)

        k_fold_test_validation_path = os.path.join(os.path.dirname(__file__), '..', 'cache',
                                                   'test_k_fold_vgg16_' + str(num_fold) + '.pickle.dat')
        k_fold_test_ids_path = os.path.join(os.path.dirname(__file__), '..', 'cache',
                                            'test_k_fold_vgg16_ids.pickle.dat')
        if not os.path.isfile(k_fold_test_validation_path):
            test_prediction = []
            for part in range(5):
                print('Reading test data part {}...'.format(part))
                test_data_chunk, test_id_chunk = \
                    read_and_normalize_test_data(part, img_rows, img_cols, use_cache, color_type)
                test_prediction_chunk = model.predict(test_data_chunk, batch_size=batch_size, verbose=1)
                test_prediction = append_chunk(test_prediction, test_prediction_chunk)
                if i == 0:
                    test_id = append_chunk(test_id, test_id_chunk)
            cache_data(test_prediction, k_fold_test_validation_path)
            if i == 0:
                cache_data(test_id, k_fold_test_ids_path)
        else:
            print('Restore data from cache...')
            test_prediction = restore_data(k_fold_test_validation_path)
            if i == 0:
                test_id = restore_data(k_fold_test_ids_path)

        y_full_test.append(test_prediction)

    test_res = merge_several_folds_fast(y_full_test, n_folds)
    # info_string = 'loss_' \
    #               + '_r_' + str(224) \
    #               + '_c_' + str(224) \
    #               + '_folds_' + str(n_folds)
    suffix = 'vgg_16' + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
    cache_data((y_full_test, test_id), os.path.join(os.path.dirname(__file__), '..', "subm",
                                                    "full_array_" + suffix + ".pickle.dat"))
    create_submission(test_res, test_id, 'vgg_16')
    # Store debug submissions
    for i in range(n_folds):
        info_string1 = 'vgg_16' + '_debug_' + str(i)
        create_submission(y_full_test[i], test_id, info_string1)
