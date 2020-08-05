""" Code file to create a simple solution using keras """

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.metrics import log_loss
from utils import *


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
    # input image dimensions
    img_rows, img_cols = 96, 128

    # Load training data
    cache_path = os.path.join(os.path.dirname(__file__), '..',  'cache', 'train.dat')
    if not os.path.isfile(cache_path):
        train_data, train_target, _, _, _ = load_train(img_rows, img_cols)
        if not os.path.isdir('cache'):
            os.mkdir('cache')
        cache_data((train_data, train_target), cache_path)
    else:
        print('Restore train from cache!')
        (train_data, train_target) = restore_data(cache_path)

    batch_size = 64
    nb_classes = 10
    nb_epoch = 2

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
    predictions = model.predict(test_data, batch_size=128, verbose=1)

    create_submission(predictions, test_id, score)
