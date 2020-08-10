""" Main Code file to get predictions to submit on Kaggle"""

import os
from keras import backend
from run_keras_simple import get_simple_keras_solution
from run_keras_cv import run_cross_validation
from run_keras_cv_drivers import run_cross_validation_cv_drivers
from run_keras_cv_drivers_v2 import run_cross_validation_v2
from kaggle_distracted_drivers_vgg16 import run_cross_validation_create_models, run_cross_validation_process_test

backend.set_image_dim_ordering('th')
get_simple_keras_solution()
run_cross_validation(10)
run_cross_validation_cv_drivers(5)
run_cross_validation_v2(13)

num_folds = 10
os.path.join(os.path.dirname(__file__), '..', 'subm')
if not os.path.isdir("subm"):
    os.mkdir("subm")
if not os.path.isdir("cache"):
    os.mkdir("cache")
weight_path = os.path.join(os.path.dirname(__file__), 'weights/vgg16_weights.h5')
if not os.path.isfile(weight_path):
    print('Please put VGG16 pretrained weights in weights/vgg16_weights.h5')
    exit()
run_cross_validation_create_models(num_folds)
run_cross_validation_process_test(num_folds)
