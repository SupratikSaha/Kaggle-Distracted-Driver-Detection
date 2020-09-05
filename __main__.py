""" Main Code file to get predictions to submit on Kaggle"""

import os
from keras import backend
from convolution_quick import get_simple_keras_solution
from convolution_cross_validation import run_cross_validation
from convolution_drivers1 import run_cross_validation_cv_drivers
from convolution_drivers2 import run_cross_validation_v2
from convolution_vgg16 import run_cross_validation_create_models, run_cross_validation_process_test

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
