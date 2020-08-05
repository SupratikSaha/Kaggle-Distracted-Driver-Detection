""" Main Code file to get predictions to submit on Kaggle"""

from run_keras_simple import get_simple_keras_solution
from run_keras_cv import run_cross_validation
from run_keras_cv_drivers import run_cross_validation_cv_drivers
from run_keras_cv_drivers_v2 import run_cross_validation_v2

get_simple_keras_solution()
run_cross_validation(10)
run_cross_validation_cv_drivers(5)
run_cross_validation_v2(13)
