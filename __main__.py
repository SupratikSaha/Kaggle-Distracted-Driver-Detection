""" Main Code file to get predictions to submit on Kaggle"""
from run_keras_simple import get_simple_keras_solution
from run_keras_cv import run_cross_validation

get_simple_keras_solution()
run_cross_validation(10)
