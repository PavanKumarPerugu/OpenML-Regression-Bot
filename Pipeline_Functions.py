'''3. Functions developed for building pipelines etc..'''



'''Importing all the required libraries'''

'''For Importing OpenML Libraries which helps in docking our Interpreter for running Tasks'''
import openml

'''For accessing particular directory and to work in that'''
import os
import os.path

'''For logging the results as Info, i.e., The task Id and the Measured Metric, i.e., Prediction accuracy'''
import logging

'''For calculating the mean etc, and to pass one dimensional arrays in the training and prediction part'''
import numpy as np

'''For ignoring the unexpected warnings from unknown handling which actually not errors'''
import warnings
warnings.filterwarnings("ignore")

'''For the better documentation purpose, i.e., to provide the info to developer upon return types 
or passing parameter types'''
import typing

'''To interact with the user to get the configuration specifications while running the task or 
to get the info from the user as an input'''
import argparse

'''For the creation of configuration spaces with several Hyperparameters 
for the optimal performance of the model'''
from configs import config

'''For all the tasks such as pre-processing of the data, training the model, predictions etc..'''
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer

'''3. Defining all the customised functions'''

'''Ref: https://github.com/openml/sklearn-bot'''

'''(a) A function which envokes for the configuration parameters in this bot from the user while running'''
def parse_args():

    all_regressors = ['decision_tree','random_forest','gradient_boosting','k_nearest_neighbors',
                      'ARDRegression']
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_id', type=int, required=True,
                        help='the openml task id')
    parser.add_argument('--openml_server', type=str, default=None,
                        help='the openml server location')
    parser.add_argument('--openml_apikey', type=str, default=None,
                        help='the apikey to authenticate to OpenML')
    parser.add_argument('--regressor_name', type=str, choices=all_regressors,
                        default='decision_tree', help='the regressor to run')
    parser.add_argument('--upload_result', action='store_true',
                        help='if true, results will be immediately uploaded to OpenML.'
                             'Otherwise they will be stored on disk. ')
    parser.add_argument('--run_defaults', action='store_true',
                        help='if true, will run default configuration')
    return parser.parse_args()

'''(b) Pre-processing the two possible categories of the data for the training and predictions
i.e., Numerical Values and Categories(such as strings, sd in the form of different classes etc..)'''
args = parse_args()
task_id1=args.task_id
task=openml.tasks.get_task(task_id1)

'''Expecting targets to be list of integers'''
numeric_indices= typing.List[int]
nominal_indices= typing.List[int]

'''However there is equal probability for the occurrence of both Numerical and Nominal indices'''
nominal_indices = task.get_dataset().get_features_by_type('nominal', [task.target_name])
numeric_indices = task.get_dataset().get_features_by_type('numeric', [task.target_name])

'''If the data is Numeric, we fill all the NaN value with SimpleImputer and scale the whole data
in a standard manner'''
numeric_transformer = sklearn.pipeline.make_pipeline(
    sklearn.impute.SimpleImputer(strategy='mean'),
    sklearn.preprocessing.StandardScaler())

'''If the data is Nominal, we transform the categories/classes into Dummy numerical categories
  with OneHotEncoder for the better results/performance'''
nominal_transformer = sklearn.pipeline.make_pipeline(
    sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore'))

'''Final, Transformed , dataset'''
col_trans = sklearn.compose.ColumnTransformer(
    transformers=[
        ('numeric', numeric_transformer, numeric_indices),
        ('nominal', nominal_transformer, nominal_indices)],
    remainder='drop')
'''(c) Defining the final pipeline from the transformed data'''
def pipeline():
    args = parse_args()
    reggression_algorithm = args.regressor_name
    reg = config(reggression_algorithm)
    reg_pipeline = sklearn.pipeline.make_pipeline(col_trans, reg)
    return reg_pipeline

'''(d) Defining a function to delete the results of previous run from the local directory'''
def del_previous_runs():
    file_name = 'Evaluations'
    path = (os.path.abspath(file_name))
    if os.path.exists(path + "/flow.xml"):
        os.remove(path + "/flow.xml")
    if os.path.exists(path + "/model.pkl"):
        os.remove(path + "/model.pkl")
    if os.path.exists(path + "/predictions.arff"):
        os.remove(path + "/predictions.arff")
    if os.path.exists(path + "/description.xml"):
        os.remove(path + "/description.xml")
    else:
        print("The local Output directory is empty/ This could be the first run in this environment")
