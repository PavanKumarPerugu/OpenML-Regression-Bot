'''1. OpenML Regression Bot'''



'''(i). Importing all the required libraries'''

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
import Pipeline_Functions as pf


'''(ii). Main Bot run Function'''

'''Ref: https://github.com/openml/sklearn-bot'''
'''The main function, which gets the inputs from the other methods such as parse_args, pipeline, etc..
and run the pipeline on the input task, and to upload them back on to user's OpenML account'''
def runbot():

    '''For initializing the new logger to log the Info abot the Task ID and the Flow ID'''
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    '''For the provision of all necessary arguments from the User and to establish the connection between
    the interpreter and the OpenML platform'''
    args = pf.parse_args()
    if args.openml_apikey:
        openml.config.apikey = pf.args.openml_apikey
    if args.openml_server:
        openml.config.server = pf.args.openml_server
    else:
        openml.config.server = 'https://test.openml.org/api/v1/'

    '''Calling the Pipeline Method'''
    reg_pipeline = pf.pipeline()

    '''Running the defined model pipeline on the specified task, through it's ID'''
    run = openml.runs.run_model_on_task(reg_pipeline, args.task_id)

    '''Creating an Empty Scores list for the track of the Model performance in the specified metric 
    and so for the evaluation of the model'''
    score = []
    evaluations = run.fold_evaluations['mean_absolute_error'][0]
    for key in evaluations:
        score.append(evaluations[key])
    absolute_error = np.mean(score)

    '''For Printing all the final Metrics once onto console and to log them as INFO'''
    print('mean_absolute_error:', absolute_error)
    print('mean_squared_error:', np.mean(run.get_metric_fn(sklearn.metrics.mean_squared_error)))
    print('r2_score:', np.mean(run.get_metric_fn(sklearn.metrics.r2_score)))
    logging.info('Task %d - %s; Accuracy: %0.2f' % (args.task_id, pf.task.get_dataset().name, absolute_error))

    '''Calling the del_del_previous_runs Method to delete all the results saved in local directory 
    from the previous run'''
    pf.del_previous_runs()

    '''For Running the Model and to save the results into the local directory named 'Evaluations' 
    and to print the condition for --upload_result'''
    run.to_filesystem(directory='Evaluations')
    print('upload_result=', args.upload_result)

    '''For uploading the results to OpenML platform fot the condition true of --upload_result'''
    if args.upload_result:
        run.publish()
        print(run)
        print('Results have been Uploaded to Openml while stored locally in Evaluations folder.')
    else:
        print('As per your request the results have not been Uploaded but stored in Evaluations')

'''To run this in the main configuration that has been specified'''
if __name__ == '__main__':
    runbot()