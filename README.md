# OpenML-Regression-Bot

OpenML Regression Bot is an AutoML Bot that runs OpenML Regression Tasks for a user pre-defined configurations parameters, such as OpenML Regression Task ID and Regression Algorithm name. This Bot runs with enormous speed with almost best accuracy. Hence, the incorporation of OpMLRB into the noble community of Machine Learning leads to faster analysis of available datasets and the organised tasks.

## Version requirements of seveeral packages

ConfigSpace==0.4.19,
scikit_learn,
numpy==1.21.5,
openml==0.12.2

## Architecture/Methodology

the Open Regressor Bot was developed in order to address this issue and help researchers more easily analyse Regression Tasks using a pre-defined ML pipeline. With the Open Regressor Bot, researchers can input the Task ID and the Name of the Regressor Algorithm, which allows the Bot to access the dataset from the specified Task ID and run the model on the data. The results are then stored locally and uploaded to the OpenML Server for further analysis of the Model's performance. Incorporating this new development, researchers can now more easily analyse and experiment with Regression Tasks on OpenML, which are crucial to the field of Machine Learning.

This entire pipeline is modulated by slecting Tuned Hyperparameters from the configurational parametes , i.e., Regression Algorithm Name to perform much more effiently in less time and with almost zero extra efforts.

The user defined configurational parameters arre as follows for one of the test run cases:

![Cinfig parameter 2](https://user-images.githubusercontent.com/110840050/225812534-52ba7e01-f92e-4580-a68d-5872871dcb42.png)

## Application

The potintiality of the bot in performing the runs with a lighning speed demonstratees that This bot can become an optimal solution in a situation where with the minimal affordability of computaion aiming for better reauklts in less time.

## Results

The Bot saves the output results on the local directory as well as on OpenML server at time. The betterment in performance of the model with less number of samples only but by tuning the specified Hyperparameters. Even, with the saved Metrics the Model perfomance can be analysed.

![OpenML info for ](https://user-images.githubusercontent.com/110840050/225812927-2397dacd-dbdf-4b75-bb72-ff3072782d0e.png)

