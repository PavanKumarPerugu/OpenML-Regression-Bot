'''2. Configurations File'''


'''Importing all the required libraries'''

'''For creating Configuration Spaces'''
from ConfigSpace import *
from ConfigSpace import ConfigurationSpace

'''Importing argparse module for the purpose of arguments provision for the ConfigSpace creation'''
import argparse

'''For provision of Hyperparameters etc..'''
import sklearn
import sklearn.datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.datasets import make_regression


'''2. Creating/Defining the Configuration Space based on the regression algorithm name that argumented'''
'''Ref: https://github.com/automl/auto-sklearn/tree/development/autosklearn/pipeline/components/regression'''

def config(reggression_algorithm):

    '''For Decision Tree Regressor..'''
    if reggression_algorithm == 'decision_tree':
        cs = ConfigurationSpace()

        criterion = CategoricalHyperparameter(
            "criterion", ["poisson", "friedman_mse", "absolute_error"]
        )
        max_features = Constant("max_features", 1.0)
        max_depth_factor = UniformFloatHyperparameter(
            "max_depth_factor", 0.0, 2.0, default_value=0.5
        )
        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default_value=2
        )
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default_value=1
        )
        min_weight_fraction_leaf = Constant("min_weight_fraction_leaf", 0.0)
        min_impurity_decrease = UnParametrizedHyperparameter(
            "min_impurity_decrease", 0.0
        )

        cs.add_hyperparameters(
            [
                criterion,
                max_features,
                min_samples_split,
                min_samples_leaf,
                min_weight_fraction_leaf,
                min_impurity_decrease,
            ]
        )

        clf = sklearn.tree.DecisionTreeRegressor()
        config = cs.sample_configuration()
        params = config.get_dictionary()
        clf.set_params(**params)
        print(clf)

    '''For Random Forest Regressor..'''
    if reggression_algorithm == 'random_forest':
        cs = ConfigurationSpace()
        criterion = CategoricalHyperparameter(
            "criterion", ["poisson", "friedman_mse", "absolute_error"]
        )

        # In contrast to the random forest classifier we want to use more max_features
        # and therefore have this not on a sqrt scale
        max_features = UniformFloatHyperparameter(
            "max_features", 0.1, 1.0, default_value=1.0
        )

        max_depth = UnParametrizedHyperparameter("max_depth", "None")
        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default_value=2
        )
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default_value=1
        )
        min_weight_fraction_leaf = UnParametrizedHyperparameter(
            "min_weight_fraction_leaf", 0.0
        )
        max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
        min_impurity_decrease = UnParametrizedHyperparameter(
            "min_impurity_decrease", 0.0
        )
        bootstrap = CategoricalHyperparameter(
            "bootstrap", [True, False], default_value=True
        )

        cs.add_hyperparameters(
            [
                criterion,
                max_features,
                # max_depth,
                min_samples_split,
                min_samples_leaf,
                min_weight_fraction_leaf,
                # max_leaf_nodes,
                min_impurity_decrease,
                bootstrap,
            ]
        )
        clf = sklearn.ensemble.RandomForestRegressor()
        config = cs.sample_configuration()
        params = config.get_dictionary()
        clf.set_params(**params)
        print(clf)

    '''For gradient_boosting..'''
    if reggression_algorithm == 'gradient_boosting':
        cs = ConfigurationSpace()
        loss = CategoricalHyperparameter(
            "loss", ["least_squares"], default_value="least_squares"
        )
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.01, upper=1, default_value=0.1, log=True
        )
        min_samples_leaf = UniformIntegerHyperparameter(
            name="min_samples_leaf", lower=1, upper=200, default_value=20, log=True
        )
        max_depth = UnParametrizedHyperparameter(name="max_depth", value="None")
        max_leaf_nodes = UniformIntegerHyperparameter(
            name="max_leaf_nodes", lower=3, upper=2047, default_value=31, log=True
        )
        max_bins = Constant("max_bins", 255)
        l2_regularization = UniformFloatHyperparameter(
            name="l2_regularization",
            lower=1e-10,
            upper=1,
            default_value=1e-10,
            log=True,
        )

        early_stop = CategoricalHyperparameter(
            name="early_stop", choices=["off", "valid", "train"], default_value="off"
        )
        tol = UnParametrizedHyperparameter(name="tol", value=1e-7)
        scoring = UnParametrizedHyperparameter(name="scoring", value="loss")
        n_iter_no_change = UniformIntegerHyperparameter(
            name="n_iter_no_change", lower=1, upper=20, default_value=10
        )
        validation_fraction = UniformFloatHyperparameter(
            name="validation_fraction", lower=0.01, upper=0.4, default_value=0.1
        )

        cs.add_hyperparameters(
            [
                loss,
                learning_rate,
                min_samples_leaf,
                max_depth,
                max_leaf_nodes,
                max_bins,
                l2_regularization,
                early_stop,
                tol,
                scoring,
                n_iter_no_change,
                validation_fraction,
            ]
        )

        n_iter_no_change_cond = InCondition(
            n_iter_no_change, early_stop, ["valid", "train"]
        )
        validation_fraction_cond = EqualsCondition(
            validation_fraction, early_stop, "valid"
        )

        cs.add_conditions([n_iter_no_change_cond, validation_fraction_cond])
        clf = sklearn.ensemble.GradientBoostingRegressor()
        config = cs.sample_configuration()
        params = config.get_dictionary()
        clf.set_params(**params)
        print(clf)

    '''For k_nearest_neighbors..'''
    if reggression_algorithm == 'k_nearest_neighbors':
        cs = ConfigurationSpace()

        n_neighbors = UniformIntegerHyperparameter(
            name="n_neighbors", lower=1, upper=100, log=True, default_value=1
        )
        weights = CategoricalHyperparameter(
            name="weights", choices=["uniform", "distance"], default_value="uniform"
        )
        p = CategoricalHyperparameter(name="p", choices=[1, 2], default_value=2)

        cs.add_hyperparameters([n_neighbors, weights, p])
        clf = sklearn.neighbors.KNeighborsRegressor()
        config = cs.sample_configuration()
        params = config.get_dictionary()
        clf.set_params(**params)
        print(clf)


    '''For ARDRegression..'''
    if reggression_algorithm == 'ARDRegression':
        cs = ConfigurationSpace()
        n_iter = UnParametrizedHyperparameter("n_iter", value=300)
        tol = UniformFloatHyperparameter(
            "tol", 10 ** -5, 10 ** -1, default_value=10 ** -3, log=True
        )
        alpha_1 = UniformFloatHyperparameter(
            name="alpha_1", lower=10 ** -10, upper=10 ** -3, default_value=10 ** -6
        )
        alpha_2 = UniformFloatHyperparameter(
            name="alpha_2",
            log=True,
            lower=10 ** -10,
            upper=10 ** -3,
            default_value=10 ** -6,
        )
        lambda_1 = UniformFloatHyperparameter(
            name="lambda_1",
            log=True,
            lower=10 ** -10,
            upper=10 ** -3,
            default_value=10 ** -6,
        )
        lambda_2 = UniformFloatHyperparameter(
            name="lambda_2",
            log=True,
            lower=10 ** -10,
            upper=10 ** -3,
            default_value=10 ** -6,
        )
        threshold_lambda = UniformFloatHyperparameter(
            name="threshold_lambda",
            log=True,
            lower=10 ** 3,
            upper=10 ** 5,
            default_value=10 ** 4,
        )
        fit_intercept = UnParametrizedHyperparameter("fit_intercept", "True")

        cs.add_hyperparameters(
            [
                n_iter,
                tol,
                alpha_1,
                alpha_2,
                lambda_1,
                lambda_2,
                threshold_lambda,
                fit_intercept,
            ]
        )
        clf = sklearn.linear_model.ARDRegression()
        config = cs.sample_configuration()
        params = config.get_dictionary()
        clf.set_params(**params)
        print(clf)


    '''Returns the Regressor Configuration..'''
    return (clf)