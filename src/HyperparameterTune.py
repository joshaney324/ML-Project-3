from itertools import permutations

import numpy as np


# This function is meant to tune the classification neural network model. It takes in datafolds, labels folds, a
# test/tune set it also takes a list hyperparameters to test in the grid search


def hyperparameter_tune_classification(data_folds, label_folds, test_data, test_labels, num_hidden_layers, num_inputs,
                                       num_outputs, max_iterations, learning_rates, all_hidden_layer_sizes):
    from CrossValidateFunctions import (cross_validate_tune_classification)

    # set up best value variables
    avg_metric = 0.0
    learning_rate = None
    hidden_layer_sizes = []

    for learning_rate_value in learning_rates:
        layer_perms = list(permutations(all_hidden_layer_sizes, num_hidden_layers))
        for layer_perm in layer_perms:
            # Get new metric for model
            new_metric = cross_validate_tune_classification(data_folds, label_folds, test_data, test_labels,
                                                            learning_rate_value, num_hidden_layers, layer_perm,
                                                            num_inputs, num_outputs, "classification", [],
                                                            max_iterations)
            # Check if model performed better
            if new_metric > avg_metric:
                avg_metric = new_metric
                learning_rate = learning_rate_value
                hidden_layer_sizes = layer_perm

    return learning_rate, hidden_layer_sizes

# This function is meant to tune the regression neural network model. It takes in datafolds, labels folds, a
# test/tune set it also takes a list hyperparameters to test in the grid search
def hyperparameter_tune_regression(data_folds, label_folds, test_data, test_labels, num_hidden_layers, num_inputs,
                                   num_outputs, max_iterations, learning_rates, all_hidden_layer_sizes):
    from CrossValidateFunctions import (cross_validate_tune_regression)

    # set up best value variables
    avg_metric = np.inf
    learning_rate = None
    hidden_layer_sizes = []

    for learning_rate_value in learning_rates:
        layer_perms = list(permutations(all_hidden_layer_sizes, num_hidden_layers))
        for layer_perm in layer_perms:
            # Get new metric for model
            new_metric = cross_validate_tune_regression(data_folds, label_folds, test_data, test_labels,
                                                            learning_rate_value, num_hidden_layers, layer_perm,
                                                            num_inputs, num_outputs, "regression", [],
                                                            max_iterations)
            # Check if model performed better
            if new_metric < avg_metric:
                avg_metric = new_metric
                learning_rate = learning_rate_value
                hidden_layer_sizes = layer_perm

    return learning_rate, hidden_layer_sizes
