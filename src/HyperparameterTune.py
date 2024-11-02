from itertools import permutations


# This function is meant to tune the classification KNN algorithms. It takes in datafolds, labels folds, a test/tune set
# it also takes a list of k vals and p vals to test in the grid search


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
            new_metric = cross_validate_tune_classification(data_folds, label_folds, test_data, test_labels,
                                                            learning_rate, num_hidden_layers, layer_perm,
                                                            num_inputs, num_outputs, "classification", [0],
                                                            max_iterations)
            if new_metric > avg_metric:
                avg_metric = new_metric
                learning_rate = learning_rate_value
                hidden_layer_sizes = layer_perm

    return learning_rate, hidden_layer_sizes


def hyperparameter_tune_regression(data_folds, label_folds, test_data, test_labels, learning_rates,
                                   all_num_hidden_layers, all_hidden_layer_sizes, num_inputs, max_iterations,
                                   num_outputs):
    from CrossValidateFunctions import (cross_validate_tune_regression)

    # set up best value variables
    avg_metric = 0.0
    learning_rate = None
    num_hidden_layers = None
    hidden_layer_sizes = None

    for learning_rate_value in learning_rates:
        for num_hidden_layers_value in all_num_hidden_layers:
            layer_perms = list(permutations(all_hidden_layer_sizes, num_hidden_layers_value))
            for layer_perm in layer_perms:
                new_metric = cross_validate_tune_regression(data_folds, label_folds, test_data, test_labels,
                                                            learning_rate, num_hidden_layers_value, layer_perm,
                                                            num_inputs, num_outputs, "regression", [0], max_iterations)
                if new_metric < avg_metric:
                    avg_metric = new_metric
                    learning_rate = learning_rate_value
                    num_hidden_layers = num_hidden_layers_value
                    hidden_layer_sizes = layer_perm

    return learning_rate, num_hidden_layers, hidden_layer_sizes

