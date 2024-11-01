import numpy as np
from Network import Network
from Metric_functions import precision, recall, accuracy, mean_squared_error
from Fold_functions import get_folds_classification, get_folds_regression


def cross_validate_classification(data_folds, label_folds, learning_rate, num_hidden_layers, hidden_layer_sizes,
                                  num_inputs, num_outputs, output_type, biased_layers, max_iterations):
    # the cross_validate function is meant to get the precision, recall and accuracy values from each fold then print
    # out the average across folds. this function takes in a list of data folds and a list of label folds. it does not
    # return anything but prints out the metrics

    # Set up variables
    precision_avg = 0.0
    recall_avg = 0.0
    accuracy_avg = 0.0
    folds = len(data_folds)
    matrix_total = np.zeros((2, 2))
    accuracies = []
    network = Network(learning_rate, num_hidden_layers, hidden_layer_sizes, num_inputs, num_outputs, output_type,
                      biased_layers)

    # For each testing fold, set up a training and testing set and then append the loss function values
    for i in range(len(data_folds)):
        train_data = []
        test_data = []
        train_labels = []
        test_labels = []
        for j in range(len(data_folds)):
            if i != j:
                for instance, label in zip(data_folds[j], label_folds[j]):
                    train_data.append(instance)
                    train_labels.append(label)
            else:
                for instance, label in zip(data_folds[j], label_folds[j]):
                    test_data.append(instance)
                    test_labels.append(label)

        # make all the data into np arrays so that naive bayes class can use them
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        test_data = np.array(test_data)
        test_labels = np.array(test_labels)

        network.train(train_data, train_labels, max_iterations)
        predictions = []
        for datapoint in test_data:
            predictions.append(network.predict(datapoint))

        precision_vals = np.array(precision(predictions, test_labels))
        recall_vals = np.array(recall(predictions, test_labels))
        accuracy_vals, matrix = accuracy(predictions, test_labels)
        accuracy_vals = np.array(accuracy_vals)

        precision_total = 0
        recall_total = 0
        accuracy_total = 0
        counter = 0

        # get the averages of all the precision, recall, and accuracy values from all the folds
        for precision_val, recall_val, accuracy_val in zip(precision_vals, recall_vals, accuracy_vals):
            precision_total += float(precision_val[1])
            recall_total += float(recall_val[1])
            accuracies.append(float(accuracy_val[1]))
            accuracy_total += float(accuracy_val[1])
            matrix_total = matrix_total + np.array(matrix)
            counter += 1

        precision_avg += precision_total / counter
        recall_avg += recall_total / counter
        accuracy_avg += accuracy_total / counter

    return [precision_avg / folds, recall_avg / folds, accuracy_avg / folds]


def cross_validate_regression(data_folds, label_folds, learning_rate, num_hidden_layers, hidden_layer_sizes, num_inputs,
                              num_outputs, output_type, biased_layers, max_iterations):

    # This function is meant to cross validate the regression sets it returns a mean squared error

    # Set up variables
    mean_squared_error_avg = 0.0
    folds = len(data_folds)
    network = Network(learning_rate, num_hidden_layers, hidden_layer_sizes, num_inputs, num_outputs, output_type,
                      biased_layers)

    # For each testing fold, set up a training and testing set and then append the loss function values
    for i in range(len(data_folds)):
        train_data = []
        test_data = []
        train_labels = []
        test_labels = []
        for j in range(len(data_folds)):
            if i != j:
                for instance, label in zip(data_folds[j], label_folds[j]):
                    train_data.append(instance)
                    train_labels.append(label)
            else:
                for instance, label in zip(data_folds[j], label_folds[j]):
                    test_data.append(instance)
                    test_labels.append(label)

        # make all the data into np arrays so that naive bayes class can use them
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        test_data = np.array(test_data)
        test_labels = np.array(test_labels)

        network.train(train_data, train_labels, max_iterations)
        predictions = []
        for datapoint in test_data:
            predictions.append(network.predict(datapoint))

        mean_squared_error_avg += mean_squared_error(test_labels, predictions, len(predictions))

    return mean_squared_error_avg / folds


def cross_validate_tune_regression(data_folds, label_folds, test_data, test_labels, learning_rate, num_hidden_layers,
                                   hidden_layer_sizes, num_inputs, num_outputs, output_type, biased_layers, max_iterations):

    # This function is meant to cross validate the regression sets but leave out the testing folds and use the tune
    # folds instead

    # Set up variables
    mean_squared_error_avg = 0.0
    folds = len(data_folds)
    network = Network(learning_rate, num_hidden_layers, hidden_layer_sizes, num_inputs, num_outputs, output_type,
                      biased_layers)

    # For each testing fold, set up a training and testing set and then append the loss function values
    for i in range(len(data_folds)):
        train_data = []
        train_labels = []
        for j in range(len(data_folds)):
            if i != j:
                for instance, label in zip(data_folds[j], label_folds[j]):
                    train_data.append(instance)
                    train_labels.append(label)

        # make all the data into np arrays and set up the test data and labels as the hold out fold
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        test_data = np.array(test_data)
        test_labels = np.array(test_labels)

        network.train(train_data, train_labels, max_iterations)

        # get predictions and append them
        predictions = []
        for datapoint in test_data:
            predictions.append(network.predict(datapoint))

        mean_squared_error_avg += mean_squared_error(test_labels, predictions, len(predictions))

    return mean_squared_error_avg / folds


def cross_validate_tune_classification(data_folds, label_folds, test_data, test_labels, learning_rate,
                                       num_hidden_layers, hidden_layer_sizes, num_inputs, num_outputs, output_type,
                                       biased_layers, max_iterations):

    # This function is meant to cross validate the classification sets but leave out the testing folds and use the tune
    # folds instead

    # Set up variables
    precision_avg = 0.0
    recall_avg = 0.0
    accuracy_avg = 0.0
    folds = len(data_folds)
    matrix_total = np.zeros((2,2))
    accuracies = []
    network = Network(learning_rate, num_hidden_layers, hidden_layer_sizes, num_inputs, num_outputs, output_type,
                      biased_layers)

    # For each testing fold, set up a training and testing set and then append the loss function values
    for i in range(len(data_folds)):
        train_data = []
        train_labels = []
        for j in range(len(data_folds)):
            if i != j:
                for instance, label in zip(data_folds[j], label_folds[j]):
                    train_data.append(instance)
                    train_labels.append(label)

        # make all the data into np arrays and set the test data and labels as the hold out fold
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        test_data = np.array(test_data)
        test_labels = np.array(test_labels)

        network.train(train_data, train_labels, max_iterations)

        predictions = []

        # Get all predictions
        for datapoint in test_data:
            predictions.append(network.predict(datapoint))

        # Get all precision recall and accuracy vals
        precision_vals = np.array(precision(predictions, test_labels))
        recall_vals = np.array(recall(predictions, test_labels))
        accuracy_vals, matrix = accuracy(predictions, test_labels)
        accuracy_vals = np.array(accuracy_vals)

        precision_total = 0
        recall_total = 0
        accuracy_total = 0
        counter = 0

        # get the averages of all the precision, recall, and accuracy values from all the folds
        for precision_val, recall_val, accuracy_val in zip(precision_vals, recall_vals, accuracy_vals):
            precision_total += float(precision_val[1])
            recall_total += float(recall_val[1])
            accuracies.append(float(accuracy_val[1]))
            accuracy_total += float(accuracy_val[1])
            matrix_total = matrix_total + np.array(matrix)
            counter += 1

        precision_avg += precision_total / counter
        recall_avg += recall_total / counter
        accuracy_avg += accuracy_total / counter

    return (precision_avg / folds + recall_avg / folds + accuracy_avg / folds) / 3
