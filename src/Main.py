import numpy as np
from BreastCancerSet import BreastCancerSet
from GlassSet import GlassSet
from SoyBeanSet import SoyBeanSet
from Abalone import AbaloneSet
from HelperFunctions import binary_encoding
from Fold_functions import get_tune_folds, get_folds_classification
from HyperparameterTune import hyperparameter_tune_classification
from CrossValidateFunctions import cross_validate_classification
from src.Network import Network
from Metric_functions import mean_squared_error

# network = Network(0.01, 2, [2, 2], 2, 2, "classification", [])
#
# data = [[0, 0], [0, 1], [1, 0], [1, 1]]
# data = np.array(data)
# labels = [[0], [1], [1], [1]]
# labels = np.array(labels)
# labels = binary_encoding(labels, [0])
#
# for i in range(500):
#     for datapoint, label in zip(data, labels):
#         network.backpropogation(datapoint, label)
#
# prediction = network.feedforward([0, 1])[-1]
#
# prediction2 = network.feedforward([1, 0])[-1]
#
# prediction3 = network.feedforward([1, 1])[-1]
#
# prediction4 = network.feedforward([0, 0])[-1]

abalone = AbaloneSet()
data = abalone.get_data()
labels = abalone.get_labels()
labels = labels.reshape(-1, 1)
network = Network(0.01, 0, [], 10, 1, "regression", [])
predictions = []
for i in range(100):
    for datum, label in zip(data, labels):
        network.backpropogation(datum, label)
for i in range(len(data)):
    layers = network.feedforward(data[i])
    prediction = layers[-1][0]
    predictions.append(prediction)

print(mean_squared_error(labels, predictions, len(predictions)))

breast = BreastCancerSet()
data = breast.get_data()
labels = breast.get_labels()
labels = labels.reshape(-1, 1)
labels = binary_encoding(labels, [0])

print("Breast Cancer without hidden Layers")

data_folds, label_folds = get_folds_classification(data, labels, 10)
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
train_data_folds, train_label_folds = get_folds_classification(train_data, train_labels, 10)
learning_rate, hidden_layer_sizes = hyperparameter_tune_classification(train_data_folds, train_label_folds, test_data,
                                                                       test_labels, 0, 9, 2, 5,
                                                                       [0.001, 0.005, 0.01, 0.1], [])
print(str(learning_rate) + " " + str(hidden_layer_sizes))
print(cross_validate_classification(train_data_folds, train_label_folds, test_data, test_labels, learning_rate, 0,
                                    hidden_layer_sizes, 9, 2, "classification", [], 10))

print("Breast Cancer with hidden Layers")

data_folds, label_folds = get_folds_classification(data, labels, 10)
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
train_data_folds, train_label_folds = get_folds_classification(train_data, train_labels, 10)
learning_rate, hidden_layer_sizes = hyperparameter_tune_classification(train_data_folds, train_label_folds, test_data,
                                                                       test_labels, 1, 9, 2, 5,
                                                                       [0.001, 0.005, 0.01, 0.1], [1, 2, 3, 4, 5, 6])
print(str(learning_rate) + " " + str(hidden_layer_sizes))
print(cross_validate_classification(train_data_folds, train_label_folds, test_data, test_labels, learning_rate, 1,
                                    hidden_layer_sizes, 9, 2, "classification", [], 10))

breast = GlassSet(7)
data = breast.get_data()
labels = breast.get_labels()
labels = labels.reshape(-1, 1)
labels = binary_encoding(labels, [0])

print("Glass without hidden Layers")

data_folds, label_folds = get_folds_classification(data, labels, 10)
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
train_data_folds, train_label_folds = get_folds_classification(train_data, train_labels, 10)
learning_rate, hidden_layer_sizes = hyperparameter_tune_classification(train_data_folds, train_label_folds, test_data,
                                                                       test_labels, 0, 9, 6, 5,
                                                                       [0.001, 0.005, 0.01, 0.1], [])
print(str(learning_rate) + " " + str(hidden_layer_sizes))
print(cross_validate_classification(train_data_folds, train_label_folds, test_data, test_labels, learning_rate, 0,
                                    hidden_layer_sizes, 9, 6, "classification", [], 10))

print("Glass with hidden Layers")

data_folds, label_folds = get_folds_classification(data, labels, 10)
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
train_data_folds, train_label_folds = get_folds_classification(train_data, train_labels, 10)
learning_rate, hidden_layer_sizes = hyperparameter_tune_classification(train_data_folds, train_label_folds, test_data,
                                                                       test_labels, 1, 9, 6, 5,
                                                                       [0.001, 0.005, 0.01, 0.1], [5, 7, 8])
print(str(learning_rate) + " " + str(hidden_layer_sizes))
print(cross_validate_classification(train_data_folds, train_label_folds, test_data, test_labels, learning_rate, 1,
                                    hidden_layer_sizes, 9, 6, "classification", [], 10))

breast = SoyBeanSet()
data = breast.get_data()
labels = breast.get_labels()
labels = labels.reshape(-1, 1)
labels = binary_encoding(labels, [0])

print("Soy without hidden Layers")

data_folds, label_folds = get_folds_classification(data, labels, 10)
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
train_data_folds, train_label_folds = get_folds_classification(train_data, train_labels, 10)
learning_rate, hidden_layer_sizes = hyperparameter_tune_classification(train_data_folds, train_label_folds, test_data,
                                                                       test_labels, 0, 21, 4, 5,
                                                                       [0.001, 0.005, 0.01, 0.1], [])
print(str(learning_rate) + " " + str(hidden_layer_sizes))
print(cross_validate_classification(train_data_folds, train_label_folds, test_data, test_labels, learning_rate, 0,
                                    hidden_layer_sizes, 21, 4, "classification", [], 10))

print("Soy with hidden Layers")

data_folds, label_folds = get_folds_classification(data, labels, 10)
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
train_data_folds, train_label_folds = get_folds_classification(train_data, train_labels, 10)
learning_rate, hidden_layer_sizes = hyperparameter_tune_classification(train_data_folds, train_label_folds, test_data,
                                                                       test_labels, 1, 21, 4, 5,
                                                                       [0.001, 0.005, 0.01, 0.1], [1, 2, 3, 4, 5, 10, 15])
print(str(learning_rate) + " " + str(hidden_layer_sizes))
print(cross_validate_classification(train_data_folds, train_label_folds, test_data, test_labels, learning_rate, 1,
                                    hidden_layer_sizes, 21, 4, "classification", [], 10))



print("test")

