from HelperFunctions import binary_encoding
import numpy as np
from Network import Network
from CrossValidateFunctions import cross_validate_classification, cross_validate_regression
from Fold_functions import get_folds_classification, get_folds_regression, get_tune_folds
from Hardware import MachineSet
from GlassSet import GlassSet
from Metric_functions import accuracy
from src.Abalone import AbaloneSet
from src.Main import data_folds, label_folds
from src.Metric_functions import mean_squared_error

# classification

print("Glass")

glass = GlassSet(7)
data = glass.get_data()
labels = glass.get_labels()
labels = labels.reshape(-1, 1)
labels = binary_encoding(labels, [0])

data_folds, label_folds = get_folds_classification(data, labels, 10)
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
data_folds, label_folds = get_folds_classification(train_data, train_labels, 10)

fold_1_train_data = []
fold_1_train_labels = []
fold_1_test_data = []
fold_1_test_labels = []
for i in range(len(data_folds) - 1):
    for instance, label in zip(data_folds[i], label_folds[i]):
        fold_1_train_data.append(instance)
        fold_1_train_labels.append(label)

for instance, label in zip(data_folds[-1], label_folds[-1]):
    fold_1_test_data.append(instance)
    fold_1_test_labels.append(label)

fold_1_train_data = np.array(fold_1_train_data)
fold_1_train_labels = np.array(fold_1_train_labels)
fold_1_test_data = np.array(fold_1_test_data)
fold_1_test_labels = np.array(fold_1_test_labels)

# 0 hidden layers

network = Network(0.01, 0, [], len(fold_1_train_data[0]), len(fold_1_test_data[0]), "classification", [])
network.train(fold_1_train_data, fold_1_train_labels, test_data, test_labels, 1)

predictions = []
for test_point in fold_1_test_data:
    predictions.append(network.predict(test_point))

print("Glass performance no hidden layers")
print(accuracy(predictions, test_labels))



# 1 hidden layers

network = Network(0.01, 1, [6], len(fold_1_train_data[0]), len(fold_1_test_data[0]), "classification", [])
network.train(fold_1_train_data, fold_1_train_labels, test_data, test_labels, 1)

predictions = []
for test_point in fold_1_test_data:
    predictions.append(network.predict(test_point))

print("Glass performance no hidden layers")
print(accuracy(predictions, test_labels))

# 2 hidden layer

network = Network(0.01, 2, [6, 6], len(fold_1_train_data[0]), len(fold_1_test_data[0]), "classification", [])
network.train(fold_1_train_data, fold_1_train_labels, test_data, test_labels, 1)

predictions = []
for test_point in fold_1_test_data:
    predictions.append(network.predict(test_point))

print("Glass performance no hidden layers")
print(accuracy(predictions, test_labels))

# Regression

abalone = AbaloneSet()
data = abalone.get_data()
labels = abalone.get_labels()
labels = labels.reshape(-1, 1)


data_folds, label_folds = get_folds_regression(data, labels, 10)
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
data_folds, label_folds = get_folds_regression(train_data, train_labels, 10)

fold_1_train_data = []
fold_1_train_labels = []
fold_1_test_data = []
fold_1_test_labels = []
for i in range(len(data_folds) - 1):
    for instance, label in zip(data_folds[i], label_folds[i]):
        fold_1_train_data.append(instance)
        fold_1_train_labels.append(label)

for instance, label in zip(data_folds[-1], label_folds[-1]):
    fold_1_test_data.append(instance)
    fold_1_test_labels.append(label)

fold_1_train_data = np.array(fold_1_train_data)
fold_1_train_labels = np.array(fold_1_train_labels)
fold_1_test_data = np.array(fold_1_test_data)
fold_1_test_labels = np.array(fold_1_test_labels)

# 0 hidden layers

network = Network(0.01, 0, [], len(fold_1_train_data[0]), len(fold_1_test_data[0]), "regression", [])
network.train(fold_1_train_data, fold_1_train_labels, test_data, test_labels, 1)

predictions = []
for test_point in fold_1_test_data:
    predictions.append(network.predict(test_point))

print("Abalone performance no hidden layers")
print(mean_squared_error(predictions, test_labels, len(predictions)))

# 1 hidden layers

network = Network(0.01, 1, [6], len(fold_1_train_data[0]), len(fold_1_test_data[0]), "regression", [])
network.train(fold_1_train_data, fold_1_train_labels, test_data, test_labels, 1)

predictions = []
for test_point in fold_1_test_data:
    predictions.append(network.predict(test_point))

print("Abalone performance 1 hidden layer")
print(mean_squared_error(predictions, test_labels, len(predictions)))

# 2 hidden layer

network = Network(0.01, 2, [6, 6], len(fold_1_train_data[0]), len(fold_1_test_data[0]), "regression", [])
network.train(fold_1_train_data, fold_1_train_labels, test_data, test_labels, 1)

predictions = []
for test_point in fold_1_test_data:
    predictions.append(network.predict(test_point))

print("Abalone performance 2 hidden layers")
print(mean_squared_error(predictions, test_labels, len(predictions)))

# print("Glass Metrics")
# print(cross_validate_classification(data_folds, label_folds, test_data, test_labels, 0.01, 0, [], len(data[0]), len(labels[0]), "classification", [], 1))
# print(cross_validate_classification(data_folds, label_folds, test_data, test_labels, 0.01, 1, [6], len(data[0]), len(labels[0]), "classification", [], 1))
# print(cross_validate_classification(data_folds, label_folds, test_data, test_labels, 0.01, 2, [6, 6], len(data[0]), len(labels[0]), "classification", [], 1))