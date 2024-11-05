from HelperFunctions import binary_encoding
import numpy as np
from Network import Network
from CrossValidateFunctions import cross_validate_classification, cross_validate_regression
from Fold_functions import get_folds_classification, get_folds_regression, get_tune_folds
from GlassSet import GlassSet
from Metric_functions import accuracy
from Abalone import AbaloneSet
from Metric_functions import mean_squared_error

#-------------------------------------------------------------------------------------------------------------------

# Performance across 1 fold

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

network = Network(0.01, 0, [], len(fold_1_train_data[0]), len(fold_1_train_labels[0]), "classification", [])
network.train(fold_1_train_data, fold_1_train_labels, test_data, test_labels, 5)

predictions = []
for test_point in fold_1_test_data:
    predictions.append(network.predict(test_point))

print("Glass performance no hidden layers")
accuracies, matrix = accuracy(predictions, test_labels)
avg_accuracy = 0.0
for i in range(len(accuracies)):
    avg_accuracy += accuracies[i][1]

print("Accuracy: ")
print(avg_accuracy / len(accuracies))




# 1 hidden layers

network = Network(0.01, 1, [6], len(fold_1_train_data[0]), len(fold_1_train_labels[0]), "classification", [])
network.train(fold_1_train_data, fold_1_train_labels, test_data, test_labels, 5)

predictions = []
for test_point in fold_1_test_data:
    predictions.append(network.predict(test_point))

print("Glass performance 1 hidden layer")
accuracies, matrix = accuracy(predictions, test_labels)
avg_accuracy = 0.0
for i in range(len(accuracies)):
    avg_accuracy += accuracies[i][1]

print("Accuracy: ")
print(avg_accuracy / len(accuracies))


# 2 hidden layer

network = Network(0.05, 2, [5, 4], len(fold_1_train_data[0]), len(fold_1_train_labels[0]), "classification", [])
network.train(fold_1_train_data, fold_1_train_labels, test_data, test_labels, 5)

predictions = []
for test_point in fold_1_test_data:
    predictions.append(network.predict(test_point))

print("Glass performance 2 hidden layers")
print("Accuracy: ")
accuracies, matrix = accuracy(predictions, test_labels)
avg_accuracy = 0.0
for i in range(len(accuracies)):
    avg_accuracy += accuracies[i][1]


print(avg_accuracy / len(accuracies))

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

network = Network(0.1, 0, [], len(fold_1_train_data[0]), 1, "regression", [])
network.train(fold_1_train_data, fold_1_train_labels, test_data, test_labels, 1)

predictions = []
for test_point in fold_1_test_data:
    predictions.append(network.predict(test_point))

print("Abalone performance no hidden layers")
print("Mean Squared Error: ")
print(mean_squared_error(predictions, test_labels, len(predictions)))

# 1 hidden layers

network = Network(0.1, 1, [5], len(fold_1_train_data[0]), 1, "regression", [])
network.train(fold_1_train_data, fold_1_train_labels, test_data, test_labels, 1)

predictions = []
for test_point in fold_1_test_data:
    predictions.append(network.predict(test_point))

print("Abalone performance 1 hidden layer")
print("Mean Squared Error: ")
print(mean_squared_error(predictions, test_labels, len(predictions)))

# 2 hidden layer

network = Network(0.1, 2, [5, 3], len(fold_1_train_data[0]), 1, "regression", [])
network.train(fold_1_train_data, fold_1_train_labels, test_data, test_labels, 1)

predictions = []
for test_point in fold_1_test_data:
    predictions.append(network.predict(test_point))

print("Abalone performance 2 hidden layers")
print("Mean Squared Error: ")
print(mean_squared_error(predictions, test_labels, len(predictions)))

# -----------------------------------------------------------------------------------------------------------------

# show weight matrices and inputs/outputs of smallest network
# show gradient at output layer

# classification

glass = GlassSet(7)
data = glass.get_data()
labels = glass.get_labels()
labels = labels.reshape(-1, 1)
labels = binary_encoding(labels, [0])

network = Network(0.01, 0, [], len(data[0]), len(labels[0]), "classification", [])

network.backpropogation(data[0], labels[0])

# regression

abalone = AbaloneSet()
data = abalone.get_data()
labels = abalone.get_labels()
labels = labels.reshape(-1, 1)

network = Network(0.1, 0, [], len(data[0]), 1, "regression", [])
network.backpropogation(data[0], labels[0])

#---------------------------------------------------------------------------------------------------------------------

# propagation of 2 layer network
# show weight updates

abalone = AbaloneSet()
data = abalone.get_data()
labels = abalone.get_labels()
labels = labels.reshape(-1, 1)

network = Network(0.1, 2, [2, 2], len(data[0]), 1, "regression", [])
network.backpropogation(data[0], labels[0])


#--------------------------------------------------------------------------------------------------------------------

# Show performance across 10 folds

# Classification

glass = GlassSet(7)
data = glass.get_data()
labels = glass.get_labels()
labels = labels.reshape(-1, 1)
labels = binary_encoding(labels, [0])

data_folds, label_folds = get_folds_classification(data, labels, 10)
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
data_folds, label_folds = get_folds_classification(train_data, train_labels, 10)

print("Glass Metrics")
print("no hidden")
print(cross_validate_classification(data_folds, label_folds, test_data, test_labels, 0.01, 0, [], len(data[0]), len(labels[0]), "classification", [], 1))
print("one hidden")
print(cross_validate_classification(data_folds, label_folds, test_data, test_labels, 0.01, 1, [6], len(data[0]), len(labels[0]), "classification", [], 1))
print("two hidden")
print(cross_validate_classification(data_folds, label_folds, test_data, test_labels, 0.01, 2, [6, 6], len(data[0]), len(labels[0]), "classification", [], 1))

# Regression

abalone = AbaloneSet()
data = abalone.get_data()
labels = abalone.get_labels()
labels = labels.reshape(-1, 1)


data_folds, label_folds = get_folds_regression(data, labels, 10)
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
data_folds, label_folds = get_folds_regression(train_data, train_labels, 10)

print("Abalone mse:")
print("no hidden")
print(cross_validate_regression(data_folds, label_folds, test_data, test_labels, 0.01, 0, [], len(data[0]), len(labels[0]), "regression", [], 1))
print("1 hidden")
print(cross_validate_regression(data_folds, label_folds, test_data, test_labels, 0.01, 1, [6], len(data[0]), len(labels[0]), "regression", [], 1))
print("2 hidden")
print(cross_validate_regression(data_folds, label_folds, test_data, test_labels, 0.01, 2, [6, 6], len(data[0]), len(labels[0]), "regression", [], 1))
