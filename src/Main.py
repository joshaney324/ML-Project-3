from BreastCancerSet import BreastCancerSet
from GlassSet import GlassSet
from SoyBeanSet import SoyBeanSet
from Abalone import AbaloneSet
from Hardware import MachineSet
from HelperFunctions import binary_encoding
from Fold_functions import get_tune_folds, get_folds_classification, get_folds_regression
from HyperparameterTune import hyperparameter_tune_classification, hyperparameter_tune_regression
from CrossValidateFunctions import cross_validate_classification, cross_validate_regression
from ForestFires import ForestFiresSet

# The process for testing a dataset will be the same for all. The comments on the first will apply to the rest

# # Soy
#
# print("Soy Bean")
#
# # No layers
#
# # Set up dataset class and collect data and labels
# soy = SoyBeanSet()
# data = soy.get_data()
# labels = soy.get_labels()
# labels = labels.reshape(-1, 1)
# labels = binary_encoding(labels, [0])
#
# # Get folds before getting hold out tune fold
# data_folds, label_folds = get_folds_classification(data, labels, 10)
#
# # Get tuning fold
# test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
#
# # Get folds seperate from the tuning fold
# train_data_folds, train_label_folds = get_folds_classification(train_data, train_labels, 10)
#
# # Tune the hyperparameters of the model
# learning_rate, hidden_layer_sizes = hyperparameter_tune_classification(train_data_folds, train_label_folds, test_data,
#                                                                        test_labels, 0, len(data[0]), len(labels[0]), 10,
#                                                            [0.001, 0.01, 0.1], [])
#
# # Test the model
# print("No Hidden Layers")
# print(str(learning_rate) + " " + str(hidden_layer_sizes))
# print(cross_validate_classification(train_data_folds, train_label_folds, test_data, test_labels, learning_rate, 0,
#                                 hidden_layer_sizes, len(data[0]), len(labels[0]), "classification", [], 10))
#
# # 1 Layer
#
# # reset folds and hyperparameter tune again
# data_folds, label_folds = get_folds_classification(data, labels, 10)
# test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
# train_data_folds, train_label_folds = get_folds_classification(train_data, train_labels, 10)
# learning_rate, hidden_layer_sizes = hyperparameter_tune_classification(train_data_folds, train_label_folds, test_data,
#                                                                        test_labels, 1, len(data[0]), len(labels[0]), 10,
#                                                            [0.001, 0.01, 0.1], [5, 10, 15])
#
# # Test the model
# print("1 Hidden Layer")
# print(str(learning_rate) + " " + str(hidden_layer_sizes))
# print(cross_validate_classification(train_data_folds, train_label_folds, test_data, test_labels, learning_rate, 1,
#                                 hidden_layer_sizes, len(data[0]), len(labels[0]), "classification", hidden_layer_sizes, 10))
#
# # 2 Layer
#
# # reset folds and hyperparameter tune again
# data_folds, label_folds = get_folds_classification(data, labels, 10)
# test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
# train_data_folds, train_label_folds = get_folds_classification(train_data, train_labels, 10)
# learning_rate, hidden_layer_sizes = hyperparameter_tune_classification(train_data_folds, train_label_folds, test_data,
#                                                                        test_labels, 2, len(data[0]), len(labels[0]), 10,
#                                                            [0.001, 0.01, 0.1], [5, 10, 15])
#
# # Test the model
# print("2 Hidden Layers")
# print(str(learning_rate) + " " + str(hidden_layer_sizes))
# print(cross_validate_classification(train_data_folds, train_label_folds, test_data, test_labels, learning_rate, 2,
#                                 hidden_layer_sizes, len(data[0]), len(labels[0]), "classification", hidden_layer_sizes, 10))
#
#
#
#
# # Glass
# print("Glass")
#
# # No layers
#
# glass = GlassSet(7)
# data = glass.get_data()
# labels = glass.get_labels()
# labels = labels.reshape(-1, 1)
# labels = binary_encoding(labels, [0])
#
#
# data_folds, label_folds = get_folds_classification(data, labels, 10)
# test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
# train_data_folds, train_label_folds = get_folds_classification(train_data, train_labels, 10)
# learning_rate, hidden_layer_sizes = hyperparameter_tune_classification(train_data_folds, train_label_folds, test_data,
#                                                                        test_labels, 0, len(data[0]), len(labels[0]), 10,
#                                                            [0.001, 0.01, 0.1], [])
# print("No Hidden Layers")
# print(str(learning_rate) + " " + str(hidden_layer_sizes))
# print(cross_validate_classification(train_data_folds, train_label_folds, test_data, test_labels, learning_rate, 0,
#                                 hidden_layer_sizes, len(data[0]), len(labels[0]), "classification", [], 10))
#
# # 1 Layer
#
# data_folds, label_folds = get_folds_classification(data, labels, 10)
# test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
# train_data_folds, train_label_folds = get_folds_classification(train_data, train_labels, 10)
# learning_rate, hidden_layer_sizes = hyperparameter_tune_classification(train_data_folds, train_label_folds, test_data,
#                                                                        test_labels, 1, len(data[0]), len(labels[0]), 10,
#                                                            [0.001, 0.01, 0.1], [4, 6, 8])
#
# print("1 Hidden Layer")
# print(str(learning_rate) + " " + str(hidden_layer_sizes))
# print(cross_validate_classification(train_data_folds, train_label_folds, test_data, test_labels, learning_rate, 1,
#                                 hidden_layer_sizes, len(data[0]), len(labels[0]), "classification", hidden_layer_sizes, 10))
#
# # 2 Layer
#
# data_folds, label_folds = get_folds_classification(data, labels, 10)
# test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
# train_data_folds, train_label_folds = get_folds_classification(train_data, train_labels, 10)
# learning_rate, hidden_layer_sizes = hyperparameter_tune_classification(train_data_folds, train_label_folds, test_data,
#                                                                        test_labels, 2, len(data[0]), len(labels[0]), 10,
#                                                            [0.001, 0.01, 0.1], [4, 6, 8])
#
# print("2 Hidden Layers")
# print(str(learning_rate) + " " + str(hidden_layer_sizes))
# print(cross_validate_classification(train_data_folds, train_label_folds, test_data, test_labels, learning_rate, 2,
#                                 hidden_layer_sizes, len(data[0]), len(labels[0]), "classification", hidden_layer_sizes, 10))
#
# # Breast
#
# print("Breast Cancer")
#
# # No layers
#
# breast = BreastCancerSet()
# data = breast.get_data()
# labels = breast.get_labels()
# labels = labels.reshape(-1, 1)
# labels = binary_encoding(labels, [0])
#
#
# data_folds, label_folds = get_folds_classification(data, labels, 10)
# test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
# train_data_folds, train_label_folds = get_folds_classification(train_data, train_labels, 10)
# learning_rate, hidden_layer_sizes = hyperparameter_tune_classification(train_data_folds, train_label_folds, test_data,
#                                                                        test_labels, 0, len(data[0]), len(labels[0]), 10,
#                                                            [0.001, 0.01, 0.1], [])
# print("No Hidden Layers")
# print(str(learning_rate) + " " + str(hidden_layer_sizes))
# print(cross_validate_classification(train_data_folds, train_label_folds, test_data, test_labels, learning_rate, 0,
#                                 hidden_layer_sizes, len(data[0]), len(labels[0]), "classification", [], 10))
#
# # 1 Layer
#
# data_folds, label_folds = get_folds_classification(data, labels, 10)
# test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
# train_data_folds, train_label_folds = get_folds_classification(train_data, train_labels, 10)
# learning_rate, hidden_layer_sizes = hyperparameter_tune_classification(train_data_folds, train_label_folds, test_data,
#                                                                        test_labels, 1, len(data[0]), len(labels[0]), 10,
#                                                            [0.001, 0.01, 0.1], [4, 6, 8])
#
# print("1 Hidden Layer")
# print(str(learning_rate) + " " + str(hidden_layer_sizes))
# print(cross_validate_classification(train_data_folds, train_label_folds, test_data, test_labels, learning_rate, 1,
#                                 hidden_layer_sizes, len(data[0]), len(labels[0]), "classification", hidden_layer_sizes, 10))
#
# # 2 Layer
#
# data_folds, label_folds = get_folds_classification(data, labels, 10)
# test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
# train_data_folds, train_label_folds = get_folds_classification(train_data, train_labels, 10)
# learning_rate, hidden_layer_sizes = hyperparameter_tune_classification(train_data_folds, train_label_folds, test_data,
#                                                                        test_labels, 2, len(data[0]), len(labels[0]), 20,
#                                                            [0.001, 0.01, 0.1], [2, 3, 4, 6])
#
# print("2 Hidden Layers")
# print(str(learning_rate) + " " + str(hidden_layer_sizes))
# print(cross_validate_classification(train_data_folds, train_label_folds, test_data, test_labels, learning_rate, 2,
#                                 hidden_layer_sizes, len(data[0]), len(labels[0]), "classification", hidden_layer_sizes, 20))
#

# Machine

print("Machine")

# No layers

machine = MachineSet()
data = machine.get_data()
labels = machine.get_labels()
labels = labels.reshape(-1, 1)

data_folds, label_folds = get_folds_regression(data, labels, 10)
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
train_data_folds, train_label_folds = get_folds_regression(train_data, train_labels, 10)
learning_rate, hidden_layer_sizes = hyperparameter_tune_regression(train_data_folds, train_label_folds, test_data,
                                                                   test_labels, 0, len(data[0]), len(labels[0]), 200,
                                                       [0.001, 0.01, 0.1], [])
print("No Hidden Layers")
print(str(learning_rate) + " " + str(hidden_layer_sizes))
print(cross_validate_regression(train_data_folds, train_label_folds, test_data, test_labels, learning_rate, 0,
                                hidden_layer_sizes, len(data[0]), len(labels[0]), "regression", [], 200))

# 1 Layer

data_folds, label_folds = get_folds_regression(data, labels, 10)
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
train_data_folds, train_label_folds = get_folds_regression(train_data, train_labels, 10)
learning_rate, hidden_layer_sizes = hyperparameter_tune_regression(train_data_folds, train_label_folds, test_data,
                                                                   test_labels, 1, len(data[0]), len(labels[0]), 100,
                                                       [0.001, 0.01, 0.1], [5, 6, 10])

print("1 Hidden Layer")
print(str(learning_rate) + " " + str(hidden_layer_sizes))
print(cross_validate_regression(train_data_folds, train_label_folds, test_data, test_labels, learning_rate, 1,
                                hidden_layer_sizes, len(data[0]), len(labels[0]), "regression", hidden_layer_sizes, 100))

# 2 Layer

data_folds, label_folds = get_folds_regression(data, labels, 10)
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
train_data_folds, train_label_folds = get_folds_regression(train_data, train_labels, 10)
# learning_rate, hidden_layer_sizes = hyperparameter_tune_regression(train_data_folds, train_label_folds, test_data,
#                                                                    test_labels, 2, len(data[0]), len(labels[0]), 100,
#                                                        [0.001, 0.01, 0.1], [5, 6, 10])
print(str(learning_rate) + " " + str(hidden_layer_sizes))
print("2 Hidden Layers")
print(cross_validate_regression(train_data_folds, train_label_folds, test_data, test_labels, 0.01, 2,
                                [10, 10], len(data[0]), len(labels[0]), "regression", [], 100))

# Forest

print("Forest")

# No layers

machine = ForestFiresSet()
data = machine.get_data()
labels = machine.get_labels()
labels = labels.reshape(-1, 1)

data_folds, label_folds = get_folds_regression(data, labels, 10)
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
train_data_folds, train_label_folds = get_folds_regression(train_data, train_labels, 10)
learning_rate, hidden_layer_sizes = hyperparameter_tune_regression(train_data_folds, train_label_folds, test_data,
                                                                   test_labels, 0, len(data[0]), len(labels[0]), 100,
                                                       [0.001, 0.01, 0.1], [])
print("No Hidden Layers")
print(str(learning_rate) + " " + str(hidden_layer_sizes))
print(cross_validate_regression(train_data_folds, train_label_folds, test_data, test_labels, learning_rate, 0,
                                hidden_layer_sizes, len(data[0]), len(labels[0]), "regression", [], 100))

# 1 Layer

data_folds, label_folds = get_folds_regression(data, labels, 10)
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
train_data_folds, train_label_folds = get_folds_regression(train_data, train_labels, 10)
learning_rate, hidden_layer_sizes = hyperparameter_tune_regression(train_data_folds, train_label_folds, test_data,
                                                                   test_labels, 1, len(data[0]), len(labels[0]), 100,
                                                       [0.001, 0.01, 0.1], [5, 10, 15])

print("1 Hidden Layer")
print(str(learning_rate) + " " + str(hidden_layer_sizes))
print(cross_validate_regression(train_data_folds, train_label_folds, test_data, test_labels, learning_rate, 1,
                                hidden_layer_sizes, len(data[0]), len(labels[0]), "regression", hidden_layer_sizes, 100))

# 2 Layer

data_folds, label_folds = get_folds_regression(data, labels, 10)
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
train_data_folds, train_label_folds = get_folds_regression(train_data, train_labels, 10)
learning_rate, hidden_layer_sizes = hyperparameter_tune_regression(train_data_folds, train_label_folds, test_data,
                                                                   test_labels, 2, len(data[0]), len(labels[0]), 100,
                                                       [0.001, 0.01, 0.1], [5, 10, 15])
print(str(learning_rate) + " " + str(hidden_layer_sizes))
print("2 Hidden Layers")
print(cross_validate_regression(train_data_folds, train_label_folds, test_data, test_labels, learning_rate, 2,
                                hidden_layer_sizes, len(data[0]), len(labels[0]), "regression", hidden_layer_sizes, 100))

print("Abalone")

# No layers

machine = AbaloneSet()
data = machine.get_data()
labels = machine.get_labels()
labels = labels.reshape(-1, 1)

data_folds, label_folds = get_folds_regression(data, labels, 10)
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
train_data_folds, train_label_folds = get_folds_regression(train_data, train_labels, 10)
learning_rate, hidden_layer_sizes = hyperparameter_tune_regression(train_data_folds, train_label_folds, test_data,
                                                                   test_labels, 0, len(data[0]), len(labels[0]), 100,
                                                       [0.001, 0.01, 0.1], [])
print("No Hidden Layers")
print(str(learning_rate) + " " + str(hidden_layer_sizes))
print(cross_validate_regression(train_data_folds, train_label_folds, test_data, test_labels, learning_rate, 0,
                                hidden_layer_sizes, len(data[0]), len(labels[0]), "regression", [], 100))

# 1 Layer

data_folds, label_folds = get_folds_regression(data, labels, 10)
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
train_data_folds, train_label_folds = get_folds_regression(train_data, train_labels, 10)
learning_rate, hidden_layer_sizes = hyperparameter_tune_regression(train_data_folds, train_label_folds, test_data,
                                                                   test_labels, 1, len(data[0]), len(labels[0]), 100,
                                                       [0.001, 0.01, 0.1], [5, 10, 15])

print("1 Hidden Layer")
print(str(learning_rate) + " " + str(hidden_layer_sizes))
print(cross_validate_regression(train_data_folds, train_label_folds, test_data, test_labels, learning_rate, 1,
                                hidden_layer_sizes, len(data[0]), len(labels[0]), "regression", hidden_layer_sizes, 100))

# 2 Layer

data_folds, label_folds = get_folds_regression(data, labels, 10)
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
train_data_folds, train_label_folds = get_folds_regression(train_data, train_labels, 10)
learning_rate, hidden_layer_sizes = hyperparameter_tune_regression(train_data_folds, train_label_folds, test_data,
                                                                   test_labels, 2, len(data[0]), len(labels[0]), 100,
                                                       [0.001, 0.01, 0.1], [5, 10, 15])
print(str(learning_rate) + " " + str(hidden_layer_sizes))
print("2 Hidden Layers")
print(cross_validate_regression(train_data_folds, train_label_folds, test_data, test_labels, learning_rate, 2,
                                hidden_layer_sizes, len(data[0]), len(labels[0]), "regression", hidden_layer_sizes, 10))







print("test")

