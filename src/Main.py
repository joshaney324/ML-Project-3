import numpy as np

from Network import Network
from BreastCancerSet import BreastCancerSet
from GlassSet import GlassSet
from Abalone import AbaloneSet
from HelperFunctions import binary_encoding
from Metric_functions import mean_squared_error

network = Network(0.01, 2, [2, 2], 2, 2, "classification", [])

data = [[0, 0], [0, 1], [1, 0], [1, 1]]
data = np.array(data)
labels = [[0], [1], [1], [1]]
labels = np.array(labels)
labels = binary_encoding(labels, [0])

for i in range(500):
    for datapoint, label in zip(data, labels):
        network.backpropogation(datapoint, label)

prediction = network.feedforward([0, 1])[-1]

prediction2 = network.feedforward([1, 0])[-1]

prediction3 = network.feedforward([1, 1])[-1]

prediction4 = network.feedforward([0, 0])[-1]

# breast = BreastCancerSet()
# data = breast.get_data()
# labels = breast.get_labels()
# labels = labels.reshape(-1, 1)
# labels = binary_encoding(labels, [0])
#
# breast_net = Network(0.01, 1, [4], 9, 2, "classification", [])
#
# for i in range(100):
#     for datapoint, label in zip(data, labels):
#         breast_net.backpropogation(datapoint, label)
#
# predictions = []
# for datapoint in data:
#     predictions.append(breast_net.feedforward(datapoint)[-1])
#
# for i in range(len(predictions)):
#     for j in range(len(predictions[i])):
#         if predictions[i][j] > .5:
#             predictions[i][j] = 1
#         else:
#             predictions[i][j] = 0
#
# predictions = np.array(predictions)
# print(predictions - labels)
#
#
#
#

glass = GlassSet(7)
data = glass.get_data()
labels = glass.get_labels()
labels = labels.reshape(-1, 1)
labels = binary_encoding(labels, [0])
#
# perm = np.random.permutation(labels.shape[1])
# labels = labels[:, perm]

breast_net = Network(0.01, 0, [], 9, 6, "classification", [])

for i in range(50):
    for datapoint, label in zip(data, labels):
        breast_net.backpropogation(datapoint, label)

predictions = []
for datapoint in data:
    predictions.append(breast_net.feedforward(datapoint)[-1])

predictions = np.array(predictions)

for i in range(len(predictions)):
    pred_max = np.max(predictions[i])
    for j in range(len(predictions[i])):

        if predictions[i][j] == pred_max:
            predictions[i][j] = 1
        else:
            predictions[i][j] = 0

predictions = np.array(predictions)
print(predictions - labels)


# abalone = AbaloneSet()
# data = abalone.get_data()
# labels = abalone.get_labels()
# labels = labels.reshape(-1, 1)
#
# abalone_net = Network(0.01, 0, [], 10, 1, "regression", [])
#
# for i in range(100):
#     for datapoint, label in zip(data, labels):
#         abalone_net.backpropogation(datapoint, label)
#
# predictions = []
#
# for datapoint in data:
#     predictions.append(abalone_net.feedforward(datapoint))
#
# print(mean_squared_error(predictions, labels, len(predictions)))



print("test")

