import numpy as np

from Network import Network
from BreastCancerSet import BreastCancerSet
from Abalone import AbaloneSet
from HelperFunctions import binary_encoding
from Metric_functions import mean_squared_error

network = Network(0.01, 1, [5], 2, 1, "regression", [])

data = [[0, 0], [0, 1], [1, 0], [1, 1]]
data = np.array(data)
labels = [[0], [1], [1], [1]]
labels = np.array(labels)

for datapoint, label in zip(data, labels):
    network.backpropogation(datapoint, label)

prediction = network.feedforward([0, 1])[-1]

print("test")

