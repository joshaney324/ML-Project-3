import numpy as np

from Network import Network
from BreastCancerSet import BreastCancerSet
from Abalone import AbaloneSet
from HelperFunctions import binary_encoding
from Metric_functions import mean_squared_error

network = Network(0.01, 1, [5], 9, 2, "classification", [])

breast = BreastCancerSet()
data = breast.get_data()
labels = breast.get_labels()

labels = labels.reshape(-1, 1)
labels = binary_encoding(labels, [0])

train_data = data[0:600, :]
test_data = data[600:, :]
test_labels = labels[600:, :]
train_labels = labels[0:600, :]


for i in range(10):
    for datapoint, label in zip(train_data, train_labels):
        network.backpropogation(datapoint, label)

predictions = []
for datapoint, label in zip(test_data, test_labels):
    predictions.append(network.feedforward(datapoint)[-1])

for i in range(len(predictions)):
    for j in range(len(predictions[i])):
        if predictions[i][j] > 0.5:
            predictions[i][j] = 1
        else:
            predictions[i][j] = 0

predictions = np.array(predictions)
accuracy = np.mean(predictions == test_labels)

print("breast")
print(accuracy)

network = Network(0.01, 2, [8, 4], 11, 1, "regression", [1])

abalone = AbaloneSet()
data = abalone.get_data()
data = binary_encoding(data, [0])
labels = abalone.get_labels()

labels = labels.reshape(-1, 1)


train_data = data[0:4000, :]
test_data = data[4000:, :]
test_labels = labels[4000:, :]
train_labels = labels[0:4000, :]



for i in range(100):
    for datapoint, label in zip(train_data, train_labels):
        network.backpropogation(datapoint, label)

predictions = []
for datapoint, label in zip(test_data, test_labels):
    predictions.append(network.feedforward(datapoint)[-1])

for i in range(len(predictions)):
    for j in range(len(predictions[i])):
        if predictions[i][j] > 0.5:
            predictions[i][j] = 1
        else:
            predictions[i][j] = 0

predictions = np.array(predictions)
mse = mean_squared_error(predictions, test_labels, len(predictions))

print("abalone")
print(mse)


