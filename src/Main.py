import numpy as np

from Network import Network
from BreastCancerSet import BreastCancerSet
from HelperFunctions import binary_encoding

network = Network(0.01, 0, [], 9, 2, "classification", [])

breast = BreastCancerSet()
data = breast.get_data()
labels = breast.get_labels()

labels = labels.reshape(-1, 1)
labels = binary_encoding(labels, [0])


for i in range(15):
    for datapoint, label in zip(data, labels):
        network.backpropogation(datapoint, label)

predictions = []
for datapoint, label in zip(data, labels):
    predictions.append(network.feedforward(datapoint)[-1])

for i in range(len(predictions)):
    for j in range(len(predictions[i])):
        if predictions[i][j] > 0.5:
            predictions[i][j] = 1
        else:
            predictions[i][j] = 0

predictions = np.array(predictions)
accuracy = np.mean(predictions == labels)

print(accuracy)
