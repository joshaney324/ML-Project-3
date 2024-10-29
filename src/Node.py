from ActivationFunction import ActivationFunction
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Node:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        output = np.dot(inputs, self.weights) + self.bias
        return sigmoid(output)


