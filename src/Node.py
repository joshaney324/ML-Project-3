from ActivationFunction import ActivationFunction
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def node_error_calc(errors_next_layer, weights_from_next_layer):
    error = np.dot(errors_next_layer, weights_from_next_layer)
    return error

class Node:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        output = np.dot(inputs, self.weights) + self.bias
        return sigmoid(output)

