from ActivationFunction import ActivationFunction
import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))


# This function takes in the errors of the next layer and the outgoing weights from this node to that layer and
# calculates the error for this node
def node_error_calc(errors_next_layer, weights_from_next_layer):
    errors_next_layer = np.array(errors_next_layer)
    weights_from_next_layer = np.array(weights_from_next_layer)

    error = np.dot(errors_next_layer, weights_from_next_layer)

    return error


class Node:
    def __init__(self, weights, isBias):
        self.weights = weights
        self.isBias = isBias

    def feedforward(self, inputs):
        if self.isBias:
            output = 1
            return output
        else:
            output = np.dot(inputs, self.weights)
            return sigmoid(output)

    def update(self, weight_changes):
        for i, weight in enumerate(self.weights):
            self.weights[i] = weight + weight_changes[i]