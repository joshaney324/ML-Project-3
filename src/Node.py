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
            return np.dot(inputs, self.weights)

    def get_updates(self, learning_rate, node_error, node_val, inputs, is_output):
        node_weight_updates = []
        if not is_output:
            for j in range(len(self.weights)):
                weight_update = learning_rate * node_error * sigmoid_derivative(node_val) * inputs[j]
                node_weight_updates.append(weight_update)
        if is_output:
            for j in range(len(self.weights)):
                weight_update = learning_rate * node_error * inputs[j]
                node_weight_updates.append(weight_update)
        return node_weight_updates

    def update(self, weight_changes):
        for i, weight in enumerate(self.weights):
            self.weights[i] = weight + weight_changes[i]