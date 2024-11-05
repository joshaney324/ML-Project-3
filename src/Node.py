import numpy as np


# This function takes in a node value x and returns the "activated" value according to the sigmoid activation function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# This function takes in a node value x and returns the derivative of the sigmoid activation function at that node value
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

    # This function takes in the inputs (either the last layer's node values or the network inputs) and returns the
    # weighted sum of those inputs (the unactivated node value for this node)
    def feedforward(self, inputs):
        if self.isBias:
            output = 1
            return output
        else:
            return np.dot(inputs, self.weights)

    # This function takes in the learning rate, this node's error and value, the inputs, and a boolean dictating whether
    # this is an output node. It returns a list of weight updates for each of the node's incoming weights
    def get_updates(self, learning_rate, node_error, node_val, inputs, is_output):
        node_weight_updates = []
        # If this node is not an output node, calculate the update for each incoming weight using the derivative of the
        # sigmoid activation function.
        if not is_output:
            for j in range(len(self.weights)):
                weight_update = learning_rate * node_error * sigmoid_derivative(node_val) * inputs[j]
                node_weight_updates.append(weight_update)
        # If this node is an output node, calculate the update for each incoming weight using the derivative of the
        # linear or softmax activation function (this is just 1 for both functions at all node values)
        else:
            for j in range(len(self.weights)):
                weight_update = learning_rate * node_error * inputs[j]
                node_weight_updates.append(weight_update)
        return node_weight_updates

    # Given an array of weight updates, update the weights of this node
    def update(self, weight_changes):
        for i, weight in enumerate(self.weights):
            self.weights[i] = weight + weight_changes[i]