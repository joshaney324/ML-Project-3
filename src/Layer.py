import numpy as np
from Node import Node, node_error_calc


class Layer:
    def __init__(self, num_nodes, num_inputs):
        self.num_inputs = num_inputs
        self.node_list = []
        for i in range(num_nodes):
            self.node_list.append(Node(np.random.rand(self.num_inputs), 1))

    def feed_forward(self, inputs):
        outputs = []
        for node in self.node_list:
            outputs.append(node.feedforward(inputs))

        return outputs

    ## This function returns the weight matrix of the current layer; each row represents a node in the previous layer,
    ## with each column representing a node on the current layer, and the value in that entry representing the weight
    ## from the previous layer's node to the current layer's node
    def get_weight_matrix(self):
        weight_matrix = np.array
        for node in self.node_list:
            weight_matrix = weight_matrix.append(node.weights, axis=0)
        weight_matrix = weight_matrix.T
        return weight_matrix

    ## This function takes in the errors on the next layer and the next layer's weight matrix, and returns an array of
    ## errors for the current layer
    def get_errors(self, errors_next_layer, weight_matrix_next_layer):
        errors = []
        for i in range(len(self.node_list)):
            errors.append(node_error_calc(errors_next_layer, weight_matrix_next_layer[i]))
        return errors