import numpy as np
from Node import Node, node_error_calc

def softmax(output_vals):
    return np.exp(output_vals)/np.sum(np.exp(output_vals))

class Layer:
    def __init__(self, num_nodes, num_inputs, has_bias):
        self.num_inputs = num_inputs
        self.node_list = []
        self.has_bias = has_bias
        for i in range(num_nodes):
            self.node_list.append(Node(np.random.uniform(-0.01, 0.01, self.num_inputs), False))
        if has_bias:
            self.node_list.append(Node([], True))

    def feed_forward(self, inputs):
        outputs = []
        for node in self.node_list:
            outputs.append(node.feedforward(inputs))

        return outputs

    ## This function returns the weight matrix of the current layer; each row represents a node in the previous layer,
    ## with each column representing a node on the current layer, and the value in that entry representing the weight
    ## from the previous layer's node to the current layer's node
    def get_weight_matrix(self):
        # Declare the weight matrix
        weight_matrix = []
        # Add all the incoming weights to each node as rows of the matrix
        for node in self.node_list:
            if not node.isBias:
                weight_matrix.append(node.weights)
        # Take the transpose of the matrix, so that rows represent outgoing weights from the last row's nodes
        weight_matrix = np.array(weight_matrix)
        weight_matrix = weight_matrix.T
        return weight_matrix

    # This function takes in the errors on the next layer and the next layer's weight matrix, and returns an array of
    # errors for the current layer
    def get_errors(self, errors_next_layer, weight_matrix_next_layer):
        errors = []
        # Calculate the error for each node in the current layer using the next layer's errors and the correct row of
        # the weight matrix
        for i in range(len(self.node_list)):
            if not self.node_list[i].isBias:
                errors.append(node_error_calc(errors_next_layer, weight_matrix_next_layer[i]))
        return errors

    # update the weights of each node on the layer, given a weight update matrix (each row corresponds to a single node's weight changes)
    def update_weights(self, weight_update_matrix):
        for i, node in enumerate(self.node_list):
            node.update(weight_update_matrix[i])