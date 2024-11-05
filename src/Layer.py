import numpy as np
from Node import Node, node_error_calc, sigmoid_derivative, sigmoid

# This function returns the soft max of a set of vals
def softmax(output_vals):
    return np.exp(output_vals)/np.sum(np.exp(output_vals))


# This function returns the sigmoid of a set of vals
def sigmoid_layer(output_vals):
    for i in range(len(output_vals)):
        output_vals[i] = sigmoid(output_vals[i])
    return output_vals


class Layer:
    def __init__(self, num_nodes, num_inputs, has_bias):
        self.num_inputs = num_inputs
        self.node_list = []
        self.has_bias = has_bias
        for i in range(num_nodes):
            self.node_list.append(Node(np.random.uniform(-0.01, 0.01, self.num_inputs), False))
        if has_bias:
            self.node_list.append(Node([], True))

    # This function takes in the values from the previous layer (or the input values for the network, if called on the
    # first layer), and returns the unactivated values for every node on the current layer.
    def feed_forward(self, inputs):
        outputs = []
        # For each node on the layer, calculate the weighted sum for that node given the input values, and append the
        # weighted
        for node in self.node_list:
            outputs.append(node.feedforward(inputs))

        return outputs

    # This function returns the weight matrix of the current layer; each row represents a node in the previous layer,
    # with each column representing a node on the current layer, and the value in that entry representing the weight
    # from the previous layer's node to the current layer's node
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

    # This function returns a matrix of weight updates for the incoming weights to this layer given this layer's errors,
    # values, and inputs. Each row corresponds to one node's weight updates
    def get_weight_updates(self, learning_rate, is_output, layer_errors, layer_vals, inputs):
        layer_weight_updates = []
        # For each node on the layer, get the updates to the weights to that node
        for h in range(len(self.node_list)):
            node_weight_updates = self.node_list[h].get_updates(learning_rate, layer_errors[h], layer_vals[h], inputs, is_output)
            layer_weight_updates.append(node_weight_updates)
        return layer_weight_updates
    # update the weights of each node on the layer, given a weight update matrix (each row corresponds to a single node's weight changes)
    def update_weights(self, weight_update_matrix):
        for i, node in enumerate(self.node_list):
            node.update(weight_update_matrix[i])