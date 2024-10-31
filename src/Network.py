import numpy as np
from Layer import Layer
from Metric_functions import mean_squared_error

# TODO: Change bias configuration or backpropogation and error algorithm so weights from biases can be updated
# TODO: Implement backpropogation for updates to weights to output layer (using derivative of output activation function)
# TODO: Implement weight update algorithms (likely descending from network to layer to node), given weight updates for all nodes
# TODO: Implement minibatch learning

class Network:
    def __init__(self, learning_rate, num_hidden_layers, hidden_layer_sizes, num_inputs, num_outputs, max_train_iterations, output_type):
        self.max_train_iterations = max_train_iterations
        self.layers = []
        self.learning_rate = learning_rate
        self.output_type = output_type

        # set up hidden layers
        for i in range(num_hidden_layers):
            if i == 0:
                self.layers.append(Layer(hidden_layer_sizes[i], num_inputs))
            else:
                self.layers.append(Layer(hidden_layer_sizes[i], hidden_layer_sizes[i - 1]))

        self.layers.append(Layer(num_outputs, hidden_layer_sizes[-1]))

    def feedforward(self, inputs):
        layer_vals = []

        for i in range(len(self.layers)):
            if i == 0:
                layer_vals.append(self.layers[i].feed_forward(inputs))
            else:
                layer_vals.append(self.layers[i].feed_forward(layer_vals[-1]))

        return layer_vals

    # This function returns a list of lists of errors for all nodes in the network
    def get_errors(self, real_outputs, expected_outputs):
        # Get the error of the output layer
        errors = np.array(np.array(expected_outputs) - np.array(real_outputs))
        for i in range(len(self.layers) - 1):
            # Get the weight matrix of the last layer with a calculated error
            weight_matrix = self.layers[-(i+2)].get_weight_matrix()
            # Add the errors of the current layer to the errors list
            errors = np.insert(errors, [0], self.layers[-(i+1)].get_errors(errors[0], weight_matrix), axis=0)
        return errors

    def backpropogation(self, inputs, expected_outputs):
        layer_vals = self.feedforward(inputs)
        error_vals = self.get_errors(layer_vals[-1], expected_outputs)
        weight_updates = []
        input_layer_weight_updates = []
        for h in range(len(self.layers[0].node_list)):
            node_weight_updates = []
            for j in range(len(self.layers[0].node_list[h].weights)):
                weight_update = self.learning_rate * error_vals[0][h] * (1 - layer_vals[0][h]) * inputs[j]
                node_weight_updates.append(weight_update)
            input_layer_weight_updates.append(node_weight_updates)
        weight_updates.append(input_layer_weight_updates)
        for k in range(1, len(self.layers) - 1):
            layer_weight_updates = []
            for h in range(len(self.layers[k].node_list)):
                node_weight_updates = []
                for j in range(len(self.layers[k].node_list[h].weights)):
                    weight_update = self.learning_rate * error_vals[k][h] * (layer_vals[k][h] * (1 - layer_vals[k][h])) * layer_vals[k-1][j]
                    node_weight_updates.append(weight_update)
                layer_weight_updates.append(node_weight_updates)
            weight_updates.append(layer_weight_updates)
        output_layer_weight_updates = []
        if self.output_type == "regression":
            for h in range(len(self.layers[-1].node_list)):
                node_weight_updates = []
                for j in range(len(self.layers[-1].node_list[h].weights)):
                    weight_update = self.learning_rate * error_vals[-1][h] * (1) * inputs[j]
                    node_weight_updates.append(weight_update)
                output_layer_weight_updates.append(node_weight_updates)
        weight_updates.append(output_layer_weight_updates)
        if self.output_type == "classification":
            for h in range(len(self.layers[-1].node_list)):
                node_weight_updates = []
                for j in range(len(self.layers[-1].node_list[h].weights)):
                    weight_update = self.learning_rate * error_vals[-1][h] * (1) * inputs[j]
                    node_weight_updates.append(weight_update)
                output_layer_weight_updates.append(node_weight_updates)
        weight_updates.append(output_layer_weight_updates)

    # def backpropagation(self, inputs):
    #     layer_vals = self.feedforward(inputs)
    #
    #     for i in range(self.max_train_iterations):



