import numpy as np
from Layer import Layer
from Metric_functions import mean_squared_error


class Network:
    def __init__(self, learning_rate, num_hidden_layers, hidden_layer_sizes, num_inputs, num_outputs, max_train_iterations):
        self.max_train_iterations = max_train_iterations
        self.layers = []
        self.learning_rate = learning_rate

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

        # return layer_vals
        return layer_vals

    # This function returns a list of lists of errors for all nodes in the network
    def get_errors(self, inputs, outputs):
        # Get the error of the output layer
        errors = np.array(np.array(self.feedforward(inputs)) - np.array(outputs))
        for i in range(len(self.layers) - 1):
            # Get the weight matrix of the last layer with a calculated error
            weight_matrix = self.layers[-(i+2)].get_weight_matrix()
            # Add the errors of the current layer to the errors list
            errors = np.insert(errors, [0], self.layers[-(i+1)].get_errors(errors[0], weight_matrix), axis=0)

    # def backpropagation(self, inputs):
    #     layer_vals = self.feedforward(inputs)
    #
    #     for i in range(self.max_train_iterations):



