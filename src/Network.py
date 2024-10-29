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

        return layer_vals

    # def backpropagation(self, inputs):
    #     layer_vals = self.feedforward(inputs)
    #
    #     for i in range(self.max_train_iterations):



