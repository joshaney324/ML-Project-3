import numpy as np
from Layer import Layer


class Network:
    def __init__(self, num_inputs, num_outputs, num_hidden_layers, hidden_node_list, learning_rate,
                 net_type, max_train_iterations):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hidden_layers
        self.hidden_node_list = hidden_node_list
        self.learning_rate = learning_rate
        self.net_type = net_type
        self.max_train_iterations = max_train_iterations
        self.layers = []

        input_layer = Layer(num_inputs, "linear")
        input_layer.layer_type = "input"
        self.layers.append(input_layer)

        for i in range(num_hidden_layers):
            layer = Layer(hidden_node_list[i], "sigmoid")
            layer.layer_type = "hidden"

            if i == 0:
                layer.weights = np.random.randn(hidden_node_list[i], num_inputs) * 0.01
            else:
                layer.weights = np.random.randn(hidden_node_list[i], hidden_node_list[i - 1]) * 0.01

            self.layers.append(layer)

        output_layer = Layer(num_outputs, "linear")
        output_layer.layer_type = "output"
        output_layer.weights = np.random.randn(hidden_node_list[-1], num_outputs) * 0.01

        self.layers.append(output_layer)

