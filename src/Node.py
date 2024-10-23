from ActivationFunction import ActivationFunction


class Node:
    def __init__(self, activation_type, node_type):
        self.weights = []
        self.old_weights = []
        self.inputs = []
        self.output = 0.0
        self.bias = 0.0
        self.old_bias = 0.0
        self.weighted_sum = 0.0
        self.delta = 0.0
        self.activation_function = ActivationFunction(activation_type)
        self.node_type = node_type

    def activate(self, weighted_sum):
        self.weighted_sum = weighted_sum
        self.output = self.activation_function.calculate_activation(weighted_sum)
        return self.output

    def calc_derivative(self):
        return self.activation_function.calc_activation_deriv(self.output)

    def update_weights(self, learning_rate):
        print("update weights")

