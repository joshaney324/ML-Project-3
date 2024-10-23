import numpy as np


class ActivationFunction:
    def __init__(self, activation_type):
        self.activation_type = activation_type

    def calculate_activation(self, input):
        if self.activation_type == "sigmoid":
            return 1 / (1 + np.exp(-input))
        elif self.activation_type == "softmax":

            return np.exp(input) / np.sum(np.exp(input), axis=0)
        elif self.activation_type == "linear":
            return input
    def calc_activation_deriv(self, output_val):
        if self.activation_type == "sigmoid":
            return output_val * (1 - output_val)
        elif self.activation_type == "softmax":
            return output_val
        elif self.activation_type == "linear":
            return 1.0
