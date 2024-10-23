import numpy as np
from Node import Node


class Layer:
    def __init__(self, num_nodes, activation_type):
        self.node_list = []
        self.layer_type = None
        self.activation_type = activation_type
        self.inputs = []
        self.outputs = []
        self.layer_gradients = []

        for i in range(num_nodes):
            self.node_list.append(Node(self.activation_type, self.layer_type))

    def forward(self, inputs):
        print("forward pass")

    def backward(self, gradients):
        print("backward pass")

    def update_layer_weights(self, learning_rate):
        print("update weights")
