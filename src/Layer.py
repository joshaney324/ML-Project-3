import numpy as np
from Node import Node


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
