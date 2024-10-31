from Network import Network

network = Network(0.1, 2, [3, 2], 2, 2, "classification", [0, 1, 2])

layer_Vals = network.feedforward([2, 1])
weight_Updates = network.backpropogation([2, 1], [1, 2])
print("test")
