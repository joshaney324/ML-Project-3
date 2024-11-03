import numpy as np
from Layer import Layer, softmax
from Metric_functions import mean_squared_error, accuracy
from CrossValidateFunctions import cross_validate_tune_classification, cross_validate_tune_regression
from Fold_functions import get_tune_folds, get_folds_classification, get_folds_regression
from src.Node import sigmoid_derivative


# TODO: Change bias configuration or backpropogation and error algorithm so weights from biases can be updated
# TODO: Implement backpropogation for updates to weights to output layer (using derivative of output activation function)
# TODO: Implement weight update algorithms (likely descending from network to layer to node), given weight updates for all nodes

# TODO: Implement minibatch learning

class Network:
    def __init__(self, learning_rate, num_hidden_layers, hidden_layer_sizes, num_inputs, num_outputs, output_type, biased_layers):
        # self.max_train_iterations = max_train_iterations
        self.layers = []
        self.learning_rate = learning_rate
        self.output_type = output_type

        # set up hidden layers
        for i in range(num_hidden_layers):
            has_bias = False
            has_input_bias = False
            for j in biased_layers:
                if i == j:
                    has_bias = True
                if i == j+1:
                    has_input_bias = True
            if i == 0:
                if has_input_bias:
                    self.layers.append(Layer(hidden_layer_sizes[i], num_inputs + 1, has_bias))
                else:
                    self.layers.append(Layer(hidden_layer_sizes[i], num_inputs, has_bias))
            else:
                if has_input_bias:
                    self.layers.append(Layer(hidden_layer_sizes[i], hidden_layer_sizes[i - 1] + 1, has_bias))
                else:
                    self.layers.append(Layer(hidden_layer_sizes[i], hidden_layer_sizes[i - 1], has_bias))
        if len(self.layers) > 0:
            if self.layers[-1].has_bias:
                self.layers.append(Layer(num_outputs, hidden_layer_sizes[-1] + 1, False))
            else:
                self.layers.append(Layer(num_outputs, hidden_layer_sizes[-1], False))
        else:
            self.layers.append(Layer(num_outputs, num_inputs, False))

    def feedforward(self, inputs):
        layer_vals = []

        for i in range(len(self.layers)):
            if i == 0:
                layer_vals.append(self.layers[i].feed_forward(inputs))
            else:
                layer_vals.append(self.layers[i].feed_forward(layer_vals[-1]))
        if self.output_type == "classification":
            layer_vals[-1] = softmax(layer_vals[-1])
        return layer_vals

    # This function returns a list of lists of errors for all nodes in the network
    def get_errors(self, real_outputs, expected_outputs):
        # Get the error of the output layer
        errors = np.array(expected_outputs) - np.array(real_outputs)
        errors = [errors]
        for i in range(len(self.layers) - 1):
            # Get the weight matrix of the last layer with a calculated error
            weight_matrix = self.layers[-(i+1)].get_weight_matrix()
            # Add the errors of the current layer to the errors list
            layer_errors = self.layers[-(i+2)].get_errors(errors[0], weight_matrix)
            errors.insert(0, layer_errors)
        return errors

    def backpropogation(self, inputs, expected_outputs):

        layer_vals = self.feedforward(inputs)
        error_vals = self.get_errors(layer_vals[-1], expected_outputs)
        weight_updates = []
        # input_layer_weight_updates = []
        # # calculate the changes in the weights to the first hidden layer -- only change from the following for loop is that the previous row values are just the input layer values
        # if len(self.layers) > 1:
        #     for h in range(len(self.layers[0].node_list)):
        #         node_weight_updates = []
        #         for j in range(len(self.layers[0].node_list[h].weights)):
        #             weight_update = self.learning_rate * error_vals[0][h] * sigmoid_derivative(layer_vals[0][h]) * inputs[j]
        #             node_weight_updates.append(weight_update)
        #         input_layer_weight_updates.append(node_weight_updates)
        #     weight_updates.append(input_layer_weight_updates)
        # # calculate the changes in the weights to the remaining hidden layers
        # for k in range(1, len(self.layers) - 1):
        #     # initialize the weight update matrix for this layer -- this will store all the updates to all the weights TO the layer
        #     layer_weight_updates = []
        #     # for each node on the layer...
        #     for h in range(len(self.layers[k].node_list)):
        #         # initialize the weight update array for that node -- this will store all the updates to all the weights TO the node
        #         node_weight_updates = []
        #         # for every node on the previous layer (equivalently, every weight incoming to this node
        #         for j in range(len(self.layers[k].node_list[h].weights)):
        #             # update function from the book -- multiply learning rate, error on this node, derivative of activation function for this node at its current value, and value of node the weight is coming from
        #             weight_update = self.learning_rate * error_vals[k][h] * sigmoid_derivative(layer_vals[k][h]) * layer_vals[k-1][j]
        #             # append the weight update for that weight to the update array for the node
        #             node_weight_updates.append(weight_update)
        #         # append the update array for that node to the update matrix for the layer
        #         layer_weight_updates.append(node_weight_updates)
        #     # append the update matrix for that layer to the structure storing all weight updates for the network
        #     weight_updates.append(layer_weight_updates)
        # output_layer_weight_updates = []
        # # calculate the changes in the weights to the output layer -- only change from the previous for loop is that the derivative of the activation function is 1, so it has been taken out (and some indexing)
        # if len(self.layers) > 1:
        #     for h in range(len(self.layers[-1].node_list)):
        #         node_weight_updates = []
        #         for j in range(len(self.layers[-1].node_list[h].weights)):
        #             weight_update = self.learning_rate * error_vals[-1][h] * layer_vals[-2][j]
        #             node_weight_updates.append(weight_update)
        #         output_layer_weight_updates.append(node_weight_updates)
        #     weight_updates.append(output_layer_weight_updates)
        #
        # if len(self.layers) == 1:
        #     for h in range(len(self.layers[0].node_list)):
        #         node_weight_updates = []
        #         for j in range(len(self.layers[0].node_list[h].weights)):
        #             weight_update = self.learning_rate * error_vals[0][h] * inputs[j]
        #             node_weight_updates.append(weight_update)
        #         input_layer_weight_updates.append(node_weight_updates)
        #     weight_updates.append(input_layer_weight_updates)

        # calculate the changes in the weights to the first hidden layer -- only change from the following for loop is that the previous row values are just the input layer values
        if len(self.layers) > 1:
            layer_weight_updates = self.layers[0].get_weight_updates(self.learning_rate, False, error_vals[0],
                                                                     layer_vals[0], inputs)
            weight_updates.append(layer_weight_updates)
            for k in range(1, len(self.layers) - 1):
                layer_weight_updates = self.layers[k].get_weight_updates(self.learning_rate, False, error_vals[0],
                                                                         layer_vals[k], layer_vals[k-1])
                weight_updates.append(layer_weight_updates)

            layer_weight_updates = self.layers[-1].get_weight_updates(self.learning_rate, True, error_vals[-1],
                                                                      layer_vals[-1], layer_vals[-2])
            weight_updates.append(layer_weight_updates)
        if len(self.layers) == 1:
            layer_weight_updates = self.layers[0].get_weight_updates(self.learning_rate, True, error_vals[0],
                                                                     layer_vals[0], inputs)
            weight_updates.append(layer_weight_updates)
        # update weights using the changes calculated above
        for i, layer in enumerate(self.layers):
            layer.update_weights(weight_updates[i])

    def train(self, data, labels, test_data, test_labels, max_iterations):
        if self.output_type == 'classification':
            best_metric = 0.0
            data_folds, label_folds = get_folds_classification(data, labels, 10)
        else:
            best_metric = np.inf
            data_folds, label_folds = get_folds_regression(data, labels, 10)
        for i in range(max_iterations):
            # print("-------------------------------------------------------------------------")
            for datapoint, label in zip(data, labels):
                self.backpropogation(datapoint, label)
                if self.output_type == 'classification':
                    predictions = []
                    for datum in test_data:
                        predictions.append(self.predict(datum))
                    accuracy_vals, matrix = accuracy(predictions, test_labels)
                    acc_val = []
                    for j in range(len(accuracy_vals)):
                        acc_val.append(accuracy_vals[j][1])

                    acc_val = np.mean(acc_val)
                    # print("Training Accuracy: " + str(acc_val))

                    new_metric = acc_val
                    if new_metric < best_metric:
                        print("convergence reached")
                        return
                elif self.output_type == 'regression':
                    train_data, train_labels, test_data, test_labels = get_tune_folds(data_folds, label_folds)
                    predictions = []
                    for datum in test_data:
                        predictions.append(self.feedforward(datum)[-1])
                    mse = mean_squared_error(test_labels, predictions, len(predictions))

                    new_metric = mse
                    if new_metric > best_metric:
                        print("convergence reached")
                        return

    def predict(self, test_point):
        prediction = self.feedforward(test_point)[-1]
        pred_max = np.max(prediction)
        for j in range(len(prediction)):

            if prediction[j] == pred_max:
                prediction[j] = 1
            else:
                prediction[j] = 0

        return prediction

