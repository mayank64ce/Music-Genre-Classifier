import numpy as np
from random import random


class MLP(object):
    """A Multilayer Perceptron class.
    """

    def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2):
        """Constructor for the MLP. Takes the number of inputs,
            a variable number of hidden layers, and number of outputs
        Args:
            num_inputs (int): Number of inputs
            hidden_layers (list): A list of ints for the hidden layers
            num_outputs (int): Number of outputs
        """

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # create a generic representation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]

        # create random connection weights for the layers
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

        activations = []

        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        derivatives = []

        for i in range(len(layers) - 1):
            d = np.zeros(layers[i])
            derivatives.append(d)
        self.derivatives = derivatives

    def forward_propagate(self, inputs):

        # the input layer activation is just the input itself
        activations = inputs
        self.activations[0] = inputs
        # iterate through the network layers
        for i, w in enumerate(self.weights):
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)

            # apply sigmoid activation function
            activations = self._sigmoid(net_inputs)
            self.activations[i + 1] = activations

        # return output layer activation
        return activations

    def back_propagate(self, error, verbose=False):

        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i + 1]
            delta = error * self._sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i]  # array([0.1, 0.2, ......]) --> array([[1.0], [0.2], ......])
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print("Derivatives for W{}: {}".format(i, self.derivatives[i]))

        return error

    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]

            derivatives = self.derivatives[i]

            weights += derivatives * learning_rate

    def train(self, inputs, targets, epochs, learning_rate):

        for i in range(epochs):
            sum_error = 0.0;
            for input, target in zip(inputs, targets):
                output = mlp.forward_propagate(input)
                # calculate error
                error = target - output
                mlp.back_propagate(error)
                # perform gradient_descent
                mlp.gradient_descent(learning_rate)
                sum_error+= self._mse(target, output)

            print("Error at Epoch {} is {}".format(i+1, sum_error/len(inputs)))


    def _mse(self,target, output):
        return np.average((target-output)**2)

    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def _sigmoid(self, x):
        """Sigmoid activation function
        Args:
            x (float): Value to be processed
        Returns:
            y (float): Output
        """

        y = 1.0 / (1 + np.exp(-x))
        return y


if __name__ == "__main__":
    # create a Multilayer Perceptron
    mlp = MLP(2, [5], 1)

    # set random values for network's input
    inputs = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0]+i[1]] for i in inputs])

    mlp.train(inputs, targets, learning_rate=0.1, epochs=100)

    inputs = np.array([0.1, 0.3])
    targets = np.array([0.4])

    output = mlp.forward_propagate(inputs)
    print()
    print()
    print("Our network believes that {} + {} is equal to {}".format(inputs[0], inputs[1], output))
    print("Final weights:")
    for i in range(len(mlp.weights)):
        print("Weight in Layer {} is {}".format(i, mlp.weights[i]))


