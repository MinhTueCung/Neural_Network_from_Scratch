import numpy as np

np.random.seed(0)

class Layer:
    def __init__(self, n_neurons, n_input_neurons):
        self.weights = np.zeros((n_input_neurons, n_neurons))
        self.n_neurons = n_neurons

    def forward_process_return_outputs(self, inputs):
        return 1 / (np.exp(-self.calculate_inputs(inputs)) + 1)

    def deriative_sigmoid_backward(self, x):
        # Use by backpropagating, x is here the calculated inputs matrix ( np.dot(inputs, self.weights) ), accepted by this layer
        return np.exp(-x) / (1 + np.exp(-x))**2

    def calculate_inputs(self, inputs):
        return np.dot(inputs, self.weights) + np.zeros((1, self.n_neurons))



