from Layer import Layer
import numpy as np

class Neural_Network:
    def __init__(self, n_input_neurons, n_hidden_layers_same_size, size_of_a_hidden_layer, n_output_neurons, learning_rate):
        if n_input_neurons > 0 and n_hidden_layers_same_size > 1 and size_of_a_hidden_layer > 0 and n_output_neurons > 0 and learning_rate >= 0.1 and learning_rate < 1:
            self.n_input_neurons = n_input_neurons
            self.first_hidden_layer = Layer(size_of_a_hidden_layer, n_input_neurons)
            self.other_hidden_layers_list = list()
            for x in range(n_hidden_layers_same_size - 1):
                self.other_hidden_layers_list.append(Layer(size_of_a_hidden_layer, size_of_a_hidden_layer))
            self.output_layer = Layer(n_output_neurons, size_of_a_hidden_layer)
            self.learning_rate = learning_rate
        else:
            print("Invalid Construction of the neural network!")

    def take_inputs_forward_process_return_outputs_list(self, inputs):
        if np.array(inputs).ndim == 1 and len(inputs) == self.n_input_neurons:
            # Outputs List, elements positioned in the same order as the layers, each element represents the output of one layer
            outputs_list = list()
            # inputs valid -> accept and start processing
            # 1st hidden layer
            first_hidden_layer_output = self.first_hidden_layer.forward_process_return_outputs(inputs)
            outputs_list.append(first_hidden_layer_output)
            # 2nd hidden layer
            second_hidden_layer_output = self.other_hidden_layers_list[0].forward_process_return_outputs(first_hidden_layer_output)
            outputs_list.append(second_hidden_layer_output)
            # from 3rd hidden layer to last hidden layer
            for x in range(1, len(self.other_hidden_layers_list)):
                output_from_a_hidden_layer = self.other_hidden_layers_list[x].forward_process_return_outputs(outputs_list[x])
                outputs_list.append(output_from_a_hidden_layer)
            # Output layer (Final outputs)
            final_output = self.output_layer.forward_process_return_outputs(outputs_list[len(outputs_list) - 1])
            outputs_list.append(final_output)
            return outputs_list
        else:
            print("Inputs invalid!")
            return None

    def back_propagating(self, initial_inputs, actual_outputs_list, desired_final_outputs):
        # Update weights of the last output layer

        # Calculate derivative of "total Error", with respect to the "outputs": dTE/dOutputs = Final Outputs - Desired Outputs (All in matrix form)
        dtotalError_dOut_from_last_output_layer = actual_outputs_list[len(actual_outputs_list) - 1] - desired_final_outputs

        # Calculate derivative of final "Outputs", with respect to the "Inputs in the last output layer": dOutputs/dInputs (last outputlayer) = derivative_sigmoid(calculated "Inputs in this layer")
        # Inputs in this layer = Outputs from last hidden layer
        dOut_dIn_from_last_output_layer = self.output_layer.deriative_sigmoid_backward(self.output_layer.calculate_inputs(actual_outputs_list[len(actual_outputs_list) - 2]))

        # Calculate derivative of "Inputs in the last output layer", with respect to the Weights of this output layer
        # dInputs/dWeights = Transposed Matrix of last hidden layer´s outputs
        dIn_dWeights_from_last_output_layer = actual_outputs_list[len(actual_outputs_list) - 2].T

        # Combined all three derivatives -> dTE/dWeights = delta_update_weights, pay attention to the dimensions of the matrices in dot products
        # Inputs, as well as Outputs from the layers, have a shape of (1, ) in Standard
        delta_update_weights = np.dot(dIn_dWeights_from_last_output_layer, dtotalError_dOut_from_last_output_layer * dOut_dIn_from_last_output_layer)

        # Update the Weights of the output layer
        self.output_layer.weights -= self.learning_rate * delta_update_weights

        # Update weights (backwards direction) of the hidden layers, except the very first hidden layer

        for x in range(len(self.other_hidden_layers_list)):
            if x == 0: # First Loop
                dIn_dOutX1_from_last_output_layer = self.output_layer.weights.T # OutX1: Output from layer before final output layer
                dtotalError_dOutX1 = np.dot(dtotalError_dOut_from_last_output_layer * dOut_dIn_from_last_output_layer, dIn_dOutX1_from_last_output_layer)
                dOutX1_dInX1 = self.other_hidden_layers_list[len(self.other_hidden_layers_list) - 1].deriative_sigmoid_backward(self.other_hidden_layers_list[len(self.other_hidden_layers_list) - 1].calculate_inputs(actual_outputs_list[len(actual_outputs_list) - 3]))
                dTotalError_dInX1 = dtotalError_dOutX1 * dOutX1_dInX1
                dInX1_dWeightsX1 = actual_outputs_list[len(actual_outputs_list) - 3 - x].T
                delta_update_weights_X1 = np.dot(dInX1_dWeightsX1, dTotalError_dInX1)
                self.other_hidden_layers_list[len(self.other_hidden_layers_list) - 1 - x].weights -= self.learning_rate * delta_update_weights_X1
            else:
                result = dtotalError_dOut_from_last_output_layer * dOut_dIn_from_last_output_layer
                for y in range(x + 1):
                    if y == 0: # First Loop
                        dIn_dOutfromlayerbefore = self.output_layer.weights.T
                        result = np.dot(result, dIn_dOutfromlayerbefore)
                    else:
                        dIn_dOutLayerbefore = self.other_hidden_layers_list[len(self.other_hidden_layers_list) - y].weights.T
                        result = np.dot(result, dIn_dOutLayerbefore)

                    dOutLayerbefore_dInlayerbefore = self.other_hidden_layers_list[len(self.other_hidden_layers_list) - y - 1].deriative_sigmoid_backward(self.other_hidden_layers_list[len(self.other_hidden_layers_list) - 1 - y].calculate_inputs(actual_outputs_list[len(actual_outputs_list) - 3 - y]))
                    result *= dOutLayerbefore_dInlayerbefore

                dIn_dWeights = actual_outputs_list[len(actual_outputs_list) - 3 - x].T
                delta_update_weights = np.dot(dIn_dWeights, result)
                self.other_hidden_layers_list[len(self.other_hidden_layers_list) - 1 - x].weights -= self.learning_rate * delta_update_weights

        # Update weights of the first hidden layer
        # This time the "dIn_dWeights" is exactly the transposed inputs matrix, not transposed outputs of a hidden layer
        result = dtotalError_dOut_from_last_output_layer * dOut_dIn_from_last_output_layer
        for t in range(len(self.other_hidden_layers_list)):
            if t == 0:  # First Loop
                dIn_dOutfromlayerbefore = self.output_layer.weights.T
                result = np.dot(result, dIn_dOutfromlayerbefore)
            else:
                dIn_dOutLayerbefore = self.other_hidden_layers_list[len(self.other_hidden_layers_list) - t].weights.T
                result = np.dot(result, dIn_dOutLayerbefore)

            dOutLayerbefore_dInlayerbefore = self.other_hidden_layers_list[len(self.other_hidden_layers_list) - 1 - t].deriative_sigmoid_backward(self.other_hidden_layers_list[len(self.other_hidden_layers_list) - 1 - t].calculate_inputs(actual_outputs_list[len(actual_outputs_list) - 3 - t]))
            result *= dOutLayerbefore_dInlayerbefore

        dIn_dOutfrom_first_hidden_Layer = self.other_hidden_layers_list[0].weights.T
        result = np.dot(result, dIn_dOutfrom_first_hidden_Layer)
        dOut_first_hidden_layer_dIn_first_hidden_layer = self.first_hidden_layer.deriative_sigmoid_backward(self.first_hidden_layer.calculate_inputs(initial_inputs))
        result *= dOut_first_hidden_layer_dIn_first_hidden_layer

        delta_update_weights = np.dot(np.array(initial_inputs).reshape(-1, 1), result)
        self.first_hidden_layer.weights -= self.learning_rate * delta_update_weights






