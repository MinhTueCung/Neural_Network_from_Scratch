from Neural_Network import Neural_Network

inputs = [5, 6, -2]
desired_outputs = [0.01, 0.04]

neural_network = Neural_Network(3, 5, 5, 2, 0.9)
outputs = neural_network.take_inputs_forward_process_return_outputs_list(inputs)
print("Initial outputs: {}".format(outputs))
for x in range(1600):
    neural_network.back_propagating(inputs, outputs, desired_outputs)
    outputs = neural_network.take_inputs_forward_process_return_outputs_list(inputs)
    print("Updated outputs: {}".format(outputs))

