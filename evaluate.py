from data_generator import generate_dataset
from neural_network import NeuralNetwork, encode_hand
from train import train

def evaluate(stage, examples, iterations, neural_network):
    data_set=generate_dataset(stage, examples, iterations)
    error=0
    for ex in data_set:
        code=encode_hand(ex[0][:2], ex[0][2:])
        predicted_output=neural_network.forward_propagation(code)[0]
        error+=(predicted_output-ex[1])**2
    return error/len(data_set)
