from data_generator import generate_dataset
from neural_network import encode_hand
from train import train

def evaluate(stage, examples, iterations, neural_network):
    '''
    stage: "preflop", "flop", "turn", or "river"
    examples: number of freshly generated test hands (not used in training)
    iterations: Monte Carlo simulations per test hand, controls equity label accuracy
    neural_network: a trained NeuralNetwork instance

    Returns: mean squared error between the network's predictions and the true
    Monte Carlo equity, averaged over all test examples
    '''
    dataset=generate_dataset(stage, examples, iterations)
    error=0
    for ex in dataset:
        code=encode_hand(ex[0][:2], ex[0][2:])
        predicted_output=neural_network.forward_propagation(code)[0]
        error+=(predicted_output-ex[1])**2
    return error/len(dataset)

if __name__ == "__main__":
    result = train("flop", 500, 500, 200, 0.1, [])
    error = evaluate("flop", 300, 500, result[0])
    print(f"error: {error[0]:.4f}")