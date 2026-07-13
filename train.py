from neural_network import encode_hand, NeuralNetwork
from data_generator import generate_dataset

def train(stage, examples, iterations, epochs, learning_rate):
    stage_dict={"preflop":2, "flop":5, "turn":6, "river":7}
    hidden_stage_dict={"preflop":16, "flop":32, "turn":32, "river":64}
    input_vector_size=(13+4)*stage_dict[stage]
    hidden_layer_size=hidden_stage_dict[stage]

    taught_network=NeuralNetwork(input_vector_size, hidden_layer_size, 1)
    dataset = generate_dataset(stage, examples, iterations)

    for _ in range(epochs):
        for card_set in dataset:
            #card_set has cards at 0th index and equity at 1st index
            hero_cards=card_set[0][:2]
            community_cards=card_set[0][2:]
            code=encode_hand(hero_cards, community_cards)
            received_list=taught_network.forward_propagation(code)
            rl=received_list
            #[output, relu, z1] is the return of the forward propagation
            taught_network.back_propagation(code, card_set[1], rl[0], rl[2], rl[1], learning_rate)
    return taught_network
