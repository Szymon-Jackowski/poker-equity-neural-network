from neural_network import encode_hand, NeuralNetwork
from data_generator import generate_dataset, simulate_equity_universal, stage_dict

def train(stage, examples, iterations, epochs, learning_rate, extra_examples=None):

    hidden_stage_dict={"preflop":16, "flop":32, "turn":32, "river":64}
    input_vector_size=(13+4)*stage_dict[stage]
    hidden_layer_size=hidden_stage_dict[stage]

    taught_network=NeuralNetwork(input_vector_size, hidden_layer_size, 1)
    dataset = generate_dataset(stage, examples, iterations)
    if extra_examples:
        hands_to_avoid=[ds[0] for ds in dataset]
        for ex in extra_examples:
            if ex not in hands_to_avoid:
                equity=simulate_equity_universal(ex[:2], ex[2:], iterations)
                dataset.append((ex, equity))
    history_of_error=[]
    hands_equity = {tuple(hand[0]): [hand[1]] for hand in dataset}
    #WARNING: 0th element in hands_equity is the equity taken from the dataset
    

    for _ in range(epochs):
        error=0
        for card_set in dataset:
            #card_set has cards at 0th index and equity at 1st index
            hero_cards=card_set[0][:2]
            community_cards=card_set[0][2:]
            code=encode_hand(hero_cards, community_cards)
            received_list=taught_network.forward_propagation(code)
            #[output, relu, z1] is the return of the forward propagation
            rl=received_list
            error+=(rl[0]-card_set[1])**2
            taught_network.back_propagation(code, card_set[1], rl[0], rl[1], rl[2], learning_rate)
            hands_equity[tuple(card_set[0])].append(rl[0])
        history_of_error.append(error/len(dataset))
    return [taught_network, history_of_error, hands_equity]

if __name__ == "__main__":
    result = train("flop", 500, 500, 200, 0.1, [])
    print(f"Final training error: {result[1][-1][0]:.4f}")