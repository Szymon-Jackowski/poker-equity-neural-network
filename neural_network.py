import numpy as np
RANKS = [i for i in range(2,15)]
SUITS = {"Spades":0, "Hearts":1, "Diamonds":2, "Clubs":3}

def encode_card(card):
    '''
    card: 2 element list with card's rank at card[0] and card's suit at card[1]
    '''
    rank_vector = [0 for i in range(2, 15)]

    rank_vector[card[0]-2]=1
    #-2 because the numbers start at 2, not at 0

    color_vector = [0 for i in range(len(SUITS))]
    color_vector[SUITS[card[1]]]=1

    return rank_vector+color_vector


def encode_hand(hero_cards, community_cards):
    '''
    hero_cards: list of 2 (rank, suit) tuples - hero's hole cards
    community_cards: list of (rank, suit) tuples on the board

    Returns: flat list combining one-hot encodings of all cards
    '''
    hand_vector=[bit for c in sorted(hero_cards) for bit in encode_card(c)]
    community_vector=[bit for c in sorted(community_cards) for bit in encode_card(c)]
    return hand_vector+community_vector


class NeuralNetwork:
    def __init__(self, hand_size, hidden_layer_size, output_size):
        '''
        hand_size: length of the encoded input vector
        hidden_layer_size: number of neurons in the hidden layer
        output_size: size of the output (1 for equity prediction)
        '''
        self.weight_first=0.01*np.random.randn(hand_size, hidden_layer_size) 
        self.bias_first=np.zeros(hidden_layer_size)
        self.weight_second=0.01*np.random.randn(hidden_layer_size, output_size)
        self.bias_second=np.zeros(output_size)

    def forward_propagation(self, hand_vector):
        '''
        hand_vector: encoded input (from encode_hand)

        Returns: [output, relu, z1] - prediction and intermediate values needed for backprop
        '''
        z1=np.dot(hand_vector, self.weight_first)+self.bias_first
        relu=np.maximum(0, z1)
        z2=np.dot(relu, self.weight_second)+self.bias_second
        output=1/(1+np.exp(-z2))
        return [output, relu, z1]

    def back_propagation(self, hand_vector, answer, output, z1, relu, learning_rate):
        '''
        answer: true equity value (target) for this example
        output, z1, relu: intermediate values from forward_propagation
        
        Updates weights and biases in place using gradient descent
        '''
        gradient_z2 = (output-answer) * output * (1-output)
        
        gradient_weight_second=np.outer(relu, gradient_z2)
        gradient_bias_second=gradient_z2
        
        gradient_relu=np.dot(gradient_z2, self.weight_second.T)
        gradient_z1=gradient_relu*(z1>0)

        gradient_weight_first=np.outer(hand_vector, gradient_z1)
        gradient_bias_first=gradient_z1

        self.weight_first-=learning_rate*gradient_weight_first
        self.bias_first-=learning_rate*gradient_bias_first
        self.weight_second-=learning_rate*gradient_weight_second
        self.bias_second-=learning_rate*gradient_bias_second

