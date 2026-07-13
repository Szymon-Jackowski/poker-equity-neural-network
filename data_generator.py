from hand_evaluator import evaluate_hand
import random

RANKS = [i for i in range(2,15)]
SUITS = ["Spades", "Hearts", "Diamonds", "Clubs"]
DECK=[]

for rank in RANKS:
    for suit in SUITS:
        DECK.append((rank, suit))



def simulate_equity_universal(hero_hand, community_cards, iterations):
    '''
    hero_hand: list of 2 (rank, suit) tuples - the hero's hole cards
    community_cards: list of 0-5 (rank, suit) tuples already known on the board
    (0 for preflop, 3 for flop, 4 for turn, 5 for river)
    iterations: number of Monte Carlo simulations to run

    For each simulation, draws a random villain hand (2 cards) and the remaining
    community cards needed to complete the board, then evaluates both hands and
    counts wins/ties. Ties are counted as 0.5 of a win.

    Returns: float, hero's win probability (equity) between 0 and 1
    '''
    #cards are hand + community cards for the hero
    remaining_cards=[c for c in DECK if c not in hero_hand+community_cards]

    hero_wins=0
    #chops (ties) are counted as a 0.5 of a win

    villain_hand_size=2

    cards_to_draw = 5-len(community_cards)+villain_hand_size
    



    for _ in range(iterations):
        drawn_cards=random.sample(remaining_cards, cards_to_draw)
        villain_hand = drawn_cards[:2]
        full_community_cards = community_cards + drawn_cards[2:]

        hero_score=evaluate_hand(hero_hand+full_community_cards)
        villain_score=evaluate_hand(villain_hand+full_community_cards)

        if hero_score>villain_score:
            hero_wins+=1
        elif hero_score==villain_score:
            hero_wins+=0.5
        
    return hero_wins/iterations

def generate_dataset(stage, examples, iterations):
    '''
    stage: which game stage to generate examples for -
    "preflop" (2 known cards), "flop" (5), "turn" (6), "river" (7)
    examples: how many labeled examples to generate
    iterations: how many Monte Carlo simulations to run per example

    For each example, draws a random hand + community cards for the given stage,
    computes the hero's equity via simulate_equity_universal, and records the pair.

    Returns: list of (cards, equity) tuples, where cards is a list of (rank, suit)
    tuples and equity is a float between 0 and 1
    '''
    stage_dict={"preflop": 2, "flop": 5, "turn": 6, "river": 7}
    dataset=[]
    
    for _ in range(examples):
        drawn_cards=random.sample(DECK, stage_dict[stage])
        hand=drawn_cards[:2]
        community_cards=drawn_cards[2:]
        equity=simulate_equity_universal(hand, community_cards, iterations)
        dataset.append((drawn_cards, equity))

    return dataset
            