from hand_evaluator import evaluate_hand
import random

RANKS = [i for i in range(2,15)]
SUITS = ["Spades", "Hearts", "Diamonds", "Clubs"]
DECK=[]

for rank in RANKS:
    for suit in SUITS:
        DECK.append((rank, suit))



def simulate_equity_universal(hero_hand, community_cards, iterations):
    #cards are hand + community cards for the hero
    remaining_cards=[c for c in DECK if c not in hero_hand+community_cards]

    hero_wins=0
    #chops are counted as a 0.5 of a win

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





def simulate_preflop_equity(hand, iterations):
    return simulate_equity_universal(hand, [], iterations)

def simulate_flop_equity(hand, flop, iterations):
    return simulate_equity_universal(hand, flop, iterations)

def simulate_turn_equity(hand, flop, turn, iterations):
    return simulate_equity_universal(hand, flop+[turn], iterations)

def simulate_river_equity(hand, flop, turn, river, iterations):
    return simulate_equity_universal(hand, flop+[turn]+[river], iterations)

