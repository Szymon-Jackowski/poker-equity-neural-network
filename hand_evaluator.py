def check_straight_flush(cards):
    '''
    cards: list of 7 (rank, suit) tuples, in any order (function sorts internally)
    Returns: [8, high_card] if a straight flush exists, None otherwise
    '''

    #grouping by suit + sorting
    cards.sort(key=lambda x: (x[1],x[0]))
    cards_in_a_row=1
    max_card=cards[-1][0]
    suit=cards[-1][1]

    for i in range(5, -1,-1):
        if cards[i]==(cards[i+1][0]-1, suit):
            cards_in_a_row+=1
            if(cards_in_a_row==5):
                return [8, max_card]

            #special case: wheel straight flush
            elif(cards_in_a_row==4 and max_card==5):
                if (14, suit) in cards:
                    return [8,5]
        else:
            max_card=cards[i][0]
            suit=cards[i][1]
            cards_in_a_row=1
    
    

def check_quads(cards):
    '''
    cards: list of 7 (rank, suit) tuples
    Returns: [7, quad_rank, kicker] if four of a kind exists, None otherwise
    '''

    cards.sort()
    if cards[-1][0]==cards[-4][0]:
        return [7, cards[-1][0], cards[-5][0]]
    
    for i in range(5, 2, -1):
        if cards[i][0]==cards[i-3][0]:
            return [7, cards[i][0], cards[-1][0]]


def check_full_house(cards):
    '''
    cards: list of 7 (rank, suit) tuples
    Returns: [6, three_of_a_kind_rank, pair_rank] if a full house exists, None otherwise
    '''
    
    cards.sort()
    paired_cards=None
    three_of_a_kind=None

    for i in range(6, 1, -1):
        if cards[i][0]==cards[i-2][0]:
            three_of_a_kind=cards[i][0]
            break
    for i in range(6, 0, -1):
        if cards[i][0]==cards[i-1][0] and cards[i][0]!=three_of_a_kind:
            paired_cards=cards[i][0]
            break
    
    if three_of_a_kind and paired_cards:
        return [6, three_of_a_kind, paired_cards]

def check_flush(cards):
    '''
    cards: list of 7 (rank, suit) tuples
    Returns: [5, rank1, rank2, rank3, rank4, rank5] (5 highest ranks of the flush suit,
    highest to lowest) if a flush exists, None otherwise
    '''

    #grouping by suit + sorting
    cards.sort(key=lambda x: (x[1], x[0]))
    for i in range(6, 3, -1):
        if cards[i][1]==cards[i-4][1]:
            #there can only be one flush so we return right away
            return [5]+[c[0] for c in reversed(cards[i-4:i+1])]

def check_straight(cards):
    '''
    cards: list of 7 (rank, suit) tuples
    Returns: [4, high_card] if a straight exists (ignoring suits), None otherwise
    '''

    cards.sort()
    #dictionary keeps keys without duplicates and we put them in sorted order
    cards_dict={c[0]:0 for c in cards}
    ranks=list(cards_dict)
    dict_size=len(cards_dict)

    for i in range(dict_size-1, 3, -1):
        if ranks[i]==ranks[i-4]+4:
            return [4, ranks[i]]
    
    if dict_size>=5:
        if ranks[3]==5 and ranks[-1]==14:
            return [4, 5]

def check_three_of_a_kind(cards):
    '''
    cards: list of 7 (rank, suit) tuples
    Returns: [3, trips_rank, kicker1, kicker2] if three of a kind exists, None otherwise
    '''

    cards.sort()
    three_of_a_kind=None
    kickers=[]
    for i in range(6, 1, -1):
        if cards[i][0]==cards[i-2][0]:
            three_of_a_kind=cards[i][0]
            break
    
    for i in range(6, -1, -1):
        if cards[i][0]!=three_of_a_kind:
            kickers.append(cards[i][0])
            if len(kickers)==2:
                break
    if three_of_a_kind:
        return [3, three_of_a_kind, kickers[0], kickers[1]]

def check_two_pair(cards):
    '''
    cards: list of 7 (rank, suit) tuples
    Returns: [2, high_pair_rank, low_pair_rank, kicker] if two pair exists, None otherwise
    '''

    cards.sort()
    pairs=[]
    kicker=None
    for i in range(6, 0, -1):
        if cards[i][0]==cards[i-1][0]:
            pairs.append(cards[i][0])
            if len(pairs)==2:
                break
    for i in range(6, -1, -1):
        if cards[i][0] not in pairs:
            kicker=cards[i][0]
            break
    if len(pairs)==2:
        return [2, pairs[0], pairs[1], kicker]

def check_one_pair(cards):
    '''
    cards: list of 7 (rank, suit) tuples
    Returns: [1, pair_rank, kicker1, kicker2, kicker3] if a pair exists, None otherwise
    '''

    cards.sort()
    pair=None
    kickers=[]
    for i in range(6, 0, -1):
        if cards[i][0]==cards[i-1][0]:
            pair=cards[i][0]
            break
    for i in range(6, -1, -1):
        if cards[i][0]!=pair:
            kickers.append(cards[i][0])
            if len(kickers)==3:
                break
    if pair:
        return [1, pair]+[k for k in kickers]

def high_card(cards):
    '''
    cards: list of 7 (rank, suit) tuples
    Returns: [0, rank1, rank2, rank3, rank4, rank5] (5 highest ranks, highest to lowest)
    '''

    cards.sort()
    return [0]+[cards[i][0] for i in range(6, 1, -1)]

def evaluate_hand(cards):   
    '''
    cards: a list of 7 tuples, each representing one card as (rank, suit)
    rank: int, 2-14 (2 through 10 are numeric, 11=Jack, 12=Queen, 13=King, 14=Ace)
    suit: string, one of "Hearts", "Spades", "Diamonds", "Clubs"

    Returns: a list representing the best possible 5-card hand from the 7 cards.
    The first element is the hand category (0-8, higher is better):
        0=High Card, 1=Pair, 2=Two Pair, 3=Three of a Kind,
        4=Straight, 5=Flush, 6=Full House, 7=Four of a Kind, 8=Straight Flush
    The remaining elements are tiebreaker ranks (highest to lowest), used to compare
    two hands of the same category. Two hand results can be compared directly with
    standard list comparison (e.g. hand_a > hand_b), since Python compares lists
    element by element in order.
    '''
    Result=check_straight_flush(cards)
    if Result:
        return Result
    Result=check_quads(cards)
    if Result:
        return Result
    Result=check_full_house(cards)
    if Result:
        return Result
    Result=check_flush(cards)
    if Result:
        return Result
    Result=check_straight(cards)
    if Result:
        return Result
    Result=check_three_of_a_kind(cards)
    if Result:
        return Result
    Result=check_two_pair(cards)
    if Result:
        return Result
    Result=check_one_pair(cards)
    if Result:
        return Result
    return high_card(cards)




    
        





