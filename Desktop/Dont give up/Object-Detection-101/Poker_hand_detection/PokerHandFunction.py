from collections import Counter

RANK_VALUES = {"A": 14, "K": 13, "Q": 12, "J": 11}
HAND_NAMES = {
    10: "Royal Flush", 9: "Straight Flush", 8: "Four of a Kind",
    7: "Full House", 6: "Flush", 5: "Straight",
    4: "Three of a Kind", 3: "Two Pair", 2: "Pair", 1: "High Card",
}


def _parse(card):
    """'10H' -> (10, 'H'),  'AS' -> (14, 'S')"""
    rank, suit = card[:-1], card[-1]
    return RANK_VALUES.get(rank) or int(rank), suit


def _is_straight(ranks):
    r = sorted(set(ranks))
    if len(r) != 5:
        return False
    if r[-1] - r[0] == 4:
        return True
    return r == [2, 3, 4, 5, 14]  # ace-low straight (wheel)


def findPokerHand(hand, verbose=False):
    """Takes exactly 5 unique card labels, returns the hand name."""
    cards = [_parse(c) for c in hand]
    ranks = [c[0] for c in cards]
    suits = [c[1] for c in cards]

    flush = len(set(suits)) == 1
    straight = _is_straight(ranks)
    counts = sorted(Counter(ranks).values(), reverse=True)

    if flush and straight:
        score = 10 if set(ranks) == {10, 11, 12, 13, 14} else 9
    elif counts == [4, 1]:
        score = 8
    elif counts == [3, 2]:
        score = 7
    elif flush:
        score = 6
    elif straight:
        score = 5
    elif counts == [3, 1, 1]:
        score = 4
    elif counts == [2, 2, 1]:
        score = 3
    elif counts == [2, 1, 1, 1]:
        score = 2
    else:
        score = 1

    if verbose:
        print(hand, HAND_NAMES[score])
    return HAND_NAMES[score]


if __name__ == "__main__":
    tests = [
        (["KH", "AH", "QH", "JH", "10H"], "Royal Flush"),
        (["QC", "JC", "10C", "9C", "8C"], "Straight Flush"),
        (["AD", "2D", "3D", "4D", "5D"], "Straight Flush"),
        (["5C", "5S", "5H", "5D", "QH"], "Four of a Kind"),
        (["2H", "2D", "2S", "10H", "10C"], "Full House"),
        (["2D", "KD", "7D", "6D", "5D"], "Flush"),
        (["JC", "10H", "9C", "8C", "7D"], "Straight"),
        (["AH", "2D", "3S", "4C", "5H"], "Straight"),
        (["QH", "KD", "AS", "2C", "3H"], "High Card"),
        (["10H", "10C", "10D", "2D", "5S"], "Three of a Kind"),
        (["KD", "KH", "5C", "5S", "6D"], "Two Pair"),
        (["2D", "2S", "9C", "KD", "10C"], "Pair"),
        (["KD", "5H", "2D", "10C", "JH"], "High Card"),
    ]
    for cards, expected in tests:
        got = findPokerHand(cards)
        print("PASS" if got == expected else "FAIL", cards, got)
