"""
Problem 987 — Straight Eight
Algorithm (summary):
Count the number of ways to choose eight disjoint 5-card straights from a 52-card deck. This is a
combinatorics counting problem (set packing). A brute-force search over combinations is feasible
with pruning; however enumerating all possibilities is large. This solver will attempt a constrained
combinatorial search with caching but may not finish within time limit.
"""
import time
from itertools import combinations

# Precompute all straights as sets of cards (rank,suit)
ranks = list(range(1,14))  # 1..13, Ace as 1 and also can be high handled explicitly
suits = list(range(4))

# Build list of straights as 5-card combinations represented by integers 0..51

def make_deck():
    deck = []
    for r in ranks:
        for s in suits:
            deck.append((r,s))
    return deck

STRAIGHTS = []

def build_straights():
    deck = make_deck()
    # Map card to index
    idx = {deck[i]: i for i in range(52)}
    straights = []
    # sequences of ranks for straights (ace-low and ace-high)
    sequences = []
    # Ace low sequences: 1-2-3-4-5 up to 10-11-12-13-1? Handle separately
    rank_seqs = [list(range(i, i+5)) for i in range(1,10)]  # 1..9 start
    # For ace-high sequence 10,11,12,13,1 represented separately as [10,11,12,13,1]
    rank_seqs.append([10,11,12,13,1])
    for seq in rank_seqs:
        # for each choice of suits for each rank
        suits_choices = [range(4) for _ in seq]
        # iterate product of suits (4^5 = 1024)
        from itertools import product
        for sc in product(*suits_choices):
            cards = tuple(sorted(idx[(seq[i], sc[i])] for i in range(5)))
            straights.append(cards)
    return straights

STRAIGHTS = build_straights()


def solve():
    # Attempt to count number of ways to choose 8 disjoint straights.
    # We'll perform a backtracking search with simple pruning. May not finish in time.
    N = len(STRAIGHTS)
    # Represent straights as bitmasks
    masks = []
    for s in STRAIGHTS:
        m = 0
        for c in s:
            m |= 1<<c
        masks.append(m)
    # sort by mask to try to help pruning
    masks_sorted = masks
    total = 0
    target = 8
    # recursive backtracking
    sys.setrecursionlimit(10000)
    masks_sorted = list(enumerate(masks_sorted))
    masks_sorted.sort(key=lambda x: bin(x[1]).count('1'))
    start_time = time.perf_counter()
    limit_sec = 55.0
    def backtrack(start, chosen, used_mask):
        nonlocal total
        if time.perf_counter() - start_time > limit_sec:
            raise TimeoutError()
        if chosen == target:
            total += 1
            return
        for i in range(start, len(masks_sorted)):
            idx, m = masks_sorted[i]
            if (m & used_mask) == 0:
                backtrack(i+1, chosen+1, used_mask | m)
    try:
        backtrack(0,0,0)
    except TimeoutError:
        return None
    return total

if __name__ == '__main__':
    import sys
    t0 = time.perf_counter()
    ans = None
    try:
        ans = solve()
    except Exception:
        ans = None
    t1 = time.perf_counter()
    elapsed = t1 - t0
    print("ID: 987")
    if ans is None:
        print("Answer: NOT_COMPUTED")
        print("Note: search timed out or not finished within resource limits.")
    else:
        print(f"Answer: {ans}")
    print(f"Time: {elapsed:.6f}s")
    sys.exit(0)
