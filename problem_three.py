"""Compare Implementations of Strategies for Different Bandits."""

from problem_two import random_strategy, naive_greedy, epsilon_first_greedy, epsilon_greedy
from bandits import MultiArmedBandit

import numpy as np
from scipy.stats import beta


strategies = {
    "random": random_strategy,
    "naive greedy": naive_greedy,
    "epsilon first greedy": epsilon_first_greedy,
    "epsilon greedy": epsilon_greedy
}


EPSILON = 0.8
NUM_GAMBLES = 100
NUM_TURNS = 1_000


num_arms = 5
easy_bandits = {}
for i in range(1, num_arms + 1):
    a = 1 / i
    b = i
    r = beta.rvs(a, b, size=1000)
    easy_bandits[i] = r

hard_bandits = {}
for i in range(1, num_arms + 1):
    a = 0.5 + (i / 10)
    b = i
    r = beta.rvs(a, b, size=1000)
    hard_bandits[i] = r

for strategy in strategies:
    easy_arms = MultiArmedBandit()
    hard_arms = MultiArmedBandit()
    for i in range(1, num_arms + 1):
        easy_arms.add_arm(i)
        hard_arms.add_arm(i)

    if strategy == "random":
        num_turns = NUM_TURNS
        random_strategy(easy_arms, easy_bandits, num_turns)
        random_strategy(hard_arms, hard_bandits, num_turns)
    elif strategy == "naive greedy":
        num_exploration_turns = 100 * num_arms
        num_turns = NUM_TURNS - num_exploration_turns
        naive_greedy(easy_arms, easy_bandits, num_turns, num_exploration_turns)
        naive_greedy(hard_arms, hard_bandits, num_turns, num_exploration_turns)
    elif strategy == "epsilon first greedy":
        num_exploration_turns = 100
        num_turns = NUM_TURNS - num_exploration_turns * num_arms
        epsilon_first_greedy(easy_arms, easy_bandits, num_turns, num_exploration_turns)
        epsilon_first_greedy(hard_arms, hard_bandits, num_turns, num_exploration_turns)
    elif strategy == "epsilon greedy":
        num_exploration_turns = 100 * num_arms
        num_turns = NUM_TURNS - num_exploration_turns
        epsilon_greedy(easy_arms, easy_bandits, num_turns, num_exploration_turns)
        epsilon_greedy(hard_arms, hard_bandits, num_turns, num_exploration_turns)