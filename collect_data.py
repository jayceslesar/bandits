"""Compare Implementations of Strategies for Different Bandits."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import beta

from bandits import MultiArmedBandit
from strategies import (epsilon_first_greedy, epsilon_greedy, naive_greedy,
                        random_strategy, ucb)

strategies = {
    "random": random_strategy,
    "naive greedy": naive_greedy,
    "epsilon first greedy": epsilon_first_greedy,
    "epsilon greedy": epsilon_greedy,
    "upper confidence bound": ucb,
}


EPSILON = 0.8
NUM_TURNS = 1_000


def plot_strategy(easy_arms, hard_arms, strategy_name):
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Easy Bandit", "Hard Bandit"))
    for i, arms in enumerate((easy_arms, hard_arms)):
        for arm in arms.arms:
            x = list(range(len(arms.arms[arm])))
            y = arms.arms[arm]
            try:
                regret = arms.get_arm_regret(arm, max_turns=100)
                total_regret = sum(regret)
                regret_str = f" with regret {total_regret:.2f}"
            except ValueError:
                regret_str = ""
            fig.add_trace(
                go.Scatter(x=x, y=y, name=f"Arm {arm} {regret_str}", legendgroup=i),
                row=i + 1,
                col=1,
            )
            fig.update_xaxes(title_text="Turn", row=i + 1, col=1)
            fig.update_yaxes(title_text="Reward", row=i + 1, col=1)

    fig.update_layout(
        title_text=f"Rewards for {strategy_name.title()} Strategy",
        legend_tracegroupgap=300,
    )
    fig.show()


num_arms = 5
easy_bandits = {}
hard_bandits = {}
for i in range(1, num_arms + 1):
    easy_a = 1 / i
    hard_a = 0.5 + (i / 1000)
    b = i
    easy_bandits[i] = beta.rvs(easy_a, b, size=1000)
    hard_bandits[i] = beta.rvs(hard_a, b, size=1000)

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
        plot_strategy(easy_arms, hard_arms, strategy)
    elif strategy == "naive greedy":
        num_exploration_turns = 100 * num_arms
        num_turns = NUM_TURNS - num_exploration_turns
        naive_greedy(easy_arms, easy_bandits, num_turns, num_exploration_turns)
        naive_greedy(hard_arms, hard_bandits, num_turns, num_exploration_turns)
        plot_strategy(easy_arms, hard_arms, strategy)
    elif strategy == "epsilon first greedy":
        num_exploration_turns = 100
        num_turns = NUM_TURNS - num_exploration_turns * num_arms
        epsilon_first_greedy(easy_arms, easy_bandits, num_turns, num_exploration_turns)
        epsilon_first_greedy(hard_arms, hard_bandits, num_turns, num_exploration_turns)
        plot_strategy(easy_arms, hard_arms, strategy)
    elif strategy == "epsilon greedy":
        num_exploration_turns = 100 * num_arms
        num_turns = NUM_TURNS - num_exploration_turns
        epsilon_greedy(
            easy_arms, easy_bandits, num_turns, num_exploration_turns, epsilon=EPSILON
        )
        epsilon_greedy(
            hard_arms, hard_bandits, num_turns, num_exploration_turns, epsilon=EPSILON
        )
        plot_strategy(easy_arms, hard_arms, strategy)
    elif strategy == "upper confidence bound":
        ucb(easy_arms, easy_bandits, num_turns)
        ucb(hard_arms, hard_bandits, num_turns)
        plot_strategy(easy_arms, hard_arms, strategy)
