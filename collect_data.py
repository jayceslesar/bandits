"""Compare Implementations of Strategies for Different Bandits."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import beta

from bandits import MultiArmedBandit
from strategies import (
    epsilon_first_greedy,
    epsilon_greedy,
    naive_greedy,
    random_strategy,
    ucb,
)

np.random.seed(seed=2023)


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
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Easy Bandit Rewards",
            "Easy Bandit Performance Metric",
            "Hard Bandit Rewards",
            "Hard Bandit Performance Metric",
        ),
        column_widths=[0.75, 0.25],
    )
    for i, arms in enumerate((easy_arms, hard_arms)):
        try:
            regret = arms.regret
            metric = arms.metric
            fig.add_trace(
                go.Scatter(
                    x=list(range(100)),
                    y=np.cumsum(regret[:100]),
                    name=f"Cumulative Regret for 100 Turns",
                    legendgroup=i,
                ),
                row=i + 1,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(100)),
                    y=np.diff(metric[:100]),
                    name=f"Delta Reward for 100 Turns",
                    legendgroup=i,
                ),
                row=i + 1,
                col=2,
            )
            fig.update_xaxes(title_text="Turn", row=i + 1, col=2)
            fig.update_yaxes(title_text="Metric", row=i + 1, col=2)
            total_regret = sum(regret)
            regret_str = f"with regret {total_regret:.2f}"
        except ValueError:
            regret_str = ""
        for arm in arms.arms:
            x = list(range(len(arms.arms[arm])))
            y = arms.arms[arm]
            fig.add_trace(
                go.Scatter(x=x, y=y, name=f"Arm {arm}", legendgroup=i),
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
fig = make_subplots(rows=1, cols=2, subplot_titles=("Easy Arms", "Hard Arms"))
for i in range(1, num_arms + 1):
    easy_a = 1 / i
    hard_a = 0.5 + (i / 25)
    b = 1
    easy_bandits[i] = beta.rvs(easy_a, b, size=10_000)
    hard_bandits[i] = beta.rvs(hard_a, b, size=10_000)
    fig.add_trace(
        go.Histogram(x=easy_bandits[i], name=f"Arm {i + 1}", legendgroup=i),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(x=hard_bandits[i], name=f"Arm {i + 1}", legendgroup=i),
        row=1,
        col=2,
    )

fig.update_layout(
    title_text=f"Reward Distributions for Arms",
    barmode="overlay",
    showlegend=False,
    title_x=0.5,
)
fig.show()

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
