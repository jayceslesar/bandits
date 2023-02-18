"""Showing Implementation of MAB."""

from bandits import MultiArmedBandit

import numpy as np
from scipy.stats import beta
import plotly.graph_objects as go

arms = MultiArmedBandit()
num_turns = 1_000

num_arms = 3
arm_distributions = {}
for i in range(1, num_arms + 1):
    a = 1 / i
    b = i
    r = beta.rvs(a, b, size=1000)
    arm_distributions[i] = r
    arms.add_arm(i)

for turn in range(num_turns):
    arm_to_play = arms.get_random_arm()
    arm_reward = np.random.choice(arm_distributions[arm_to_play])
    arms.add_arm_reward(arm_to_play, arm_reward)

fig = go.Figure()
for arm in arms.arms:
    arm_rewards = arms.arms[arm]
    fig.add_trace(go.Scatter(x=list(range(len(arm_rewards))), y=arm_rewards, name=f"Arm {arm} Rewards"))

fig.update_layout(title=f"Rewards over Turns for {num_arms} Arms")
fig.update_xaxes(title="Turn Index")
fig.update_yaxes(title="Reward Value")

fig.show()