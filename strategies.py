"""Implementation of the 4 strategies for MAB."""

import numpy as np


def random_strategy(arms, arm_distributions, num_turns):
    for turn in range(num_turns):
        arm_to_play = arms.get_random_arm()
        arm_reward = np.random.choice(arm_distributions[arm_to_play])
        arms.add_arm_reward(arm_to_play, arm_reward)


def naive_greedy(arms, arm_distributions, num_turns, num_exploration_turns):
    for turn in range(num_exploration_turns):
        arm_to_play = arms.get_random_arm()
        arm_reward = np.random.choice(arm_distributions[arm_to_play])
        arms.add_arm_reward(arm_to_play, arm_reward)

    arm_to_play = arms.optimal_arm

    for turn in range(num_turns):
        arm_reward = np.random.choice(arm_distributions[arm_to_play])
        arms.add_arm_reward(arm_to_play, arm_reward)


def epsilon_first_greedy(arms, arm_distributions, num_turns, num_exploration_turns):
    for arm_to_play in arms.arms:
        for turn in range(num_exploration_turns):
            arm_reward = np.random.choice(arm_distributions[arm_to_play])
            arms.add_arm_reward(arm_to_play, arm_reward)

    arm_to_play = arms.optimal_arm

    for turn in range(num_turns):
        arm_reward = np.random.choice(arm_distributions[arm_to_play])
        arms.add_arm_reward(arm_to_play, arm_reward)


def epsilon_greedy(
    arms, arm_distributions, num_turns, num_exploration_turns, epsilon=0.8
):
    for turn in range(num_exploration_turns):
        arm_to_play = arms.get_random_arm()
        arm_reward = np.random.choice(arm_distributions[arm_to_play])
        arms.add_arm_reward(arm_to_play, arm_reward)

    for turn in range(num_turns):
        if np.random.uniform(0, 1) > epsilon:
            arm_to_play = arms.get_random_arm()
        else:
            arm_to_play = arms.optimal_arm

        arm_reward = np.random.choice(arm_distributions[arm_to_play])
        arms.add_arm_reward(arm_to_play, arm_reward)


def ucb(arms, arm_distributions, num_turns):
    for turn in range(num_turns):
        ucb_map = {}
        for arm in arms.arms:
            reward = arms.get_arm_reward(arm)
            delta_i = np.sqrt(
                (3 * np.log(len(arms.arms[arm]))) / (2 * len(arms.arms[arm]))
            )
            ucb_map[arm] = delta_i + reward
        arm_to_play = max(ucb_map, key=ucb_map.get)
        arm_reward = np.random.choice(arm_distributions[arm_to_play])
        arms.add_arm_reward(arm_to_play, arm_reward)
