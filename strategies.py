"""Implementation of the 4 strategies for MAB."""

import numpy as np

from bandits import MultiArmedBandit


def random_strategy(
    arms: MultiArmedBandit, arm_distributions: dict[int, list[float]], num_turns: int
) -> None:
    for turn in range(num_turns):
        arm_to_play = arms.get_random_arm()
        arm_reward = np.random.choice(arm_distributions[arm_to_play])
        arms.add_arm_reward(arm_to_play, arm_reward)


def naive_greedy(
    arms: MultiArmedBandit,
    arm_distributions: dict[int, list[float]],
    num_turns: int,
    num_exploration_turns: int,
) -> None:
    for turn in range(num_exploration_turns):
        arm_to_play = arms.get_random_arm()
        arm_reward = np.random.choice(arm_distributions[arm_to_play])
        arms.add_arm_reward(arm_to_play, arm_reward)

    arm_to_play = arms.optimal_arm

    for turn in range(num_turns):
        arm_reward = np.random.choice(arm_distributions[arm_to_play])
        arms.add_arm_reward(arm_to_play, arm_reward)


def epsilon_first_greedy(
    arms: MultiArmedBandit,
    arm_distributions: dict[int, list[float]],
    num_turns: int,
    num_exploration_turns: int,
) -> None:
    for arm_to_play in arms.arms:
        for turn in range(num_exploration_turns):
            arm_reward = np.random.choice(arm_distributions[arm_to_play])
            arms.add_arm_reward(arm_to_play, arm_reward)

    arm_to_play = arms.optimal_arm

    for turn in range(num_turns):
        arm_reward = np.random.choice(arm_distributions[arm_to_play])
        arms.add_arm_reward(arm_to_play, arm_reward)


def epsilon_greedy(
    arms: MultiArmedBandit,
    arm_distributions: dict[int, list[float]],
    num_turns: int,
    num_exploration_turns: int,
    epsilon: float = 0.8,
) -> None:
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


def ucb(
    arms: MultiArmedBandit, arm_distributions: dict[int, list[float]], num_turns: int
) -> None:
    for turn in range(num_turns):
        ucb_map = {}
        for arm in arms.arms:
            reward = arms.get_arm_reward(arm)
            arm_plays = len(arms.arms[arm])
            total_plays = len(arms.regret)
            exploration_bonus = 2 * np.sqrt(
                (np.log(total_plays)) / (arm_plays)
            )
            ucb_map[arm] = exploration_bonus + reward
        arm_to_play = max(ucb_map, key=ucb_map.get)
        arm_reward = np.random.choice(arm_distributions[arm_to_play])
        arms.add_arm_reward(arm_to_play, arm_reward)
