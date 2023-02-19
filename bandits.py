"""Multi-Armed Bandits Implementation."""
from __future__ import annotations

import random


class MultiArmedBandit:
    """Implementation of a Multi-Armed Bandit Engine."""

    def __init__(self):
        self.arms = {}
        self.arm_regret = {}

    @property
    def num_arms(self) -> int:
        """Get the number of arms for this instance.

        Returns:
            Number of arms
        """
        return len(self.arms)

    def add_arm(self, arm_number: int) -> None:
        """Adds an arm to the instance.

        Args:
            arm_number: index of the arm
        """
        self.arms[arm_number] = []
        self.arm_regret[arm_number] = []

    def get_random_arm(self) -> int:
        """Draw a random arm.

        Returns:
            arm
        """
        arm = random.choice(list(self.arms.keys()))
        return arm

    def get_arm_reward(self, arm_number: int) -> float | int:
        """Get the reward (r bar) for a specific arm.

        Args:
            arm_number: arm index/number to use

        Returns:
            reward for that arm
        """
        total_reward = sum(self.arms[arm_number])
        num_plays = len(self.arms[arm_number])

        if num_plays:
            return total_reward / num_plays
        else:
            return 0

    def add_arm_reward(self, arm_number: int, reward: float | int) -> None:
        """Set or add a reward for some arm

        Args:
            arm_number: what arm to add to
            reward: reward we are adding
        """
        self.arms[arm_number].append(reward)
        # also update regret
        self.arm_regret[arm_number].append(self.get_arm_regret(arm_number))

    def get_arm_rewards(self) -> dict[int, float | int | None]:
        """Wrapper of get_arm_reward for getting all of the arm rewards.

        Returns:
            dict of {arm: reward} for each arm in self.arms
        """
        rewards = {}
        for arm_number in self.arms:
            rewards[arm_number] = self.get_arm_reward(arm_number)

        return rewards

    @property
    def optimal_arm(self) -> int:
        """Calculate and return the current optimal arm.

        Raises:
            ValueError: If nothing has been played yet

        Returns:
            optimal_arm_number
        """
        if not self.arms:
            raise ValueError("Cannot get optimal arm with no plays.")
        rewards = self.get_arm_rewards()
        optimal_arm = max(rewards, key=rewards.get)

        return optimal_arm

    def get_arm_regret(self, arm_number: int) -> float | int:
        """Calculate the regret of a given arm.

        Args:
            arm_number: arm to use

        Raises:
            ValueError: If no turns have been played for that arm

        Returns:
            the regret
        """
        turns = self.arms[arm_number]
        if not turns:
            raise ValueError(f"Arm {arm_number} has no plays!")

        optimal_arm = self.optimal_arm
        optimal_arm_reward = self.get_arm_reward(optimal_arm)
        regret = sum([optimal_arm_reward - turn for turn in turns])

        return regret
