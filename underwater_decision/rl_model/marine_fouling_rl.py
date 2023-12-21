import gym
from gym import spaces
import numpy as np
import pandas as pd


class MarineFoulingCleaningEnv(gym.Env):
    def __init__(self, dataset, max_steps=50):
        super(MarineFoulingCleaningEnv, self).__init__()

        # Load the dataset
        self.dataset = pd.read_csv(dataset)

        # Define a state space range based on the dataset
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([100, 100, 100, 100], dtype=np.float32),
            dtype=np.float32,
        )

        # Define an action space range
        self.action_space = spaces.Discrete(4)  # Assuming 4 different cleaning methods
        self.max_steps = max_steps
        self.current_step = 0
        self.success_threshold = 0.1

    def step(self, action):
        # Apply action and determine reward
        self.current_step += 1
        reward = self._calculate_reward(action)
        self.state = self._next_state(action)
        done = self._is_done()
        return self.state, reward, done, {}

    def reset(self):
        # Reset the state using a new sample from the dataset
        self.state = self.dataset.sample()[
            ["hardPerc", "hardmm", "softPerc", "softmm"]
        ].values.flatten()
        self.current_step = 0
        return self.state

    def render(self, mode="console"):
        if mode == "console":
            print(f"Current state: {self.state}")

    def seed(self, seed=None):
        np.random.seed(seed)

    def _calculate_reward(self, action):
        # Simulated effectiveness scores for each cleaning method
        effectiveness_scores = [0.8, 0.7, 0.9, 0.6]  # These values are hypothetical

        # Calculate the reward based on the effectiveness of the chosen action and the current state
        # For simplicity, let's assume the reward is proportional to the reduction in fouling
        reward = 0
        for i in range(4):  # Loop over the fouling components in the state
            reward += effectiveness_scores[action] * (100 - self.state[i]) / 100

        return reward

    def _next_state(self, action):
        # Define the efficiency of each cleaning method
        # These values represent the percentage reduction in fouling
        efficiency = {
            0: {"hard": 0.3, "soft": 0.2},  # Cleaning method 0
            1: {"hard": 0.1, "soft": 0.5},  # Cleaning method 1
            2: {"hard": 0.4, "soft": 0.1},  # Cleaning method 2
            3: {"hard": 0.2, "soft": 0.3},  # Cleaning method 3
        }

        # Update the state based on the efficiency of the chosen action
        hard_fouling_reduction = efficiency[action]["hard"]
        soft_fouling_reduction = efficiency[action]["soft"]

        # Apply reduction to hard and soft fouling components
        self.state = [
            max(0, value - hard_fouling_reduction * value)
            if index < 2
            else max(0, value - soft_fouling_reduction * value)
            for index, value in enumerate(self.state)
        ]

        return self.state

    def _is_done(self):
        # Check if the fouling levels are below the success threshold
        fouling_levels = self.state[
            :4
        ]  # Assuming the first four elements are fouling levels
        fouling_reduced = all(
            fouling <= self.success_threshold for fouling in fouling_levels
        )

        # Check if maximum number of steps has been reached
        max_steps_reached = self.current_step >= self.max_steps

        # The episode is done if successful cleaning is achieved or max steps are reached
        return fouling_reduced or max_steps_reached
