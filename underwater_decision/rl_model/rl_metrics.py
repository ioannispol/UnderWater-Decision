from stable_baselines3.common.callbacks import BaseCallback


class CustomCallback(BaseCallback):
    def __init__(self, logger, verbose=0):
        super().__init__(verbose)
        self.logger = logger

    def _on_step(self):
        infos = self.locals.get('infos')
        if infos:
            for info in infos:
                if 'reward' in info:
                    self.logger.log_reward(info['reward'])
        return True


class MetricLogger:
    def __init__(self):
        self.rewards = []
        self.losses = []
        self.exploration_rates = []

    def log_reward(self, reward):
        self.rewards.append(reward)

    def log_loss(self, loss):
        self.losses.append(loss)

    def log_exploration_rate(self, rate):
        self.exploration_rates.append(rate)

    def get_average_reward(self, last_n_episodes=100):
        """Calculate the average reward for the last n episodes."""
        if len(self.rewards) < last_n_episodes:
            return sum(self.rewards) / len(self.rewards)
        else:
            return sum(self.rewards[-last_n_episodes:]) / last_n_episodes

    def reset(self):
        """Resets the stored metrics."""
        self.rewards = []
        self.losses = []
        self.exploration_rates = []
