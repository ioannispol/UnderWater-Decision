from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

from underwater_decision.rl_model.marine_fouling_rl import MarineFoulingCleaningEnv
from underwater_decision.rl_model.rl_metrics import CustomCallback, MetricLogger

# Initialize your custom environment
env = MarineFoulingCleaningEnv(
    "/workspaces/UnderWater-Decision/data/default_synthetic_dataset.csv"
)

# Initialize the metric logger
logger = MetricLogger()

# Wrap it if necessary (for instance to have vectorized environments)
# This is optional and depends on your specific needs
env = make_vec_env(lambda: env, n_envs=1)

# Create the agent
model = DQN("MlpPolicy", env, verbose=1)

# Initialize the callback
callback = CustomCallback(logger)

# Train the agent
model.learn(total_timesteps=10000, callback=callback)

# Save the agent
model.save("data/dqn_marine_fouling_cleaning")
