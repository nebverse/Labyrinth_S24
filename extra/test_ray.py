import ray
from ray.rllib.agents.ppo import PPOTrainer

ray.init()

config = {
    "env": "CartPole-v0",  # Simple environment for testing
    "framework": "tf"
}

trainer = PPOTrainer(config=config)

print("Ray and RLlib are working correctly.")
