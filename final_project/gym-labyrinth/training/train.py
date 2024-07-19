# training.py img-obs

import ray
from ray import tune
from ray.rllib.algorithms.dreamerv3 import DreamerV3, DreamerV3Config, dreamerv3_catalog
from ray.tune.registry import register_env
from gym_labyrinth.envs.labyrinth_env import LabyrinthEnv
from gymnasium.utils.env_checker import check_env
from ray.rllib.models import ModelCatalog
import gc
import logging
import time


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Register the custom environment
def create_labyrinth_env(config):
    env = LabyrinthEnv()
    check_env(env)
    return env

register_env("labyrinth-v0", create_labyrinth_env)
# ModelCatalog.register_custom_model("custom_dreamer_model", CustomDreamerModel)

# Configuration for DreamerV3 with the custom model
config = DreamerV3Config()\
    .environment('labyrinth-v0')\
    .framework("tf2")\
    .training(
        model_size="XS",
        # batch_size_B=4,
        # batch_length_T=16,
        horizon_H=15,
        gamma=0.997,
        gae_lambda=0.95,
        entropy_scale=3e-4,
        return_normalization_decay=0.99,
        train_critic=True,
        train_actor=True,
        intrinsic_rewards_scale=0.1,
        world_model_lr=1e-4,
        actor_lr=3e-5,
        critic_lr=3e-5,
        world_model_grad_clip_by_global_norm=1000.0,
        critic_grad_clip_by_global_norm=100.0,
        actor_grad_clip_by_global_norm=100.0,
        symlog_obs="auto",
        use_float16=False,
        replay_buffer_config={
            "type": "EpisodeReplayBuffer",
            "capacity": int(1e6),
        }
    )\
    .rollouts(num_rollout_workers=1)\
    .learners(num_learners=1, num_gpus_per_learner=1)\
    # .model(custom_model="custom_dreamer_model")

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Initialize DreamerV3 Trainer
trainer = DreamerV3(config=config)

# Training loop
num_episodes = 1000

for episode in range(num_episodes):
    try:
        obs, _ = trainer.workers.local_worker().env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = trainer.compute_action(obs)
            next_obs, reward, done, _, info = trainer.workers.local_worker().env.step(action)
            obs = next_obs
            episode_reward += reward

        results = trainer.train()
        logger.info(f"Episode {episode + 1}: Reward = {episode_reward}")
        logger.info(f"Training Results: {results}")

        if episode % 10 == 0:
            checkpoint = trainer.save()
            logger.info(f"Checkpoint saved at: {checkpoint}")

        if episode % config.gc_frequency_train_steps == 0:
            gc.collect()

    except Exception as e:
        logger.error(f"Error during episode {episode + 1}: {e}")
        time.sleep(1)

# Clean up
trainer.stop()
ray.shutdown()

# ----------------------------------------------------------------#

# training.py img-obs tune

# import ray
# from ray import tune
# from ray.rllib.algorithms.dreamerv3 import DreamerV3, DreamerV3Config
# from ray.tune.registry import register_env
# from gym_labyrinth.envs.labyrinth_env import LabyrinthEnv
# from gymnasium.utils.env_checker import check_env
# import gc
# import logging
# import time
# from ray.tune.logger import UnifiedLogger
# import os


# # Setup logging
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

# # Register the custom environment
# def create_labyrinth_env(config):
#     env = LabyrinthEnv()
#     check_env(env)
#     return env

# register_env("labyrinth-v0", create_labyrinth_env)

# # Configuration for DreamerV3 with the custom model
# config = {
#     "environment": "labyrinth-v0",
#     "framework": "tf2",
#     "model_size": tune.choice(["XS", "S", "M", "L"]),
#     "batch_size_B": tune.choice([4, 8, 16]),
#     "batch_length_T": 16,
#     "horizon_H": 15,
#     "gamma": 0.997,
#     "gae_lambda": 0.95,
#     "entropy_scale": 3e-4,
#     "return_normalization_decay": 0.99,
#     "train_critic": True,
#     "train_actor": True,
#     "intrinsic_rewards_scale": 0.1,
#     "world_model_lr": tune.loguniform(1e-5, 1e-3),
#     "actor_lr": 3e-5,
#     "critic_lr": 3e-5,
#     "world_model_grad_clip_by_global_norm": 1000.0,
#     "critic_grad_clip_by_global_norm": 100.0,
#     "actor_grad_clip_by_global_norm": 100.0,
#     "symlog_obs": "auto",
#     "use_float16": False,
#     "replay_buffer_config": {
#         "type": "EpisodeReplayBuffer",
#         "capacity": int(1e6),
#     },
#     "num_rollout_workers": 1,
#     "num_learners": 1,
#     "num_gpus_per_learner": 1,
# }

# print("sdcs")
# # Initialize Ray
# ray.init(ignore_reinit_error=True)  # , _temp_dir="G:/My Drive/MIR/a/Internship/Summer internship/CYBERRUNNER/project/my_code/final_project/gym-labyrinth"

# print("vsa")
# # Setup the experiment
# experiment = tune.Experiment(
#     name="DreamerV3_Labyrinth_Tuning",
#     run="DreamerV3",
#     config=config,
#     stop={"training_iteration": 100},
#     num_samples=10
# )

# print("fd")
# # Start tuning
# results = tune.run(experiment)

# # print("vsa")
# # # Then use the custom_log_creator in your Tuner configuration
# # tuner = tune.Tuner(
# #     trainable="DreamerV3",
# #     param_space=config
# # )

# # print("fd")
# # results = tuner.fit()


# # Clean up
# ray.shutdown()



# --------------------------------------------------------------- #

# training.py for vec-obs

# import ray
# from ray import tune
# from ray.rllib.algorithms.dreamerv3 import DreamerV3, DreamerV3Config
# from ray.tune.registry import register_env
# from gym_labyrinth.envs.labyrinth_env import LabyrinthEnv
# from gymnasium.utils.env_checker import check_env
# import gc
# import logging

# # Setup logging
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

# # Register the custom environment
# def create_labyrinth_env(config):
#     env = LabyrinthEnv()
#     check_env(env)
#     return env

# register_env("labyrinth-v0", create_labyrinth_env)

# # Configuration for DreamerV3
# config = DreamerV3Config()\
#     .environment('labyrinth-v0')\
#     .framework("tf2")\
#     .training(
#         model_size="XS",  # Adjust based on the complexity of your task and available resources
#         batch_size_B=16,  # Initial batch size, can be adjusted
#         batch_length_T=64,  # Length of each batch
#         horizon_H=15,  # Horizon length for dreaming
#         gamma=0.997,  # Discount factor
#         gae_lambda=0.95,  # GAE lambda for advantage estimation
#         entropy_scale=3e-4,  # Scale for entropy regularization
#         return_normalization_decay=0.99,  # Decay for return normalization
#         train_critic=True,  # Train critic network
#         train_actor=True,  # Train actor network
#         intrinsic_rewards_scale=0.1,  # Scale for intrinsic rewards
#         world_model_lr=1e-4,  # Learning rate for world model
#         actor_lr=3e-5,  # Learning rate for actor
#         critic_lr=3e-5,  # Learning rate for critic
#         world_model_grad_clip_by_global_norm=1000.0,  # Gradient clipping for world model
#         critic_grad_clip_by_global_norm=100.0,  # Gradient clipping for critic
#         actor_grad_clip_by_global_norm=100.0,  # Gradient clipping for actor
#         symlog_obs="auto",  # Symlog observations, set to "auto" for automatic handling
#         use_float16=False,  # Use mixed precision training (float16)
#         replay_buffer_config={
#             "type": "EpisodeReplayBuffer",
#             "capacity": int(1e6),  # Capacity for the replay buffer
#         }
#     )\
#     .rollouts(
#         num_rollout_workers=1,  # Number of rollout workers, 1 for physical systems
#     )\
#     .learners(
#         num_learners=2,  # Number of learners, ensure system can handle this
#         num_cpus_per_learner=1,  # CPUs per learner
#         num_gpus_per_learner=1,  # GPUs per learner, adjust based on available hardware
#     )\
#     .reporting(
#         report_individual_batch_item_stats=False,
#         report_dream_data=False,
#         report_images_and_videos=False,
#     )

# # Initialize Ray
# ray.init(ignore_reinit_error=True)

# # Initialize DreamerV3 Trainer
# trainer = DreamerV3(config=config)

# # Training loop
# num_episodes = 1000

# for episode in range(num_episodes):
#     obs, _ = trainer.workers.local_worker().env.reset()
#     done = False
#     episode_reward = 0
    
#     while not done:
#         action = trainer.compute_action(obs)  # Compute action for the current observation
#         next_obs, reward, done, info = trainer.workers.local_worker().env.step(action)
#         obs = next_obs
#         episode_reward += reward
    
#     results = trainer.train()
#     logger.info(f"Episode {episode + 1}: Reward = {episode_reward}")
#     logger.info(f"Training Results: {results}")

#     if episode % 10 == 0:
#         checkpoint = trainer.save()
#         logger.info(f"Checkpoint saved at: {checkpoint}")

#     # Perform garbage collection to prevent memory leaks
#     if episode % config.gc_frequency_train_steps == 0:
#         gc.collect()

# # Clean up
# trainer.stop()
# ray.shutdown()

# -------------------------------------------------- #

# training.py

# training.py

# import gym
# from gym_labyrinth.envs.labyrinth_env import LabyrinthEnv
# from sheeprl.algos.dreamer_v3 import DreamerV3
# from sheeprl.utils import set_seed
# from custom_models.custom_dreamer_model import CustomDreamerModel
# import torch

# # Register custom environment
# gym.envs.register(
#     id='Labyrinth-v0',
#     entry_point='gym_labyrinth.envs:LabyrinthEnv',
# )

# # Set the seed for reproducibility
# set_seed(42)

# # Initialize the custom environment
# env = gym.make('Labyrinth-v0')

# # Define the DreamerV3 configuration
# dreamer_config = {
#     "model_size": "XS",  # Adjust based on the complexity of your task and available resources
#     "batch_size": 32,
#     "sequence_length": 50,
#     "gamma": 0.99,
#     "lambda": 0.95,
#     "model_lr": 1e-4,
#     "actor_lr": 1e-4,
#     "critic_lr": 1e-4,
#     "replay_buffer_size": int(1e6),
#     "use_cuda": torch.cuda.is_available(),  # Automatically use CUDA if available
#     "custom_model": CustomDreamerModel,  # Specify the custom model here
#     "custom_model_config": {
#         "input_channels": 3,
#         "hidden_channels": [32, 64, 128],  # Adjust based on your needs
#         "output_dim": 256,  # Example configuration for the custom model
#     },
# }

# # Initialize the DreamerV3 agent
# agent = DreamerV3(env, config=dreamer_config)

# # Training loop
# num_episodes = 1000
# for episode in range(num_episodes):
#     obs = env.reset()
#     done = False
#     episode_reward = 0

#     while not done:
#         action = agent.act(obs)
#         next_obs, reward, done, info = env.step(action)
#         agent.observe(next_obs, action, reward, done)
#         obs = next_obs
#         episode_reward += reward

#     agent.train()
#     print(f"Episode {episode + 1}: Reward = {episode_reward}")

#     if episode % 10 == 0:
#         agent.save(f"checkpoint_{episode + 1}.pth")

# # Clean up
# env.close()
