from gym.envs.registration import register

register(
    id='labyrinth-v0',
    entry_point='gym_labyrinth.envs:LabyrinthEnv',
)
