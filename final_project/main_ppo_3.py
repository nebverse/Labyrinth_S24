import gym
from gym import spaces
import numpy as np
import cv2
import itertools
import pypot.dynamixel
import ball_detect_reward_calc

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
import tensorflow as tf
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()

import pkg_resources

@ray.remote
class WebcamActor:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Could not read from webcam")
        return frame.astype(np.uint8)

    def close(self):
        self.cap.release()

@ray.remote
class HardwareActor:
    def __init__(self, port='COM7'):
        try:
            self.dxl_io = pypot.dynamixel.DxlIO(port)
        except Exception as e:
            raise Exception(f"Error opening port {port}: {e}")

        found_ids = self.dxl_io.scan()
        if len(found_ids) < 2:
            raise IOError('You should connect at least two motors on the bus for this test.')
        self.ids = found_ids[:2]
        self.dxl_io.enable_torque(self.ids)
        speed = dict(zip(self.ids, itertools.repeat(200)))
        self.dxl_io.set_moving_speed(speed)

    def set_goal_position(self, pos):
        self.dxl_io.set_goal_position(pos)

    def get_present_position(self):
        return self.dxl_io.get_present_position(self.ids)

    def reset_positions(self):
        self.dxl_io.set_goal_position({self.ids[0]: 0, self.ids[1]: 0})

    def close(self):
        self.dxl_io.close()

class LabyrinthEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(LabyrinthEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)  # Only img_obs
        print(f"[DEBUG] Observation space in __init__: {self.observation_space}")

        self.webcam_actor = WebcamActor.remote()
        self.hardware_actor = HardwareActor.remote(port='COM7')

        self.lower_red = np.array([160, 100, 100])
        self.upper_red = np.array([180, 255, 255])
        self.lower_green = np.array([35, 100, 100])
        self.upper_green = np.array([85, 255, 255])

        self.previous_ball_position = (0, 0)

        self.edge_points = np.loadtxt('path_edge_points.txt')
        self.waypoints = [(x, y) for x, y in self.edge_points]
        # print(f"waypoints: {self.waypoints}")

        self.ball_hole = False

    def reset(self, seed=None, options=None):
        print("[DEBUG] Entered reset")
        super().reset(seed=seed)  # Call to the parent class to handle the seed
        self.reset_physical_system()
        img = ray.get(self.webcam_actor.read_frame.remote())
        self.previous_ball_position = (0, 0)
        obs = self._get_observation(img)
        print(f"[DEBUG] Observation in reset: {obs}")
        return obs, {}

    def step(self, action):
        print(f"[DEBUG] Entered step with action: {action}")
        self._take_action(action)
        img = ray.get(self.webcam_actor.read_frame.remote())
        reward = self._compute_reward(img)
        done = self._check_done(img)
        obs = self._get_observation(img)
        vec_obs = self._get_vec_observation()
        print(f"[DEBUG] Observation in step: {obs}, vec_obs: {vec_obs}")
        return obs, reward, done, done, {"vec_obs": vec_obs}

    def render(self, mode='human'):
        pass

    def close(self):
        ray.get(self.webcam_actor.close.remote())
        ray.get(self.hardware_actor.close.remote())

    def _check_observation_space(self, obs):
        print(f"[DEBUG] Checking observation space: {obs}")
        assert self.observation_space.contains(obs), f"Observation {obs} is not within the observation space {self.observation_space}"

    def _take_action(self, action):
        AMP = 10
        pos = {0: 0, 1: 0}
        if action == 0:
            pos[0] = -AMP
        elif action == 1:
            pos[0] = AMP
        elif action == 2:
            pos[1] = -AMP
        elif action == 3:
            pos[1] = AMP
        ray.get(self.hardware_actor.set_goal_position.remote(pos))

    def _compute_reward(self, img):
        reward, self.previous_ball_position, _, self.ball_hole = ball_detect_reward_calc.extract_reward(
            img, self.previous_ball_position,
            self.lower_red, self.upper_red,
            self.lower_green, self.upper_green
        )
        return reward

    def _get_observation(self, img):
        ball_position = self.previous_ball_position
        img_obs = self._get_image_patch(img, ball_position)
        return img_obs

    def _get_vec_observation(self):
        ball_position = self.previous_ball_position
        plate_angles = ray.get(self.hardware_actor.get_present_position.remote())
        path_direction = self._get_path_direction(ball_position)
        vec_obs = np.array([ball_position[0], ball_position[1], plate_angles[0], plate_angles[1]] + path_direction, dtype=np.float32)

        # Verify the length of vec_obs
        assert len(vec_obs) == 16, f"vec_obs length is incorrect: {len(vec_obs)}"

        return vec_obs

    def _get_path_direction(self, ball_position):
        waypoints = self.waypoints
        print(f"[DEBUG] waypoints: {waypoints}")  # Debug print to check waypoints
        if len(waypoints) == 0:
            print(f"[ERROR] Waypoints are not initialized correctly: {waypoints}")
            return [0.0] * 12

        goal_position = waypoints[-1]

        # Find the nearest waypoint to the ball
        nearest_waypoint, nearest_index = self._find_nearest_waypoint(ball_position, waypoints)
        print(f"[DEBUG] Nearest waypoint: {nearest_waypoint} at index {nearest_index}")

        # Interpolate waypoints between nearest waypoint and goal position
        path_segment = waypoints[nearest_index:]
        interpolated_waypoints = self._interpolate_path(path_segment, num_points=5)

        # Ensure that the points include the nearest waypoint and the goal position
        if interpolated_waypoints[0] != nearest_waypoint:
            interpolated_waypoints = [nearest_waypoint] + interpolated_waypoints
        if interpolated_waypoints[-1] != goal_position:
            interpolated_waypoints.append(goal_position)

        # Calculate direction vector
        next_waypoint = interpolated_waypoints[1]
        direction_vector = np.array(next_waypoint) - np.array(ball_position)
        norm = np.linalg.norm(direction_vector)
        if norm != 0:
            direction_vector = direction_vector / norm
        else:
            direction_vector = np.array([0, 0])

        # Flatten interpolated waypoints and ensure length is always 12
        interpolated_waypoints_flat = list(itertools.chain.from_iterable(interpolated_waypoints))
        if len(interpolated_waypoints_flat) < 10:
            interpolated_waypoints_flat += [0] * (10 - len(interpolated_waypoints_flat))

        return direction_vector.tolist() + interpolated_waypoints_flat[:10]

    def _find_nearest_waypoint(self, ball_position, waypoints):
        distances = np.linalg.norm(np.array(waypoints) - np.array(ball_position), axis=1)
        nearest_index = np.argmin(distances)
        return waypoints[nearest_index], nearest_index

    def _interpolate_path(self, path_segment, num_points=5):
        """Interpolates waypoints along the given path segment."""
        if len(path_segment) == 1:
            return path_segment  # No interpolation needed if only one point

        segment_distances = np.linalg.norm(np.diff(path_segment, axis=0), axis=1)
        cumulative_distances = np.cumsum(segment_distances)
        total_distance = cumulative_distances[-1]
        interpolated_distances = np.linspace(0, total_distance, num_points - 2)  # Exclude start and end

        interpolated_waypoints = [path_segment[0]]
        for d in interpolated_distances:
            for i in range(len(cumulative_distances)):
                if d <= cumulative_distances[i]:
                    t = (d - (cumulative_distances[i-1] if i > 0 else 0)) / (segment_distances[i] if segment_distances[i] != 0 else 1)
                    interp_point = np.array(path_segment[i]) + t * (np.array(path_segment[i+1]) - np.array(path_segment[i]))
                    interpolated_waypoints.append(tuple(interp_point))
                    break
        interpolated_waypoints.append(path_segment[-1])  # Include the end point
        return interpolated_waypoints

    def _get_image_patch(self, img, ball_position):
        x, y = ball_position
        img_patch = img[max(0, y-32):min(img.shape[0], y+32), max(0, x-32):min(img.shape[1], x+32)]
        resized_patch = cv2.resize(img_patch, (64, 64))
        return resized_patch.astype(np.uint8)

    def _check_done(self, img):
        return self.ball_hole

    def reset_physical_system(self):
        ray.get(self.hardware_actor.reset_positions.remote())

class CustomRecurrentModel(RecurrentNetwork):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CustomRecurrentModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        
        # Define the layers of the custom model
        self.cell_size = 256  # Size of the LSTM cell
        
        # Input layers for vector and image observations
        self.vec_obs_input = tf.keras.layers.Input(shape=(obs_space.spaces["vec_obs"].shape[0],), name="vec_obs")
        self.img_obs_input = tf.keras.layers.Input(shape=obs_space.spaces["img_obs"].shape, name="img_obs")
        
        # Convolutional layers for image observations
        conv1 = tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(self.img_obs_input)
        conv2 = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv1)
        conv3 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv2)
        img_obs_flat = tf.keras.layers.Flatten()(conv3)
        
        # Concatenate vector and processed image observations
        concat_layer = tf.keras.layers.Concatenate(axis=1)([self.vec_obs_input, img_obs_flat])
        
        # LSTM layer
        lstm_out, self.state_h, self.state_c = tf.keras.layers.LSTM(self.cell_size, return_sequences=True, return_state=True)(tf.expand_dims(concat_layer, axis=1))
        
        # Output layer
        self.logits = tf.keras.layers.Dense(num_outputs, activation=None, kernel_initializer=normc_initializer(0.01))(lstm_out)
        self.value_function_layer = tf.keras.layers.Dense(1, activation=None, kernel_initializer=normc_initializer(0.01))(lstm_out)
        
        # Keras model
        self.model = tf.keras.Model(inputs=[self.vec_obs_input, self.img_obs_input], outputs=[self.logits, self.value_function_layer])
        
    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        vec_obs, img_obs = inputs["vec_obs"], inputs["img_obs"]
        logits, value_function, state_h, state_c = self.model([vec_obs, img_obs, state], training=False)
        return logits, [state_h, state_c]
    
    @override(RecurrentNetwork)
    def get_initial_state(self):
        return [tf.zeros(self.cell_size, dtype=tf.float32), tf.zeros(self.cell_size, dtype=tf.float32)]
    
    @override(RecurrentNetwork)
    def value_function(self):
        return tf.reshape(self.value_function_layer, [-1])

# Register the custom model
ModelCatalog.register_custom_model("custom_recurrent_model", CustomRecurrentModel)

# Initialize the environment and trainer
ray.init(ignore_reinit_error=True)
register_env("LabyrinthEnv-v0", lambda config: LabyrinthEnv())

config = PPOConfig().environment(env="LabyrinthEnv-v0").rollouts(num_rollout_workers=1).training(model={
    "custom_model": "custom_recurrent_model",
    "custom_model_config": {},
    "max_seq_len": 20,
}).framework("tf").to_dict()

trainer = PPO(config=config)

# Training loop
num_episodes = 1000
env = LabyrinthEnv()

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        action = trainer.compute_single_action(obs)
        next_obs, reward, done, info = env.step(action)
        # trainer.optimizer.learn_on_batch(next_obs)
        obs = next_obs
        episode_reward += reward
    
    print(f"Episode {episode + 1}: Reward = {episode_reward}")
    input("Press Enter to start the next episode...")

env.close()
