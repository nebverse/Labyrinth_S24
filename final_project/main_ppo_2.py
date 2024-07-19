import gym
from gym import spaces
import numpy as np
import cv2
import itertools
import pypot.dynamixel
import ball_detect_reward_calc as reward_calc
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.annotations import override
from ray.rllib.models.tf.misc import normc_initializer

tf1, tf, tfv = try_import_tf()

class LabyrinthEnv(gym.Env):
    def __init__(self):
        super(LabyrinthEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # 4 possible actions: tilt left, tilt right, tilt up, tilt down
        self.observation_space = spaces.Dict({
            "vec_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32),
            "img_obs": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        })  # Vector observations and image observations
        self.cap = cv2.VideoCapture(0)  # Initialize the webcam
        
        # Initialize the motors
        ports = pypot.dynamixel.get_available_ports()
        if not ports:
            raise IOError('No port available.')
        port = ports[-1]
        self.dxl_io = pypot.dynamixel.DxlIO(port)
        found_ids = self.dxl_io.scan()
        if len(found_ids) < 2:
            raise IOError('You should connect at least two motors on the bus for this test.')
        self.ids = found_ids[:2]
        self.dxl_io.enable_torque(self.ids)
        speed = dict(zip(self.ids, itertools.repeat(200)))
        self.dxl_io.set_moving_speed(speed)
        
        # Define HSV color ranges for detecting the ball and markers
        self.lower_red = np.array([160, 100, 100])
        self.upper_red = np.array([180, 255, 255])

        self.lower_green = np.array([35, 100, 100])
        self.upper_green = np.array([85, 255, 255])

        # Initialize previous ball position
        self.previous_ball_position = (0, 0)
        
        # Initialize waypoint
        self.waypoints = 0
        
        # Is ball in the hole
        self.ball_hole = False

    def step(self, action):
        self._take_action(action)
        _, img = self.cap.read()
        reward = self._compute_reward(img)
        done = self._check_done(img)
        obs = self._get_observation(img)
        info = {}
        return obs, reward, done, info

    def reset(self):
        self.reset_physical_system()
        _, img = self.cap.read()
        self.previous_ball_position = (0, 0)
        return self._get_observation(img)
        
    def reset_physical_system(self):
        # Ensure motors are reset to their initial positions
        self.dxl_io.set_goal_position({self.ids[0]: 0, self.ids[1]: 0})
        # Wait for the user to manually reset the ball
        input("Reset the ball on the board and press Enter to continue...")

    def render(self, mode='human'):
        pass

    def close(self):
        self.cap.release()

    def _take_action(self, action):
        # Send command to physical system to tilt labyrinth
        AMP = 15  # small value

        # Initialize position dictionary
        pos = {self.ids[0]: 0, self.ids[1]: 0}

        # Map action to motor commands
        if action == 0:  # tilt left
            pos[self.ids[0]] = -AMP
        elif action == 1:  # tilt right
            pos[self.ids[0]] = AMP
        elif action == 2:  # tilt up
            pos[self.ids[1]] = -AMP
        elif action == 3:  # tilt down
            pos[self.ids[1]] = AMP

        # Set the goal position for the motors
        self.dxl_io.set_goal_position(pos)

    def _compute_reward(self, img):
        # Compute the reward based on the position of the ball
        reward, self.previous_ball_position, self.waypoints, self.ball_hole = reward_calc.extract_reward(
            img, self.previous_ball_position,
            self.lower_red, self.upper_red,
            self.lower_green, self.upper_green
        )
        return reward
        
    def _get_observation(self, img):
        # Extract vector and image observations
        ball_position = self.previous_ball_position
        plate_angles = self._get_plate_angles()
        path_direction = self._get_path_direction(ball_position)
        vec_obs = np.array([ball_position[0], ball_position[1], plate_angles[0], plate_angles[1]] + path_direction)
        
        # Extract image patch centered around the ball position
        img_obs = self._get_image_patch(img, ball_position)
        
        return {"vec_obs": vec_obs, "img_obs": img_obs}
        
    def _get_plate_angles(self):
        # Get the current motor angles (plate inclination angles)
        angles = self.dxl_io.get_present_position(self.ids)
        plate_angles = (angles[0], angles[1])
        return plate_angles
        
    def _get_path_direction(self, ball_position):
        waypoints = self.waypoints
        goal_position = waypoints[-1]  # Assuming the last waypoint is the goal
        nearest_waypoint, nearest_index = self._find_nearest_waypoint(ball_position, waypoints)
        interpolated_waypoints = self._get_interpolated_waypoints(waypoints, nearest_index, len(waypoints) - 1)

        # The direction vector to the first interpolated waypoint
        next_waypoint = interpolated_waypoints[1]  # skip the first one as it is the nearest waypoint itself
        direction_vector = (next_waypoint[0] - ball_position[0], next_waypoint[1] - ball_position[1])
        norm = np.linalg.norm(direction_vector)
        if norm != 0:
            direction_vector = [direction_vector[0] / norm, direction_vector[1] / norm]
        else:
            direction_vector = [0, 0]

        # Flatten the list of the 5 interpolated waypoints to include in the observation
        interpolated_waypoints_flat = interpolated_waypoints.flatten().tolist()
    
        return direction_vector + interpolated_waypoints_flat

    def _find_nearest_waypoint(self, ball_position, waypoints):
        distances = np.linalg.norm(waypoints - np.array(ball_position), axis=1)
        nearest_index = np.argmin(distances)
        return waypoints[nearest_index], nearest_index
        
    def _get_interpolated_waypoints(self, waypoints, start_index, end_index, num_points=5):
        path_segment = waypoints[start_index:end_index+1]
        segment_distances = np.linalg.norm(np.diff(path_segment, axis=0), axis=1)
        cumulative_distances = np.cumsum(segment_distances)
        total_distance = cumulative_distances[-1]
        interpolated_distances = np.linspace(0, total_distance, num_points)

        interpolated_waypoints = [path_segment[0]]
        for d in interpolated_distances[1:]:
            for i in range(len(cumulative_distances)):
                if d <= cumulative_distances[i]:
                    t = (d - (cumulative_distances[i-1] if i > 0 else 0)) / (segment_distances[i] if segment_distances[i] != 0 else 1)
                    interp_point = path_segment[i] + t * (path_segment[i+1] - path_segment[i])
                    interpolated_waypoints.append(interp_point)
                    break
        return np.array(interpolated_waypoints)


    def _get_image_patch(self, img, ball_position):
        # Extract a 64x64 image patch centered around the ball position
        x, y = ball_position
        img_patch = img[max(0, y-32):min(img.shape[0], y+32), max(0, x-32):min(img.shape[1], x+32)]
        img_patch = cv2.resize(img_patch, (64, 64))
        return img_patch

    def _check_done(self, img):
        # Check if the episode is done based on the position of the ball
        ball_position = self.previous_ball_position
        return self._is_ball_at_goal(ball_position) or self.ball_hole

    def _is_ball_at_goal(self, ball_position, tolerance=10):
        goal_position = self.waypoints[-1]  # Assuming the last waypoint is the goal
        distance_to_goal = np.linalg.norm(np.array(ball_position) - np.array(goal_position))
        return distance_to_goal <= tolerance

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

config = PPOConfig().to_dict()
config["env"] = "LabyrinthEnv-v0"
config["num_workers"] = 1
config["model"] = {
    "custom_model": "custom_recurrent_model",
    "custom_model_config": {},
    "max_seq_len": 20,
}
config["framework"] = "tf"  # or "torch" if you prefer PyTorch

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
        trainer.optimizer.learn_on_batch(next_obs)
        obs = next_obs
        episode_reward += reward
    
    print(f"Episode {episode + 1}: Reward = {episode_reward}")
    input("Press Enter to start the next episode...")

env.close()
