import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import itertools
import pypot.dynamixel
import ball_detect_reward_calc as reward_calc
from ray.tune.registry import register_env
import tensorflow as tf
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.models import ModelCatalog
import gym
import numpy as np
import ray
from ray.rllib.algorithms.dreamerv3 import DreamerV3, DreamerV3Config
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.utils.replay_buffers import MultiAgentReplayBuffer
from ray.rllib.utils.framework import try_import_tf

# Remote function to initialize webcam
@ray.remote
def init_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open webcam")
    return cap

# Remote function to initialize hardware (Dynamixel motors)
@ray.remote
def init_hardware():
    port = 'COM7'
    try:
        dxl_io = pypot.dynamixel.DxlIO(port)
        found_ids = dxl_io.scan()
        if len(found_ids) < 2:
            raise IOError('You should connect at least two motors on the bus for this test.')
        dxl_io.enable_torque(found_ids)
        speed = dict(zip(found_ids, itertools.repeat(200)))
        dxl_io.set_moving_speed(speed)
        return dxl_io, found_ids
    except Exception as e:
        raise Exception(f"Error opening port {port}: {e}")

class LabyrinthEnv(gym.Env):
    def __init__(self):
        super(LabyrinthEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # 4 possible actions: tilt left, tilt right, tilt up, tilt down
        self.observation_space = spaces.Dict({
            "vec_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32),
            "img_obs": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        })  # Vector observations and image observations

        self.cap = ray.get(init_webcam.remote())  # Initialize the webcam remotely
        self.dxl_io, self.ids = ray.get(init_hardware.remote())  # Initialize hardware remotely
        
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
        AMP = 10  # small value

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
    
class CustomRNN(RecurrentNetwork):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.obs_size = obs_space["vec_obs"].shape[0]
        self.cell_size = 256

        # Define input layers
        self.inputs = tf.keras.layers.Input(shape=(None, self.obs_size), name="inputs")
        self.state_in_h = tf.keras.layers.Input(shape=(self.cell_size,), name="h_in")
        self.state_in_c = tf.keras.layers.Input(shape=(self.cell_size,), name="c_in")
        self.seq_lens = tf.keras.layers.Input(shape=(), name="seq_lens", dtype=tf.int32)

        # Define LSTM cell
        self.lstm = tf.keras.layers.LSTM(
            units=self.cell_size, return_sequences=True, return_state=True, name="lstm"
        )

        lstm_out, state_h, state_c = self.lstm(
            inputs=self.inputs,
            initial_state=[self.state_in_h, self.state_in_c]
        )

        # Output layer
        self.logits = tf.keras.layers.Dense(
            units=num_outputs,
            activation=None,
            kernel_initializer=normc_initializer(0.01),
            name="logits"
        )(lstm_out)

        self.values = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=normc_initializer(0.01),
            name="values"
        )(lstm_out)

        self.rnn_model = tf.keras.Model(
            inputs=[self.inputs, self.state_in_h, self.state_in_c, self.seq_lens],
            outputs=[self.logits, self.values, state_h, state_c]
        )

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        logits, values, state_h, state_c = self.rnn_model([inputs, state[0], state[1], seq_lens])
        return logits, [state_h, state_c]

    @override(RecurrentNetwork)
    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]
    
tf1, tf, tfv = try_import_tf()

# Register the custom environment and model
# ModelCatalog.register_custom_model("custom_rnn", CustomRNN)
register_env("labyrinth_env", lambda config: LabyrinthEnv())

# Define the configuration for DreamerV3
config = (
    DreamerV3Config()
    .environment("labyrinth_env")
    .framework(eager_tracing=False)
    .env_runners(num_env_runners=2)
    .training(
        batch_size_B=4,
        horizon_H=5,
        batch_length_T=16,
        model_size="nano",
        symlog_obs=True,
        use_float16=False,
    )
    .learners(
        num_learners=2,
        num_cpus_per_learner=1,
        num_gpus_per_learner=1,
    )
)

config.rollout_fragment_length = 50  # Correctly set the rollout_fragment_length

# Build the DreamerV3 algorithm
algo = config.build()

# Define augmentation function
def augment_data(batch):
    # Implement data augmentation techniques such as mirroring
    augmented_batch = batch.copy()
    # Apply transformations to observations
    vec_obs = augmented_batch["vec_obs"]
    img_obs = augmented_batch["img_obs"]

    # Apply random mirroring
    if np.random.rand() > 0.5:
        vec_obs[:, 0] = -vec_obs[:, 0]  # Mirroring x position
        img_obs = img_obs[:, ::-1, :, :]  # Horizontal flip

    if np.random.rand() > 0.5:
        vec_obs[:, 1] = -vec_obs[:, 1]  # Mirroring y position
        img_obs = img_obs[:, :, ::-1, :]  # Vertical flip

    augmented_batch["vec_obs"] = vec_obs
    augmented_batch["img_obs"] = img_obs

    return augmented_batch

# Create a replay buffer
replay_buffer = MultiAgentReplayBuffer(capacity=10000, env=LabyrinthEnv(), policy_map={})

# Training loop with replay buffer and augmentation
num_iterations = 1000
for _ in range(num_iterations):
    # Collect experiences
    rollout_worker = RolloutWorker(env_creator=lambda _: LabyrinthEnv(), policy_config=config)
    experiences = rollout_worker.sample()

    # Add experiences to the replay buffer
    replay_buffer.add_batch(experiences)

    # Sample from the replay buffer and perform data augmentation
    batch = replay_buffer.sample()
    augmented_batch = augment_data(batch)

    # Train with the augmented batch
    result = algo.train_with_batch(augmented_batch)
    print(result)

algo.stop()
ray.shutdown()