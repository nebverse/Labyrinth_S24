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
import ray
from ray.rllib.algorithms.dreamerv3 import DreamerV3, DreamerV3Config
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.utils.replay_buffers import MultiAgentReplayBuffer
from ray.rllib.utils.framework import try_import_tf


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
        return frame

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
        self.waypoints = np.zeros((10, 2))  # Adjusted for the example
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
        return obs, reward, done, {"vec_obs": vec_obs}

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
        reward, self.previous_ball_position, self.waypoints, self.ball_hole = ball_detect_reward_calc.extract_reward(
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
        goal_position = waypoints[-1]
        nearest_waypoint, nearest_index = self._find_nearest_waypoint(ball_position, waypoints)
        interpolated_waypoints = self._get_interpolated_waypoints(waypoints, nearest_index, len(waypoints) - 1)

        next_waypoint = interpolated_waypoints[1]
        direction_vector = np.array(next_waypoint) - np.array(ball_position)
        norm = np.linalg.norm(direction_vector)
        if norm != 0:
            direction_vector = direction_vector / norm
        else:
            direction_vector = np.array([0, 0])

        interpolated_waypoints_flat = np.array(interpolated_waypoints).flatten().tolist()

        # Ensure the length is always 12
        if len(interpolated_waypoints_flat) < 10:
            interpolated_waypoints_flat += [0] * (10 - len(interpolated_waypoints_flat))

        return direction_vector.tolist() + interpolated_waypoints_flat[:10]

    def _find_nearest_waypoint(self, ball_position, waypoints):
        distances = np.linalg.norm(waypoints - np.array(ball_position), axis=1)
        nearest_index = np.argmin(distances)
        return waypoints[nearest_index], nearest_index

    def _get_interpolated_waypoints(self, waypoints, start_index, end_index, num_points=5):
        path_segment = np.array(waypoints[start_index:end_index+1])
        segment_distances = np.linalg.norm(np.diff(path_segment, axis=0), axis=1)
        cumulative_distances = np.cumsum(segment_distances)
        total_distance = cumulative_distances[-1]
        interpolated_distances = np.linspace(0, total_distance, num_points)

        interpolated_waypoints = [path_segment[0].tolist()]
        for d in interpolated_distances[1:]:
            for i in range(len(cumulative_distances)):
                if d <= cumulative_distances[i]:
                    t = (d - (cumulative_distances[i-1] if i > 0 else 0)) / (segment_distances[i] if segment_distances[i] != 0 else 1)
                    interp_point = path_segment[i] + t * (path_segment[i+1] - path_segment[i])
                    interpolated_waypoints.append(interp_point.tolist())
                    break
        return interpolated_waypoints

    def _get_image_patch(self, img, ball_position):
        x, y = ball_position
        img_patch = img[max(0, y-32):min(img.shape[0], y+32), max(0, x-32):min(img.shape[1], x+32)]
        resized_patch = cv2.resize(img_patch, (64, 64))
        return resized_patch

    def _check_done(self, img):
        return self.ball_hole

    def reset_physical_system(self):
        ray.get(self.hardware_actor.reset_positions.remote())


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
register_env("labyrinth_env", lambda config: LabyrinthEnv())
ModelCatalog.register_custom_model("custom_rnn", CustomRNN)

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
        num_gpus_per_learner=0,
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
