"""Some classes to modify the environment during training."""

from __future__ import annotations

from typing import Any

import numpy as np
import yaml
from gymnasium.spaces import Box


class ObservationParser:
    """Class to parse the observation space of the firmware environment."""

    def __init__(
        self,
        n_gates: int,
        n_obstacles: int,
        drone_pos_limits: list = [3, 3, 2],
        drone_yaw_limits: list = [np.pi],
        gate_pos_limits: list = [5, 5, 5],
        gate_yaw_limits: list = [np.pi],
        gate_in_range_limits: list = [1],
        obstacle_pos_limits: list = [5, 5, 5],
        obstacle_in_range_limits: list = [1],
    ):
        """Initialize the observation parser."""
        self.n_gates = n_gates
        self.n_obstacles = n_obstacles

        obs_limits = (
            drone_pos_limits
            + drone_yaw_limits
            + gate_pos_limits * n_gates
            + gate_yaw_limits * n_gates
            + gate_in_range_limits * n_gates
            + obstacle_pos_limits * n_obstacles
            + obstacle_in_range_limits * n_obstacles
            + [n_gates]
        )
        obs_limits_high = np.array(obs_limits)
        obs_limits_low = np.concatenate([-obs_limits_high[:-1], [-1]])
        self.observation_space = Box(obs_limits_low, obs_limits_high, dtype=np.float32)

        # Observable variables
        self.drone_pos = None
        self.drone_yaw = None
        self.gates_pos = None
        self.gates_yaw = None
        self.gates_in_range = None
        self.obstacles_pos = None
        self.obstacles_in_range = None
        self.gate_id = None

        # Hidden states that are not part of the observation space
        self.just_passed_gate: bool = False

    def uninitialized(self) -> bool:
        """Check if the observation parser is uninitialized."""
        return self.drone_pos is None

    def out_of_bounds(self) -> bool:
        """Check if the drone is out of bounds."""
        return not self.observation_space.contains(self.get_observation())

    def update(self, obs: np.ndarray, info: dict[str, Any]):
        """Update the observation parser with the new observation and info dict.

        Remark:
            We do not update the gate height here, the info dict does not contain this information.

        Args:
            obs: The new observation.
            info: The new info dict.
        """
        self.drone_pos = obs[0:6:2]
        self.drone_yaw = obs[8]
        self.gates_pos = info["gates_pose"][:, :3]
        self.gates_yaw = info["gates_pose"][:, 5]
        self.gates_in_range = info["gates_in_range"]
        self.obstacles_pos = info["obstacles_pose"][:, :3]
        self.obstacles_in_range = info["obstacles_in_range"]
        self.just_passed_gate = self.gate_id != info["current_gate_id"]
        self.gate_id = info["current_gate_id"]

    def get_observation(self) -> np.ndarray:
        """Return the current observation.

        Returns:
            The current observation.
        """
        obs = np.concatenate(
            [
                self.drone_pos,
                [self.drone_yaw],
                self.gates_pos.flatten(),
                self.gates_yaw,
                self.gates_in_range,
                self.obstacles_pos.flatten(),
                self.obstacles_in_range,
                [self.gate_id],
            ]
        )
        return obs.astype(np.float32)


class Rewarder:
    """Class to allow custom rewards."""

    def __init__(
        self,
        collision: float = -1000.0,
        out_of_bounds: float = -3000.0,
        times_up: float = -1000.0,
        end_reached: float = 1000.0,
        close_to_gate: float = 0.0,
        dist_to_gate_mul: float = -1.0,
        dist_to_obstacle_mul: float = 0.2,
        gate_reached: float = 1000.0,
    ):
        """Initialize the rewarder."""
        self.collision = collision
        self.out_of_bounds = out_of_bounds
        self.end_reached = end_reached
        self.close_to_gate = close_to_gate
        self.dist_to_gate_mul = dist_to_gate_mul
        self.dist_to_obstacle_mul = dist_to_obstacle_mul
        self.gate_reached = gate_reached

    @classmethod
    def from_yaml(cls, file_path: str) -> Rewarder:  # noqa: ANN102
        """Load the rewarder from a YAML file.

        Args:
            file_path: The path to the YAML file.

        Returns:
            The rewarder.
        """
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
        return cls(**data)

    def get_custom_reward(self, obs: ObservationParser, info: dict) -> float:
        """Compute the custom reward.

        Args:
            reward: The reward from the firmware environment.
            obs: The current observation.
            terminated: True if the episode is terminated.
            truncated: True if the episode is truncated.
            info: The info dict from the firmware environment.

        Returns:
            The custom reward.
        """
        reward = 0.0

        if info["collision"][1]:
            return self.collision

        if obs.out_of_bounds():
            return self.out_of_bounds

        if info["task_completed"]:
            return self.end_reached

        # Reward for gating close to the next gate
        dist_to_gate = np.linalg.norm(obs.drone_pos - obs.gates_pos[obs.gate_id])
        reward += self.close_to_gate + dist_to_gate * self.dist_to_gate_mul

        # Reward for avoiding obstacles
        dist_to_obstacle = np.linalg.norm(obs.drone_pos - obs.obstacles_pos[obs.obstacles_in_range])
        reward += dist_to_obstacle * self.dist_to_obstacle_mul

        # Reward for passing a gate
        if obs.just_passed_gate:
            reward += self.gate_reached

        return reward


def map_reward_to_color(reward: float) -> str:
    """Convert the reward to a color.

    We use a red-green color map, where red indicates a negative reward and green a positive reward.

    Args:
        reward: The reward.

    Returns:
        The color.
    """
    if reward < 0:
        return "red"
    return "green"


def transform_action(
    raw_action: np.ndarray,
    observation_parser: ObservationParser,
    transform_type: str = "relative",
    pos_scaling: np.array = [1.0, 1.0, 1.0],
    yaw_scaling: float = np.pi,
) -> np.ndarray:
    """Transform the raw action to the action space.

    Args:
        raw_action: The raw action from the model is in the range [-1, 1].
        observation_parser: The observation parser.
        transform_type: The type of transformation, either "relative" or "absolute".
        pos_scaling: The scaling of the position.
        yaw_scaling: The scaling of the angle

    Returns:
        The transformed action to control the drone.
    """
    if transform_type == "relative":
        action_transform = np.zeros(14)
        action_transform[:3] = observation_parser.drone_pos + raw_action[:3]
        action_transform[9] = yaw_scaling * raw_action[3]

    elif transform_type == "absolute":
        action_transform = np.zeros(14)
        scaled_action = raw_action * np.concatenate([pos_scaling, [yaw_scaling]])
        action_transform[:3] = scaled_action[:3]
        action_transform[9] = scaled_action[3]

    return action_transform
