"""Observation parser classes used to parse the observations of the default environment."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import yaml
from gymnasium.spaces import Box
from transforms3d.euler import euler2mat

from lsy_drone_racing.speed_estimation import make_speed_estimator

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ObservationParser(ABC):
    """Class to parse the observation space of the firmware environment."""

    def __init__(
        self,
        n_gates: int,
        n_obstacles: int,
        **kwargs: Any,
    ):
        """Initialize the observation parser."""
        self.n_gates = n_gates
        self.n_obstacles = n_obstacles

        # Observable variables
        self.drone_pos = None
        self.drone_speed = None
        self.drone_relative_speed = None
        self.measured_drone_speed = None
        self.drone_rpy = None
        self.drone_rot_matrix = None
        self.drone_gyro = None
        self.drone_angular_speed = None
        self.drone_relative_angular_speed = None
        self.measured_drone_angular_speed = None
        self.gates_pos = None
        self.gates_yaw = None
        self.gates_in_range = None
        self.obstacles_pos = None
        self.obstacles_in_range = None
        self.gate_id = None

        # Hidden states that are not part of the observation space
        self.ctrl_freq: float = None
        self.just_passed_gate: bool = False
        self.gate_edge_size: float = None
        self.previous_action: np.array = np.zeros(4)
        self.previous_drone_pos: np.array = None
        self.reference_position: np.array = None
        self.drone_pos_limits: list = [3.0, 3.0, 2.0]
        self.drone_pos_space = Box(
            low=-1.0 * np.array(self.drone_pos_limits), high=np.array(self.drone_pos_limits), dtype=np.float32
        )

        # Speed estimator
        speed_estimator_dict = kwargs.get("speed_estimator", {"type": "default"})
        self.speed_estimator = make_speed_estimator(self, speed_estimator_dict)

    @classmethod
    def from_yaml(cls, n_gates: int, n_obstacles: int, file_path: str) -> ObservationParser:  # noqa: ANN102
        """Load the observation parser from a YAML file."""
        if file_path is None or file_path == "":
            logger.info("No observation parser file path provided. Returning default observation parser.")
            return make_observation_parser(n_gates, n_obstacles, {"type": "minimal"})

        try:
            with open(file_path, "r") as file:
                data = yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading observation parser from {file_path}: {e}")
            logger.error("Returning default observation parser.")
            data = {"type": "minimal"}
        return make_observation_parser(n_gates, n_obstacles, data)

    def uninitialized(self) -> bool:
        """Check if the observation parser is uninitialized."""
        return self.drone_pos is None

    @property
    def drone_on_ground(self) -> bool:
        """Check if the drone is on the ground."""
        return self.drone_pos[2] < 0.1

    def out_of_bounds(self) -> bool:
        """Check if the drone is out of bounds."""
        not_in_observation_space = not self.observation_space.contains(self.get_observation())
        not_in_position_space = not self.drone_pos_space.contains(self.drone_pos.astype(np.float32))
        return not_in_observation_space or not_in_position_space

    def update(
        self,
        obs: np.ndarray,
        info: dict[str, Any],
        initial: bool = False,
        action: np.ndarray = None,
    ):
        """Update the observation parser with the new observation and info dict.

        Remark:
            We do not update the gate height here, the info dict does not contain this information.

        Args:
            obs: The new observation.
            info: The new info dict.
            initial: True if this is the initial observation.
            action: The previous action.
        """
        self.previous_drone_pos = self.drone_pos if not initial else obs[0:6:2]
        self.drone_pos = obs[0:6:2]
        self.measured_drone_speed = obs[1:6:2]
        self.drone_rpy = obs[6:9]
        self.drone_rot_matrix = euler2mat(*self.drone_rpy)
        self.drone_gyro = self.drone_rot_matrix.T @ np.array([0.0,0.0, 1.0])

        self.measured_drone_angular_speed = obs[9:12]

        # We update the speed estimator
        self.speed_estimator.update()
        self.drone_speed = self.speed_estimator.speed_estimate
        self.drone_relative_speed = self.drone_rot_matrix.T @ self.drone_speed
        self.drone_angular_speed = self.speed_estimator.angular_speed_estimate
        self.drone_relative_angular_speed = self.drone_rot_matrix.T @ self.drone_angular_speed

        if initial:
            self.gate_edge_size = info["gate_dimensions"]["tall"]["edge"]
            self.reference_position = info["x_reference"][0:6:2]
            self.previous_action = np.concatenate([self.drone_pos, [0]])
            self.ctrl_freq = info["ctrl_freq"]
            self.dt = 1 / self.ctrl_freq
            self.speed_estimator.reset()
        if action is not None:
            self.previous_action = action
        # norm_sped = np.linalg.norm(self.drone_speed)
        # angular_speed = np.linalg.norm(self.drone_angular_speed)
        # logger.info(f"Speed: {norm_sped}, Angular speed: {angular_speed}")

        self.gates_pos = info["gates_pose"][:, :3]
        self.gates_yaw = info["gates_pose"][:, 5]
        self.gates_in_range = info["gates_in_range"]
        self.obstacles_pos = info["obstacles_pose"][:, :3]
        self.obstacles_in_range = info["obstacles_in_range"]
        self.just_passed_gate = self.gate_id != info["current_gate_id"]
        self.gate_id = info["current_gate_id"]

    @property
    def drone_yaw(self) -> float:
        """Return the yaw of the drone."""
        return self.drone_rpy[2]

    @abstractmethod
    def get_shortname(self) -> str:
        """Return a shortname to identify learned model after training."""
        raise NotImplementedError

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        """Return the current observation.

        Returns:
            The current observation.
        """
        raise NotImplementedError

    def get_relative_corners(self, include_reference_position: bool = False) -> np.ndarray:
        """Return the relative position of the corners of the gates with respect to the drone."""
        gates_pos = (
            np.vstack([self.gates_pos, self.reference_position]) if include_reference_position else self.gates_pos
        )
        gates_yaw = np.hstack([self.gates_yaw, 0]) if include_reference_position else self.gates_yaw

        edge_cos = np.cos(gates_yaw)
        edge_sin = np.sin(gates_yaw)
        ones = np.ones_like(edge_cos)
        edge_vector_pos = self.gate_edge_size / 2 * np.array([edge_cos, edge_sin, ones]).T
        edge_vector_neg = self.gate_edge_size / 2 * np.array([-edge_cos, -edge_sin, ones]).T

        first_corners = gates_pos + edge_vector_pos - self.drone_pos
        second_corners = gates_pos + edge_vector_neg - self.drone_pos
        third_corners = gates_pos - edge_vector_pos - self.drone_pos
        fourth_corners = gates_pos - edge_vector_neg - self.drone_pos

        relative_distance_corners = np.array([first_corners, second_corners, third_corners, fourth_corners])
        return relative_distance_corners

    def get_relative_obstacles(self) -> np.ndarray:
        """Return the relative position of the obstacles."""
        return self.obstacles_pos - self.drone_pos

    def get_relative_gates(self) -> np.ndarray:
        """Return the relative position of the gates."""
        return self.gates_pos - self.drone_pos

    def __repr__(self) -> str:
        """Return the string representation of the observation parser."""
        return f"{self.__class__.__name__}(n_gates={self.n_gates}, n_obstacles={self.n_obstacles})"


class NormalizedObservationParser(ObservationParser):
    """Class to normalize the observation space of a provided observation parser."""

    def __init__(self, observation_parser: ObservationParser):
        """Initialize the normalized observation parser."""
        self.observation_parser = observation_parser
        self.low = self.observation_parser.observation_space.low
        self.high = self.observation_parser.observation_space.high
        self.observation_space = Box(-1.0, 1.0, shape=observation_parser.observation_space.shape)
        logger.info(f"NormalizedObservationParser: Observation space: {self.observation_space}")

    def update(
        self,
        obs: np.ndarray,
        info: dict[str, Any],
        initial: bool = False,
        action: np.ndarray = None,
    ):
        """Update the observation parser with the new observation and info dict.

        Args:
            obs: The new observation.
            info: The new info dict.
            initial: True if this is the initial observation.
            action: The previous action.
        """
        self.observation_parser.update(obs, info, initial, action)

    def __getattr__(self, name: str) -> Any:
        """Return the attribute from the observation parser."""
        return getattr(self.observation_parser, name)

    def get_observation(self) -> np.ndarray:
        """Return the current observation, normalized."""
        obs = self.observation_parser.get_observation()
        normalized_obs = 2 * (obs - self.low) / (self.high - self.low) - 1
        return normalized_obs

    def get_shortname(self) -> str:
        """Return shortname to identify learned model after training."""
        return f"{self.observation_parser.get_shortname()}Norm"
        


class FullRelativeObservationParser(ObservationParser):
    """Class to parse the observation space of the firmware environment as provided by the Scaramuzza lab."""

    def __init__(
        self,
        n_gates: int,
        n_obstacles: int,
        n_gates_in_sight: int = 2,
        action_limits: list = [5] * 3 + [np.pi],
        drone_speed_limits: list = [5] * 3,
        drone_rot_matrix_limits: list = [1] * 9,
        drone_angular_speed_limits: list = [10] * 3,
        gate_pos_limits: list = [10] * 3,
        obstacle_pos_limits: list = [10] * 3,
        speed_noise: float = 0.0,
        **kwargs: Any,
    ):
        """Initialize the Scaramuzza observation parser."""
        super().__init__(n_gates, n_obstacles, **kwargs)
        n_corners = 4
        self.n_gates_in_sight = n_gates_in_sight
        self.speed_noise = speed_noise
        relative_corners_limits = gate_pos_limits * n_gates_in_sight * n_corners
        obs_limits = (
            action_limits
            + drone_speed_limits
            + drone_rot_matrix_limits
            + drone_angular_speed_limits
            + relative_corners_limits
            + obstacle_pos_limits * n_obstacles
            + [n_gates]
        )
        obs_limits_high = np.array(obs_limits)
        obs_limits_low = np.concatenate([-obs_limits_high[:-1], [-1]])
        self.observation_space = Box(obs_limits_low, obs_limits_high, dtype=np.float32)
        logger.info(
            f"FullRelativeObservationParser: Action limits: {action_limits}, Drone speed limits: {drone_speed_limits}, \
            Drone rot matrix limits: {drone_rot_matrix_limits}, Drone angular speed limits: {drone_angular_speed_limits}, \
            Gate pos limits: {gate_pos_limits}, Obstacle pos limits: {obstacle_pos_limits}, Speed noise: {speed_noise}"
        )

    def get_shortname(self) -> str:
        """Return shortname to identify learned model after training."""
        return "fullRel"

    def get_observation(self) -> np.ndarray:
        """Return the current observation."""
        relative_corners = self.get_relative_corners(include_reference_position=True)
        if self.gate_id == -1:
            gates_ids_in_sight = [-1] * self.n_gates_in_sight
        else:
            gates_ids_in_sight = range(self.gate_id, self.gate_id + self.n_gates_in_sight)
            gates_ids_in_sight = [i if i < self.n_gates else -1 for i in gates_ids_in_sight]

        relative_corners_in_sight = [relative_corners[:, i, :] for i in gates_ids_in_sight]

        relative_obstacles = self.get_relative_obstacles()
        obs = np.concatenate(
            [
                self.previous_action,
                self.drone_relative_speed + np.random.normal(0, self.speed_noise, 3),
                self.drone_rot_matrix.ravel(),
                self.drone_angular_speed + np.random.normal(0, self.speed_noise, 3),
                np.array(relative_corners_in_sight).ravel(),
                relative_obstacles.flatten(),
                [self.gate_id],
            ]
        )
        # logger.debug(f"Gates id in sight: {gates_ids_in_sight}")
        return obs.astype(np.float32)

class GyroObservationParser(ObservationParser):
    """TEST."""

    def __init__(
        self,
        n_gates: int,
        n_obstacles: int,
        n_gates_in_sight: int = 2,
        action_limits: list = [5] * 3 + [np.pi],
        drone_speed_limits: list = [5] * 3,
        drone_gyro_limits: list = [1] * 3,
        drone_angular_speed_limits: list = [10] * 3,
        gate_pos_limits: list = [10] * 3,
        obstacle_pos_limits: list = [10] * 3,
        speed_noise: float = 0.0,
        **kwargs: Any,
    ):
        """Initialize the Scaramuzza observation parser."""
        super().__init__(n_gates, n_obstacles, **kwargs)
        n_corners = 4
        self.n_gates_in_sight = n_gates_in_sight
        self.speed_noise = speed_noise
        relative_corners_limits = gate_pos_limits * n_gates_in_sight * n_corners
        obs_limits = (
            action_limits
            + drone_speed_limits
            + drone_gyro_limits
            + drone_angular_speed_limits
            + relative_corners_limits
            + obstacle_pos_limits * n_obstacles
            + [n_gates]
        )
        obs_limits_high = np.array(obs_limits)
        obs_limits_low = np.concatenate([-obs_limits_high[:-1], [-1]])
        self.observation_space = Box(obs_limits_low, obs_limits_high, dtype=np.float32)
        logger.info(
            f"ActionObservationParser: Action limits: {action_limits}, Drone speed limits: {drone_speed_limits}, \
            Drone gravity limits {drone_gyro_limits}, Drone angular speed limits: {drone_angular_speed_limits}, \
            Gate pos limits: {gate_pos_limits}, Obstacle pos limits: {obstacle_pos_limits}, Speed noise: {speed_noise}"
        )
        self.shortname = kwargs.get("shortname", "gyro")

    def get_shortname(self) -> str:
        """Return shortname to identify learned model after training."""
        return self.shortname

    def get_observation(self) -> np.ndarray:
        """Return the current observation."""
        relative_corners = self.get_relative_corners(include_reference_position=True)
        if self.gate_id == -1:
            gates_ids_in_sight = [-1] * self.n_gates_in_sight
        else:
            gates_ids_in_sight = range(self.gate_id, self.gate_id + self.n_gates_in_sight)
            gates_ids_in_sight = [i if i < self.n_gates else -1 for i in gates_ids_in_sight]

        relative_corners_in_sight = [relative_corners[:, i, :] for i in gates_ids_in_sight]

        relative_obstacles = self.get_relative_obstacles()
        obs = np.concatenate(
            [
                self.previous_action[0:3] - self.drone_pos,
                [self.previous_action[3] - self.drone_yaw],
                self.drone_relative_speed+ np.random.normal(0, self.speed_noise, 3),
                self.drone_gyro,
                self.drone_relative_angular_speed + np.random.normal(0, self.speed_noise, 3),
                np.array(relative_corners_in_sight).ravel(),
                relative_obstacles.flatten(),
                [self.gate_id],
            ]
        )
        # logger.debug(f"Gates id in sight: {gates_ids_in_sight}")
        return obs.astype(np.float32)

class Action2ObservationParser(ObservationParser):
    """TEST."""

    def __init__(
        self,
        n_gates: int,
        n_obstacles: int,
        n_gates_in_sight: int = 2,
        action_limits: list = [5] * 3 + [np.pi],
        drone_speed_limits: list = [10] * 3,
        drone_rot_matrix_limits: list = [1] * 9,
        drone_angular_speed_limits: list = [10] * 3,
        gate_pos_limits: list = [10] * 3,
        obstacle_pos_limits: list = [10] * 3,
        speed_noise: float = 0.0,
        **kwargs: Any,
    ):
        """Initialize the Scaramuzza observation parser."""
        super().__init__(n_gates, n_obstacles, **kwargs)
        n_corners = 4
        self.n_gates_in_sight = n_gates_in_sight
        self.speed_noise = speed_noise
        relative_corners_limits = gate_pos_limits * n_gates_in_sight * n_corners
        obs_limits = (
            action_limits
            + drone_speed_limits
            + drone_rot_matrix_limits
            + drone_angular_speed_limits
            + relative_corners_limits
            + obstacle_pos_limits * n_obstacles
            + [n_gates]
        )
        obs_limits_high = np.array(obs_limits)
        obs_limits_low = np.concatenate([-obs_limits_high[:-1], [-1]])
        self.observation_space = Box(obs_limits_low, obs_limits_high, dtype=np.float32)
        logger.info(
            f"ActionObservationParser: Action limits: {action_limits}, Drone speed limits: {drone_speed_limits}, \
            Drone rot matrix limits {drone_rot_matrix_limits}, Drone angular speed limits: {drone_angular_speed_limits}, \
            Gate pos limits: {gate_pos_limits}, Obstacle pos limits: {obstacle_pos_limits}, Speed noise: {speed_noise}"
        )

    def get_shortname(self) -> str:
        """Return shortname to identify learned model after training."""
        return "act2"

    def get_observation(self) -> np.ndarray:
        """Return the current observation."""
        relative_corners = self.get_relative_corners(include_reference_position=True)
        if self.gate_id == -1:
            gates_ids_in_sight = [-1] * self.n_gates_in_sight
        else:
            gates_ids_in_sight = range(self.gate_id, self.gate_id + self.n_gates_in_sight)
            gates_ids_in_sight = [i if i < self.n_gates else -1 for i in gates_ids_in_sight]

        relative_corners_in_sight = [relative_corners[:, i, :] for i in gates_ids_in_sight]

        relative_obstacles = self.get_relative_obstacles()
        obs = np.concatenate(
            [
                self.previous_action[0:3] - self.drone_pos,
                [self.previous_action[3] - self.drone_yaw],
                self.drone_relative_speed+ np.random.normal(0, self.speed_noise, 3),
                self.drone_rot_matrix.ravel(),
                self.drone_relative_angular_speed + np.random.normal(0, self.speed_noise, 3),
                np.array(relative_corners_in_sight).ravel(),
                relative_obstacles.flatten(),
                [self.gate_id],
            ]
        )
        # logger.debug(f"Gates id in sight: {gates_ids_in_sight}")
        return obs.astype(np.float32)

class ActionObservationParser(ObservationParser):
    """Class to parse the observation space of the firmware environment as provided by the Scaramuzza lab."""

    def __init__(
        self,
        n_gates: int,
        n_obstacles: int,
        n_gates_in_sight: int = 2,
        action_limits: list = [5] * 3 + [np.pi],
        drone_speed_limits: list = [10] * 3,
        drone_rpy_limits: list = [np.pi] * 3,
        drone_angular_speed_limits: list = [10] * 3,
        gate_pos_limits: list = [10] * 3,
        obstacle_pos_limits: list = [10] * 3,
        speed_noise: float = 0.0,
        **kwargs: Any,
    ):
        """Initialize the Scaramuzza observation parser."""
        super().__init__(n_gates, n_obstacles, **kwargs)
        n_corners = 4
        self.n_gates_in_sight = n_gates_in_sight
        self.speed_noise = speed_noise
        relative_corners_limits = gate_pos_limits * n_gates_in_sight * n_corners
        obs_limits = (
            action_limits
            + drone_speed_limits
            + drone_rpy_limits
            + drone_angular_speed_limits
            + relative_corners_limits
            + obstacle_pos_limits * n_obstacles
            + [n_gates]
        )
        obs_limits_high = np.array(obs_limits)
        obs_limits_low = np.concatenate([-obs_limits_high[:-1], [-1]])
        self.observation_space = Box(obs_limits_low, obs_limits_high, dtype=np.float32)
        logger.info(
            f"ActionObservationParser: Action limits: {action_limits}, Drone speed limits: {drone_speed_limits}, \
            Drone rpy limits: {drone_rpy_limits}, Drone angular speed limits: {drone_angular_speed_limits}, \
            Gate pos limits: {gate_pos_limits}, Obstacle pos limits: {obstacle_pos_limits}, Speed noise: {speed_noise}"
        )

    def get_shortname(self) -> str:
        """Return shortname to identify learned model after training."""
        return "act"

    def get_observation(self) -> np.ndarray:
        """Return the current observation."""
        relative_corners = self.get_relative_corners(include_reference_position=True)
        if self.gate_id == -1:
            gates_ids_in_sight = [-1] * self.n_gates_in_sight
        else:
            gates_ids_in_sight = range(self.gate_id, self.gate_id + self.n_gates_in_sight)
            gates_ids_in_sight = [i if i < self.n_gates else -1 for i in gates_ids_in_sight]

        relative_corners_in_sight = [relative_corners[:, i, :] for i in gates_ids_in_sight]

        relative_obstacles = self.get_relative_obstacles()
        obs = np.concatenate(
            [
                self.previous_action,
                self.drone_speed + np.random.normal(0, self.speed_noise, 3),
                self.drone_rpy,
                self.drone_angular_speed + np.random.normal(0, self.speed_noise, 3),
                np.array(relative_corners_in_sight).ravel(),
                relative_obstacles.flatten(),
                [self.gate_id],
            ]
        )
        # logger.debug(f"Gates id in sight: {gates_ids_in_sight}")
        return obs.astype(np.float32)


class MinimalObservationParser(ObservationParser):
    """Class to parse the observation space of the firmware environment, to only include the drone information."""

    def __init__(
        self,
        n_gates: int,
        n_obstacles: int,
        drone_pos_limits: list = [3, 3, 2],
        drone_speed_limits: list = [10] * 3,
        drone_rpy_limits: list = [np.pi] * 3,
        drone_angular_speed_limits: list = [10] * 3,
        **kwargs: Any,
    ):
        """Initialize the Scaramuzza observation parser."""
        super().__init__(n_gates, n_obstacles)
        obs_limits = drone_pos_limits + drone_speed_limits + drone_rpy_limits + drone_angular_speed_limits + [n_gates]
        obs_limits_high = np.array(obs_limits)
        obs_limits_low = np.concatenate([-obs_limits_high[:-1], [-1]])
        self.observation_space = Box(obs_limits_low, obs_limits_high, dtype=np.float32)

    def get_shortname(self) -> str:
        """Return shortname to identify learned model after training."""
        return "min"

    def get_observation(self) -> np.ndarray:
        """Return the current observation."""
        obs = np.concatenate(
            [
                self.drone_pos,
                self.drone_speed,
                self.drone_rpy,
                self.drone_angular_speed,
                [self.gate_id],
            ]
        )
        return obs.astype(np.float32)


class RelativePositionObservationParser(ObservationParser):
    """Class to parse the observation space of the firmware environment as provided by the Scaramuzza lab."""

    def __init__(
        self,
        n_gates: int,
        n_obstacles: int,
        n_gates_in_sight: int = 2,
        drone_speed_limits: list = [10] * 3,
        drone_rpy_limits: list = [np.pi] * 3,
        drone_angular_speed_limits: list = [10] * 3,
        gate_pos_limits: list = [5] * 3,
        gate_yaw_limits: list = [np.pi],
        obstacle_pos_limits: list = [5] * 3,
        **kwargs: Any,
    ):
        """Initialize the Scaramuzza observation parser."""
        super().__init__(n_gates, n_obstacles, **kwargs)
        self.n_gates_in_sight = n_gates_in_sight
        obs_limits = (
            drone_speed_limits
            + drone_angular_speed_limits
            + drone_rpy_limits
            + gate_pos_limits * n_gates_in_sight
            + gate_yaw_limits * n_gates_in_sight
            + obstacle_pos_limits * n_obstacles
            + [n_gates]
        )
        obs_limits_high = np.array(obs_limits)
        obs_limits_low = np.concatenate([-obs_limits_high[:-1], [-1]])
        self.observation_space = Box(obs_limits_low, obs_limits_high, dtype=np.float32)

    def get_shortname(self) -> str:
        """Return shortname to identify learned model after training."""
        return "rel"

    def get_observation(self) -> np.ndarray:
        """Return the current observation."""
        gates_relative_pos = self.get_relative_gates()

        if self.gate_id == -1:
            gates_in_sight = [self.reference_position - self.drone_pos] * self.n_gates_in_sight
            gates_yaw_in_sight = [0] * self.n_gates_in_sight
        else:
            gates_ids_in_sight = range(self.gate_id, self.gate_id + self.n_gates_in_sight)
            gates_ids_in_sight = [i if i < self.n_gates else -1 for i in gates_ids_in_sight]
            gates_in_sight = [
                gates_relative_pos[i] if i != -1 else self.reference_position - self.drone_pos
                for i in gates_ids_in_sight
            ]
            gates_yaw_in_sight = [self.gates_yaw[i] if i != -1 else 0 for i in gates_ids_in_sight]
        relative_obstacles = self.get_relative_obstacles()
        obs = np.concatenate(
            [
                self.drone_speed,
                self.drone_angular_speed,
                self.drone_rpy,
                np.array(gates_in_sight).ravel(),
                np.array(gates_yaw_in_sight).ravel(),
                relative_obstacles.flatten(),
                [self.gate_id],
            ]
        )
        return obs.astype(np.float32)


class ScaramuzzaObservationParser(ObservationParser):
    """Class to parse the observation space of the firmware environment as provided by the Scaramuzza lab."""

    def __init__(
        self,
        n_gates: int,
        n_obstacles: int,
        n_gates_in_sight: int = 2,
        drone_speed_limits: list = [10] * 3,
        drone_rpy_limits: list = [np.pi] * 3,
        gate_pos_limits: list = [10] * 3,
        obstacle_pos_limits: list = [10] * 3,
        **kwargs: Any,
    ):
        """Initialize the Scaramuzza observation parser."""
        super().__init__(n_gates, n_obstacles)
        n_corners = 4
        self.n_gates_in_sight = n_gates_in_sight
        relative_corners_limits = gate_pos_limits * n_gates_in_sight * n_corners
        obs_limits = (
            drone_speed_limits
            + drone_rpy_limits
            + relative_corners_limits
            + obstacle_pos_limits * n_obstacles
            + [n_gates]
        )
        obs_limits_high = np.array(obs_limits)
        obs_limits_low = np.concatenate([-obs_limits_high[:-1], [-1]])
        self.observation_space = Box(obs_limits_low, obs_limits_high, dtype=np.float32)

    def __repr__(self) -> str:
        """Return the string representation of the observation parser."""
        obs = self.get_observation()
        return f"{self.__class__.__name__} Obs space: {self.observation_space}, \
                observation size: {obs.size}, \
                drone_speed: {obs[0:3]}, drone_rpy: {obs[3:6]}, \
                relative corners and obstacles {obs[6:-1]}, gate_id: {obs[-1]}"

    def get_shortname(self) -> str:
        """Return shortname to identify learned model after training."""
        return "sca"

    def get_observation(self) -> np.ndarray:
        """Return the current observation."""
        relative_corners = self.get_relative_corners(include_reference_position=True)
        if self.gate_id == -1:
            relative_corners_in_sight = [relative_corners[:, -1, :]] * self.n_gates_in_sight
        else:
            gates_ids_in_sight = range(self.gate_id, self.gate_id + self.n_gates_in_sight)
            gates_ids_in_sight = [i if i < self.n_gates else -1 for i in gates_ids_in_sight]
            relative_corners_in_sight = [relative_corners[:, i, :] for i in gates_ids_in_sight]

        relative_obstacles = self.get_relative_obstacles()
        obs = np.concatenate(
            [
                self.drone_speed,
                self.drone_rpy,
                np.array(relative_corners_in_sight).ravel(),
                relative_obstacles.flatten(),
                [self.gate_id],
            ]
        )
        return obs.astype(np.float32)


class RelativeCornersObservationParser(ObservationParser):
    """Class to parse the observation space of the firmware environment to provide relative corners of the gates."""

    def __init__(
        self,
        n_gates: int,
        n_obstacles: int,
        n_gates_in_sight: int = 2,
        drone_speed_limits: list = [10] * 3,
        drone_rpy_limits: list = [np.pi] * 3,
        drone_angular_speed_limits: list = [10] * 3,
        gate_pos_limits: list = [10] * 3,
        obstacle_pos_limits: list = [10] * 3,
        speed_noise: float = 0.0,
        **kwargs: Any,
    ):
        """Initialize the Scaramuzza observation parser."""
        super().__init__(n_gates, n_obstacles, **kwargs)
        n_corners = 4
        self.n_gates_in_sight = n_gates_in_sight
        self.speed_noise = speed_noise
        relative_corners_limits = gate_pos_limits * n_gates_in_sight * n_corners
        obs_limits = (
            drone_speed_limits
            + drone_rpy_limits
            + drone_angular_speed_limits
            + relative_corners_limits
            + obstacle_pos_limits * n_obstacles
            + [n_gates]
        )
        obs_limits_high = np.array(obs_limits)
        obs_limits_low = np.concatenate([-obs_limits_high[:-1], [-1]])
        self.observation_space = Box(obs_limits_low, obs_limits_high, dtype=np.float32)

    def __repr__(self) -> str:
        """Return the string representation of the observation parser."""
        obs = self.get_observation()
        return f"{self.__class__.__name__} Obs space: {self.observation_space}, \
                observation size: {obs.size}, \
                drone_speed: {obs[0:3]}, drone_rpy: {obs[3:6]}, \
                relative corners and obstacles {obs[6:-1]}, gate_id: {obs[-1]}"

    def get_shortname(self) -> str:
        """Return shortname to identify learned model after training."""
        return "rel_corners"

    def get_observation(self) -> np.ndarray:
        """Return the current observation."""
        relative_corners = self.get_relative_corners(include_reference_position=True)
        if self.gate_id == -1:
            relative_corners_in_sight = [relative_corners[:, -1, :]] * self.n_gates_in_sight
        else:
            gates_ids_in_sight = range(self.gate_id, self.gate_id + self.n_gates_in_sight)
            gates_ids_in_sight = [i if i < self.n_gates else -1 for i in gates_ids_in_sight]
            relative_corners_in_sight = [relative_corners[:, i, :] for i in gates_ids_in_sight]

        relative_obstacles = self.get_relative_obstacles()
        obs = np.concatenate(
            [
                self.drone_speed + np.random.normal(0, self.speed_noise, 3),
                self.drone_rpy,
                self.drone_angular_speed + np.random.normal(0, self.speed_noise, 3),
                np.array(relative_corners_in_sight).ravel(),
                relative_obstacles.flatten(),
                [self.gate_id],
            ]
        )
        return obs.astype(np.float32)


class ClassicObservationParser(ObservationParser):
    """Class to parse the observation space of the firmware environment as provided at the beginning of the competition."""

    def __init__(
        self,
        n_gates: int,
        n_obstacles: int,
        drone_pos_limits: list = [3, 3, 2],
        drone_speed_limits: list = [2] * 3,
        drone_yaw_limits: list = [np.pi],
        drone_yaw_speed_limits: list = [np.pi],
        gate_pos_limits: list = [5, 5, 5],
        gate_yaw_limits: list = [np.pi],
        gate_in_range_limits: list = [1],
        obstacle_pos_limits: list = [5, 5, 5],
        obstacle_in_range_limits: list = [1],
        **kwargs: Any,
    ):
        """Initialize the classic observation parser."""
        super().__init__(n_gates, n_obstacles)

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

    def get_shortname(self) -> str:
        """Return shortname to identify learned model after training."""
        return "classic"

    def get_observation(self) -> np.ndarray:
        """Return the current observation."""
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


def make_observation_parser(
    n_gates: int,
    n_obstacles: int,
    data: dict,
) -> ObservationParser:
    """Create an observation parser.

    Args:
        observation_parser_type: The type of the observation parser.
        n_gates: The number of gates.
        n_obstacles: The number of obstacles.
        data: The data to create the observation parser.

    Returns:
        The observation parser.
    """
    type = data["type"]
    normalized = data.get("normalized", False)
    if type == "full_relative":
        obs_parser = FullRelativeObservationParser(n_gates, n_obstacles, **data)
    elif type == "action":
        obs_parser = ActionObservationParser(n_gates, n_obstacles, **data)
    elif type == "action2":
        obs_parser = Action2ObservationParser(n_gates, n_obstacles, **data)
    elif type == "gyro":
        obs_parser = GyroObservationParser(n_gates, n_obstacles, **data)
    elif type == "minimal":
        obs_parser = MinimalObservationParser(n_gates, n_obstacles, **data)
    elif type == "relative_position":
        obs_parser = RelativePositionObservationParser(n_gates, n_obstacles, **data)
    elif type == "scaramuzza":
        obs_parser = ScaramuzzaObservationParser(n_gates, n_obstacles, **data)
    elif type == "relative_corners":
        obs_parser = RelativeCornersObservationParser(n_gates, n_obstacles, **data)
    elif type == "classic":
        obs_parser = ClassicObservationParser(n_gates, n_obstacles, **data)
    else:
        raise ValueError(f"Unknown observation parser type: {type}")
    return NormalizedObservationParser(obs_parser) if normalized else obs_parser
