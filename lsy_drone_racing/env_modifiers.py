"""Some classes to modify the environment during training."""

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
        self.measured_drone_speed = None
        self.drone_rpy = None
        self.drone_angular_speed = None
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
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
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
        self.measured_drone_angular_speed = obs[9:12]

        # We update the speed estimator
        self.speed_estimator.update()
        self.drone_speed = self.speed_estimator.speed_estimate
        self.drone_relative_speed = self.drone_rot_matrix.T @ self.drone_speed
        self.drone_angular_speed = self.speed_estimator.angular_speed_estimate

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


class FullRelativeObservationParser(ObservationParser):
    """Class to parse the observation space of the firmware environment as provided by the Scaramuzza lab."""

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
    if type == "full_relative":
        return FullRelativeObservationParser(n_gates, n_obstacles, **data)
    if type == "action":
        return ActionObservationParser(n_gates, n_obstacles, **data)
    if type == "minimal":
        return MinimalObservationParser(n_gates, n_obstacles, **data)
    if type == "relative_position":
        return RelativePositionObservationParser(n_gates, n_obstacles, **data)
    if type == "scaramuzza":
        return ScaramuzzaObservationParser(n_gates, n_obstacles, **data)
    if type == "relative_corners":
        return RelativeCornersObservationParser(n_gates, n_obstacles, **data)
    if type == "classic":
        return ClassicObservationParser(n_gates, n_obstacles, **data)
    raise ValueError(f"Unknown observation parser type: {type}")


class Rewarder:
    """Class to allow custom rewards."""

    def __init__(
        self,
        collision: float = -1.0,
        out_of_bounds: float = -1.0,
        times_up: float = -1.0,
        dist_to_gate_mul: float = 1.0,
        end_reached: float = 10.0,
        gate_reached: float = 3.0,
        z_penalty: float = -0.1,
        z_penalty_threshold: float = 1.5,
        action_smoothness: float = 1e-4,
        body_rate_penalty: float = -1e-3,
        speed_threshold: float = np.inf,
        speed_penalty: float = -1e-3,
        angular_speed_threshold: float = np.inf,
        angular_speed_penalty: float = -1e-3,
        hovering_goal: list = None,
        shortname: str = "default",
    ):
        """Initialize the rewarder."""
        self.collision = collision
        self.out_of_bounds = out_of_bounds
        self.end_reached = end_reached
        self.gate_reached = gate_reached
        self.times_up = times_up
        self.dist_to_gate_mul = dist_to_gate_mul
        self.z_penalty = z_penalty
        self.z_penalty_threshold = z_penalty_threshold
        self.action_smoothness = action_smoothness
        self.body_rate_penalty = body_rate_penalty
        self.speed_threshold = speed_threshold
        self.speed_penalty = speed_penalty
        self.angular_speed_threshold = angular_speed_threshold
        self.angular_speed_penalty = angular_speed_penalty
        self.hovering_goal = np.array(hovering_goal) if hovering_goal is not None else None

        # Check that all are floats
        for attr in [
            "collision",
            "out_of_bounds",
            "end_reached",
            "gate_reached",
            "times_up",
            "dist_to_gate_mul",
            "z_penalty",
            "z_penalty_threshold",
            "action_smoothness",
            "body_rate_penalty",
            "speed_threshold",
            "speed_penalty",
            "angular_speed_threshold",
            "angular_speed_penalty",
        ]:
            if not isinstance(getattr(self, attr), float):
                raise ValueError(f"{attr} must be a float.")

        self.shortname = shortname

        logger.info(
            f"Rewarder: Collision: {self.collision}, Out of bounds: {self.out_of_bounds}, \
            End reached: {self.end_reached}, Gate reached: {self.gate_reached}, \
            Times up: {self.times_up}, Dist to gate mul: {self.dist_to_gate_mul}, \
            Z penalty: {self.z_penalty}, Z penalty threshold: {self.z_penalty_threshold}, \
            Action smoothness: {self.action_smoothness}, Body rate penalty: {self.body_rate_penalty}, \
            Speed threshold: {self.speed_threshold}, Speed penalty: {self.speed_penalty}, \
            Angular speed threshold: {self.angular_speed_threshold}, Angular speed penalty: {self.angular_speed_penalty}, \
            Hovering goal: {self.hovering_goal}, Shortname: {self.shortname}"
        )

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

    def get_shortname(self) -> str:
        """Return shortname to identify learned model after training."""
        return self.shortname

    def get_custom_reward(
        self, obs_parser: ObservationParser, info: dict, terminated: bool = False, action: np.ndarray = None
    ) -> float:
        """Compute the custom reward.

        Args:
            reward: The reward from the firmware environment.
            obs_parser: The current observation.
            terminated: True if the episode is terminated.
            truncated: True if the episode is truncated.
            info: The info dict from the firmware environment.
            action: The action used by the agent.

        Returns:
            The custom reward.
        """
        reward = 0.0

        if info["collision"][1]:
            return self.collision

        if info.get("TimeLimit.truncated", False):
            return self.times_up

        if obs_parser.out_of_bounds():
            return self.out_of_bounds

        if info["task_completed"]:
            logger.info("End reached. Hooray!")
            return self.end_reached

        if self.hovering_goal is not None:
            reward += np.exp(-np.linalg.norm(obs_parser.drone_pos - self.hovering_goal))
            return reward

        if obs_parser.gate_id == -1:
            # Reward for getting closer to the reference position
            dist_to_ref = np.linalg.norm(obs_parser.drone_pos - obs_parser.reference_position)
            previos_dist_to_ref = np.linalg.norm(obs_parser.previous_drone_pos - obs_parser.reference_position)
            reward += (previos_dist_to_ref - dist_to_ref) * self.dist_to_gate_mul
        else:
            dist_to_gate = np.linalg.norm(obs_parser.drone_pos - obs_parser.gates_pos[obs_parser.gate_id])
            previos_dist_to_gate = np.linalg.norm(
                obs_parser.previous_drone_pos - obs_parser.gates_pos[obs_parser.gate_id]
            )
            reward += (previos_dist_to_gate - dist_to_gate) * self.dist_to_gate_mul

        if obs_parser.drone_pos[2] > self.z_penalty_threshold:
            reward += self.z_penalty

        if np.linalg.norm(obs_parser.drone_angular_speed) > self.angular_speed_threshold:
            reward += self.angular_speed_penalty * np.linalg.norm(obs_parser.drone_angular_speed)

        if np.linalg.norm(obs_parser.drone_speed) > self.speed_threshold:
            reward += self.speed_penalty * np.linalg.norm(obs_parser.drone_speed)

        if action is not None:
            reward += -np.linalg.norm(action - obs_parser.previous_action) * self.action_smoothness

        if obs_parser.just_passed_gate:
            reward += self.gate_reached

        return reward


class ActionTransformer(ABC):
    """Class to transform the action space."""

    def __init__(self):
        """Initialize the action transformer."""
        pass

    @abstractmethod
    def transform(self, raw_action: np.ndarray, obs_parser: "ObservationParser") -> np.ndarray:
        """Transform the raw action to the action space."""
        raise NotImplementedError

    def create_firmware_action(self, action: np.ndarray, sim_time: float) -> np.ndarray:
        """Create the firmware action, from the transformed action.

        Args:
            action: The transformed action which has the form [x, y, z, yaw].
            sim_time: The simulation time.

        Returns:
            The firmware action. The firmware action is a 14-dimensional vector.
        """
        zeros3 = np.zeros(3)
        action = [action[:3], zeros3, zeros3, action[3], zeros3, sim_time]
        return action

    @classmethod
    def from_yaml(cls, file_path: str) -> ActionTransformer:  # noqa: ANN102
        """Load the action transformer from a YAML file.

        Args:
            file_path: The path to the YAML file.

        Returns:
            The action transformer.
        """
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
        return make_action_transformer(data)

    @abstractmethod
    def get_shortname(self) -> str:
        """Return shortname to identify learned model after training."""
        raise NotImplementedError

    def clip_mpi_to_pi(self, angle: float) -> float:
        """Clip the angle to the range [-pi, pi].

        Args:
            angle: The angle to clip.

        Returns:
            The clipped angle.
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi


class DoubleRelativeActionTransformer(ActionTransformer):
    """Class to transform the action space to relative actions."""

    def __init__(
        self,
        pos_scaling: np.array = 0.01 * np.ones(3),
        yaw_scaling: float = np.pi / 100.0,
        yaw_relative: bool = True,
        shortname: str = "2rel",
        **kwargs: Any,
    ):
        """Initialize the relative action transformer."""
        super().__init__()
        self.pos_scaling = pos_scaling
        self.yaw_scaling = yaw_scaling
        self.yaw_relative = yaw_relative
        logger.info(
            f"DoubleRelativeActionTransformer: Pos scaling: {self.pos_scaling}, Yaw scaling: {self.yaw_scaling}"
        )

        self.shortname = shortname

    def transform(self, raw_action: np.ndarray, obs_parser: "ObservationParser") -> np.ndarray:
        """Return a reative action based on the previous action."""
        action_transform = np.zeros(4)
        previous_action = obs_parser.previous_action
        action_transform[:3] = previous_action[:3] + raw_action[:3] * self.pos_scaling
        action_transform[3] = self.clip_mpi_to_pi(previous_action[3] + self.yaw_scaling * raw_action[3])
        return action_transform

    def get_shortname(self) -> str:
        """Return shortname to identify learned model after training."""
        return self.shortname


class RelativeActionTransformer(ActionTransformer):
    """Class to transform the action space to relative actions."""

    def __init__(
        self,
        pos_scaling: np.array = 0.5 * np.ones(3),
        yaw_scaling: float = np.pi,
        yaw_relative: bool = False,
        **kwargs: Any,
    ):
        """Initialize the relative action transformer."""
        super().__init__()
        self.pos_scaling = pos_scaling
        self.yaw_scaling = yaw_scaling
        self.yaw_relative = yaw_relative

    def transform(self, raw_action: np.ndarray, obs_parser: "ObservationParser") -> np.ndarray:
        """Transform the raw action to the action space.

        Args:
            raw_action: The raw action from the model is in the range [-1, 1].
            obs_parser: observation parser to get some needed information.

        Returns:
            The transformed action to control the drone as a 4-dimensional vector.
        """
        drone_pos = obs_parser.drone_pos
        drone_yaw = obs_parser.drone_yaw
        action_transform = np.zeros(4)
        action_transform[:3] = drone_pos + raw_action[:3] * self.pos_scaling
        if self.yaw_relative:
            action_transform[3] = self.clip_mpi_to_pi(self.yaw_scaling * raw_action[3] + drone_yaw)
        else:
            action_transform[3] = self.yaw_scaling * raw_action[3]
        return action_transform

    def get_shortname(self) -> str:
        """Return shortname to identify learned model after training."""
        return "rel"


class AbsoluteActionTransformer(ActionTransformer):
    """Class to transform the action space to absolute actions."""

    def __init__(self, pos_scaling: np.array = 5.0 * np.ones(3), yaw_scaling: float = np.pi, **kwargs: Any):
        """Initialize the absolute action transformer."""
        super().__init__()
        self.pos_scaling = pos_scaling
        self.yaw_scaling = yaw_scaling

    def transform(self, raw_action: np.ndarray, obs_parser: "ObservationParser") -> np.ndarray:
        """Transform the raw action to the action space.

        Args:
            raw_action: The raw action from the model is in the range [-1, 1].
            obs_parser: not actually needed for  this action transformer.

        Returns:
            The transformed action to control the drone.
        """
        action_transform = np.zeros(4)
        scaled_action = raw_action * np.concatenate([self.pos_scaling, [self.yaw_scaling]])
        action_transform[:3] = scaled_action[:3]
        action_transform[3] = scaled_action[3]
        return action_transform

    def get_shortname(self) -> str:
        """Return shortname to identify learned model after training."""
        return "abs"


def make_action_transformer(
    data: dict,
) -> ActionTransformer:
    """Create an action transformer.

    Args:
        data: The data to create the action transformer.

    Returns:
        The action transformer.
    """
    action_transformer_type = data["type"]
    if action_transformer_type == "relative":
        return RelativeActionTransformer(**data)
    if action_transformer_type == "absolute":
        return AbsoluteActionTransformer(**data)
    if action_transformer_type == "double_relative":
        return DoubleRelativeActionTransformer(**data)
    raise ValueError(f"Unknown action transformer type: {action_transformer_type}")
