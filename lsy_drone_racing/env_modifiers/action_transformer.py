"""Module to transform the action space."""

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import yaml
from gymnasium.spaces import Box

from lsy_drone_racing.env_modifiers.observation_parser import ObservationParser

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ActionTransformer(ABC):
    """Class to transform the action space."""

    def __init__(self):
        """Initialize the action transformer."""
        pass

    @abstractmethod
    def transform(self, raw_action: np.ndarray, obs_parser: ObservationParser) -> np.ndarray:
        """Transform the raw action to the action space."""
        raise NotImplementedError

    def create_firmware_action(
        self,
        action: np.ndarray,
        sim_time: float,
    ) -> np.ndarray:
        """Create the firmware action, from the transformed action.

        Args:
            action: The transformed action which has the form [x, y, z, yaw].
            sim_time: The simulation time.

        Returns:
            The firmware action. The firmware action is a 14-dimensional vector.
        """
        zeros3 = np.zeros(3)
        action = [action[:3], zeros3, zeros3, action[3] if len(action) == 4 else 0.0, zeros3, sim_time]
        return action

    def get_action_space(self) -> Box:
        """Return the action space."""
        return Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

    @classmethod
    def from_yaml(cls, file_path: str) -> "ActionTransformer":  # noqa: ANN102
        """Load the action transformer from a YAML file.

        Args:
            file_path: The path to the YAML file.

        Returns:
            The action transformer.
        """
        if file_path is None or file_path == "":
            logger.error("No action transformer file path provided. Returning default action transformer.")
            data = {"type": "relative"}
            return make_action_transformer()
        try:
            with open(file_path, "r") as file:
                data = yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading action transformer from {file_path}: {e}")
            logger.error("Returning default action transformer.")
            data = {"type": "relative"}

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


class NoYawActionTransformer(ActionTransformer):
    """Class to transform the action space to relative actions."""

    def __init__(
        self,
        pos_scaling: np.array = 0.025 * np.ones(3),
        shortname: str = "2rel",
        **kwargs: Any,
    ):
        """Initialize the relative action transformer."""
        super().__init__()
        self.pos_scaling = pos_scaling
        logger.info(f"NoYawActionTransformer: Pos scaling: {self.pos_scaling}")

        self.shortname = shortname

    def transform(self, raw_action: np.ndarray, obs_parser: "ObservationParser") -> np.ndarray:
        """Return a reative action based on the previous action."""
        action_transform = np.zeros(3)
        previous_action = obs_parser.previous_action
        action_transform = previous_action[:3] + raw_action[:3] * self.pos_scaling
        return action_transform

    def get_shortname(self) -> str:
        """Return shortname to identify learned model after training."""
        return self.shortname


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
        shortname: str = "rel",
        **kwargs: Any,
    ):
        """Initialize the relative action transformer."""
        super().__init__()
        self.pos_scaling = pos_scaling
        self.yaw_scaling = yaw_scaling
        self.yaw_relative = yaw_relative

        self.shortname = shortname

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
        return self.shortname


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
    if action_transformer_type == "no_yaw":
        return NoYawActionTransformer(**data)
    raise ValueError(f"Unknown action transformer type: {action_transformer_type}")
