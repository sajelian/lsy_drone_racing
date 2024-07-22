"""Helper functions and classes for state estimation.

Remark: This module is not used in the final implementation of the project.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from typing_extensions import override

logger = logging.getLogger(__name__)


def make_speed_estimator(
    observation_parser: Any,
    speed_estimator_dict: dict,
) -> "SpeedEstimator":
    """Create a state estimator object."""
    speed_estimator_type = speed_estimator_dict.get("type", "default")
    speed_estimator_params = speed_estimator_dict.get("params", {})
    if speed_estimator_type == "low_pass":
        return LowPassSpeedEstimator(observation_parser=observation_parser, **speed_estimator_params)
    elif speed_estimator_type == "unity" or speed_estimator_type == "none" or speed_estimator_type == "default":
        return UnityStateEstimator(observation_parser=observation_parser, **speed_estimator_params)
    elif speed_estimator_type == "differential":
        return DifferentialSpeedEstimator(observation_parser=observation_parser, **speed_estimator_params)
    else:
        raise ValueError(f"Invalid state estimator type: {speed_estimator_type}")


class SpeedEstimator(ABC):
    """Abstract class for state estimation."""

    def __init__(
        self,
        observation_parser: Any,
        **kwargs: Any,
    ):
        """Initialize the state estimator."""
        self.observation_parser = observation_parser
        self._speed_estimate = 0.0
        self._angular_speed_estimate = 0.0

    @abstractmethod
    def update(self) -> None:
        """Update the state estimate given the current state and measurement."""
        raise NotImplementedError

    @property
    def speed_estimate(self) -> float:
        """Get the estimated speed."""
        return self._speed_estimate

    @property
    def angular_speed_estimate(self) -> float:
        """Get the estimated angular speed."""
        return self._angular_speed_estimate

    def reset(self) -> None:
        """Reset the state estimate."""
        self._speed_estimate = 0.0
        self._angular_speed_estimate = 0.0


class UnityStateEstimator(SpeedEstimator):
    """State estimator that directly uses the measurement as the state."""

    @override
    def update(self) -> None:
        self._speed_estimate = self.observation_parser.measured_drone_speed
        self._angular_speed_estimate = self.observation_parser.measured_drone_angular_speed


class LowPassSpeedEstimator(SpeedEstimator):
    """State estimator that uses a low-pass filter to estimate the speed."""

    def __init__(
        self,
        observation_parser: Any,
        alpha: float = 0.5,
        **kwargs: Any,
    ):
        """Initialize the low-pass speed estimator."""
        super().__init__(observation_parser, **kwargs)
        self.alpha = alpha
        self.previous_pos = None
        self.previous_rpy = None

        logger.info(f"Low-pass speed estimator initialized with alpha={self.alpha}.")

    @override
    def update(self) -> None:
        if self.previous_pos is None:
            self.previous_pos = self.observation_parser.drone_pos
            self.previous_rpy = self.observation_parser.drone_rpy
            return

        speed = (self.observation_parser.drone_pos - self.previous_pos) / self.observation_parser.dt
        self._speed_estimate = self.alpha * self._speed_estimate + (1 - self.alpha) * speed
        angular_speed = (self.observation_parser.drone_rpy - self.previous_rpy) / self.observation_parser.dt
        self._angular_speed_estimate = self.alpha * self._angular_speed_estimate + (1 - self.alpha) * angular_speed

class DifferentialSpeedEstimator(SpeedEstimator):
    """State estimator that uses a differential filter to estimate the speed."""

    def __init__(
        self,
        observation_parser: Any,
        **kwargs: Any,
    ):
        """Initialize the differential speed estimator."""
        super().__init__(observation_parser, **kwargs)
        self.previous_pos = None
        self.previous_rpy = None

        logger.info("Differential speed estimator initialized.")

    @override
    def update(self) -> None:
        if self.previous_pos is None:
            self.previous_pos = self.observation_parser.drone_pos
            self.previous_rpy = self.observation_parser.drone_rpy
            return

        speed = (self.observation_parser.drone_pos - self.previous_pos) / self.observation_parser.dt
        angular_speed = (self.observation_parser.drone_rpy - self.previous_rpy) / self.observation_parser.dt
        self._speed_estimate = speed
        self._angular_speed_estimate = angular_speed
