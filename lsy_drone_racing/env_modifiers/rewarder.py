"""Rewarder class for custom rewards."""

import logging

import numpy as np
import yaml

from lsy_drone_racing.env_modifiers.observation_parser import ObservationParser

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
        z_penalty: float = 0.0,
        z_penalty_threshold: float = np.inf,
        action_smoothness: float = 1e-4,
        body_rate_penalty: float = -1e-3,
        speed_threshold: float = np.inf,
        speed_penalty: float = 0.0,
        angular_speed_threshold: float = np.inf,
        angular_speed_penalty: float = 0.0,
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
    def from_yaml(cls, file_path: str) -> "Rewarder":  # noqa: ANN102
        """Load the rewarder from a YAML file.

        Args:
            file_path: The path to the YAML file.

        Returns:
            The rewarder.
        """
        if file_path is None or file_path == "":
            logger.error("No rewarder file path provided. Returning default rewarder.")
            return cls()
        try:
            with open(file_path, "r") as file:
                data = yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading rewarder from {file_path}: {e}")
            logger.error("Returning default rewarder.")
            data = {}
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

        reward += self.body_rate_penalty * np.linalg.norm(obs_parser.drone_angular_speed)

        if action is not None:
            reward += -np.linalg.norm(action - obs_parser.previous_action) * self.action_smoothness

        if obs_parser.just_passed_gate:
            reward += self.gate_reached

        return reward

