from enum import Enum, auto  # noqa: D100
from typing import Any, Dict

import numpy as np
from safe_control_gym.controllers.firmware.firmware_wrapper import logging

from lsy_drone_racing.command import Command
from lsy_drone_racing.env_modifiers import transform_action

logger = logging.getLogger(__name__)


class DroneState(Enum):
    """Class to define the Drone State Machine states."""

    TAKEOFF = auto()
    POLICY_CONTROL = auto()
    NOTIFY_SETPOINT_STOP = auto()
    GOTO = auto()
    LAND = auto()
    FINISHED = auto()
    NONE = auto()


class DroneStateMachine:
    """Class to handle the Drone State Machine transitions."""

    def __init__(self, initial_goal: np.ndarray, model: Any):
        """Initialize the State Machine.

        Args:
            initial_goal: Stabilization goal the drone is trying to reach before landing.
            model: Trained PPO model for policy control.
        """
        self.state = DroneState.TAKEOFF
        self.goal = initial_goal
        self.model = model
        self.stamp = 0

    def transition(
        self,
        ep_time: float,
        obs: np.ndarray,
        info: Dict[str, Any],
    ) -> tuple:
        """Transition states inside state machine.

        Args:
            ep_time: current simulation episode time.
            obs: The new observation array.
            info: The new info dict.
        """
        if self.state == DroneState.TAKEOFF:
            self.state = DroneState.POLICY_CONTROL
            return Command.TAKEOFF, [0.4, 2]

        elif self.state == DroneState.POLICY_CONTROL:          
            return self.policy_control(ep_time, obs, info)

        elif self.state == DroneState.NOTIFY_SETPOINT_STOP and info["current_gate_id"] == -1:
            self.state = DroneState.GOTO
            return Command.GOTO, [self.goal, 0.0, 3.0, False]

        elif self.state == DroneState.GOTO and info["at_goal_position"]:
            self.state = DroneState.LAND
            return Command.LAND, [0.0, 10]

        elif self.state == DroneState.LAND:
            self.state = DroneState.FINISHED
            return Command.FINISHED, []

        return Command.NONE, []

    def policy_control(self, ep_time: float, obs: np.ndarray, info: Dict[str, Any]) -> tuple:
        """Handle the policy control state.

        Args:
            ep_time: current simulation episode time.
            obs: The new observation array.
            info: The new info dict.

        Returns:
            The command type and arguments to be sent to the quadrotor. See `Command`.
        """
        if ep_time - 2 > 0 and info["current_gate_id"] != -1:
            action, next_predicted_state = self.model.predict(obs, deterministic=True)
            action = transform_action(action, drone_pos=obs[:3])

            x = float(action[0])
            y = float(action[1])
            z = float(action[2])
            target_pos = np.array([x, y, z])
            target_vel = np.zeros(3)
            target_acc = np.zeros(3)
            target_yaw = float(action[3])
            target_rpy_rates = np.zeros(3)
            command_type = Command.FULLSTATE
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates, ep_time]
            return command_type, args

        if info["current_gate_id"] == -1:
            self.state = DroneState.NOTIFY_SETPOINT_STOP
            return Command.NOTIFYSETPOINTSTOP, []

        return Command.NONE, []
