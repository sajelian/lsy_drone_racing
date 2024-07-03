from enum import Enum, auto  # noqa: D100
from typing import Any, Dict

import numpy as np
from safe_control_gym.controllers.firmware.firmware_wrapper import logging

from lsy_drone_racing.command import Command
from lsy_drone_racing.env_modifiers import ActionTransformer, ObservationParser

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

    def __init__(
        self,
        initial_goal: np.ndarray,
        model: Any,
        action_transformer: ActionTransformer,
        only_policy: bool = False,
        takeoff_duration: float = 1.0,
        takeoff_height: float = 0.2,
        landing_duration: float = 10.0,
        go_to_stabilization_duration: float = 3.0,
    ):
        """Initialize the State Machine.

        Args:
            initial_goal: Stabilization goal the drone is trying to reach before landing.
            model: Trained PPO model for policy control.
            action_transformer: The action transformer object.
            only_policy: If True, the drone will only use the policy control state.
            takeoff_duration: The duration of the takeoff state if only_policy is False.
            takeoff_height: The height the drone should takeoff to if only_policy is False.
            landing_duration: The duration of the landing state if only_policy is False.
            go_to_stabilization_duration: The duration of the go to stabilization state if only_policy is False.
        """
        self.goal = initial_goal
        self.model = model
        self.action_transformer = action_transformer
        self.only_policy = only_policy
        self.takeoff_duration = takeoff_duration if not only_policy else 0.0
        self.takeoff_height = takeoff_height
        self.landing_duration = landing_duration
        self.go_to_stabilization_duration = go_to_stabilization_duration

        self.stamp = 0
        self.state = DroneState.TAKEOFF

    def transition(
        self,
        ep_time: float,
        obs_parser: ObservationParser,
        info: Dict[str, Any],
    ) -> tuple:
        """Transition states inside state machine.

        Args:
            ep_time: current simulation episode time.
            obs_parser: The new observation parser object.
            info: The new info dict.
        """
        if self.only_policy:
            return self.policy_control(ep_time, obs_parser, info)

        if self.state == DroneState.TAKEOFF:
            self.state = DroneState.POLICY_CONTROL
            drone_takeoff_height = self.takeoff_height if obs_parser.drone_on_ground else obs_parser.drone_pos[2]
            return Command.TAKEOFF, [drone_takeoff_height, self.takeoff_duration]

        if self.state == DroneState.POLICY_CONTROL:
            return self.policy_control(ep_time, obs_parser, info)

        if self.state == DroneState.NOTIFY_SETPOINT_STOP and info["current_gate_id"] == -1:
            self.state = DroneState.GOTO
            return Command.GOTO, [self.goal, 0.0, self.go_to_stabilization_duration, False]

        if self.state == DroneState.GOTO and info["at_goal_position"]:
            self.state = DroneState.LAND
            return Command.LAND, [0.0, self.landing_duration]

        if self.state == DroneState.LAND:
            self.state = DroneState.FINISHED
            return Command.FINISHED, []

        return Command.NONE, []

    def policy_control(self, ep_time: float, obs_parser: np.ndarray, info: Dict[str, Any]) -> tuple:
        """Handle the policy control state.

        Args:
            ep_time: current simulation episode time.
            obs_parser: The new observation parser object.
            info: The new info dict.

        Returns:
            The command type and arguments to be sent to the quadrotor. See `Command`.
        """
        gate_id = info["current_gate_id"] if "current_gate_id" in info.keys() else info["current_target_gate_id"]
        if gate_id != -1 and ep_time > self.takeoff_duration:
            obs = obs_parser.get_observation()

            action, next_predicted_state = self.model.predict(obs, deterministic=True)
            transformed_action = self.action_transformer.transform(raw_action=action, obs_parser=obs_parser)
            firmware_action = self.action_transformer.create_firmware_action(transformed_action, sim_time=ep_time)
            command_type = Command.FULLSTATE
            return command_type, firmware_action

        if gate_id == -1 and not self.only_policy:
            self.state = DroneState.NOTIFY_SETPOINT_STOP
            return Command.NOTIFYSETPOINTSTOP, []

        return Command.NONE, []
