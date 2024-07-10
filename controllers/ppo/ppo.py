"""Controller using a PPO agent trained with stable-baselines3."""
from __future__ import annotations

from enum import Enum, auto  # noqa: D100
from typing import (
    Any,  # Python 3.10 type hints
    Dict,
)

import numpy as np
from safe_control_gym.controllers.firmware.firmware_wrapper import logging
from stable_baselines3 import PPO, SAC

from lsy_drone_racing.command import Command
from lsy_drone_racing.controller import BaseController
from lsy_drone_racing.env_modifiers import ActionTransformer, ObservationParser, RelativeActionTransformer

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
        only_policy: bool = True,
        takeoff_duration: float = 1.0,
        takeoff_height: float = 0.4,
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

class Controller(BaseController):
    """Template controller class."""

    def __init__(
        self,
        initial_obs: np.ndarray,
        initial_info: dict,
        buffer_size: int = 100,
        verbose: bool = False,
        model_name: str | None = None,
        action_transformer: str | None = None,
        **kwargs: Any,
    ):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori
            infromation contained in dictionary `initial_info`. Use this method to initialize
            constants, counters, pre-plan trajectories, etc.

        Args:
            initial_obs: The initial observation of the environment's state. Consists of
                [drone_xyz_yaw, gates_xyz_yaw, gates_in_range, obstacles_xyz, obstacles_in_range,
                gate_id]
            initial_info: The a priori information as a dictionary with keys 'symbolic_model',
                'nominal_physical_parameters', 'nominal_gates_pos_and_type', etc.
            buffer_size: Size of the data buffers used in method `learn()`.
            verbose: Turn on and off additional printouts and plots.
            model_name: The path to the trained model.
            action_transformer: The action transformer object.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(initial_obs, initial_info, buffer_size, verbose)

        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size

        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]

        self.reset()
        self.episode_reset()

        self.model_name = model_name if model_name else "models/working_model"
        self.model = PPO.load(self.model_name)
        # self.model = SAC.load(self.model_name)
        self.action_transformer = (
            ActionTransformer.from_yaml(action_transformer) if action_transformer else RelativeActionTransformer()
        )

        self._goal = np.array(
            [
                initial_info["x_reference"][0],
                initial_info["x_reference"][2],
                initial_info["x_reference"][4],
            ]
        )
        self.state_machine = DroneStateMachine(self._goal, self.model, self.action_transformer) 

    def compute_control(
        self,
        ep_time: float,
        obs: ObservationParser | np.ndarray,
        reward: float | None = None,
        done: bool | None = None,
        info: dict | None = None,
    ) -> tuple[Command, list]:
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        INSTRUCTIONS:
            Re-implement this method to return the target position, velocity, acceleration,
            attitude, and attitude rates to be sent from Crazyswarm to the Crazyflie using, e.g., a
            `cmdFullState` call.

        Args:
            ep_time: Episode's elapsed time, in seconds.
            obs: The environment's observation [drone_xyz_yaw, gates_xyz_yaw, gates_in_range,
                obstacles_xyz, obstacles_in_range, gate_id].
            reward: The reward signal.
            done: Wether the episode has terminated.
            info: Current step information as a dictionary with keys 'constraint_violation',
                'current_target_gate_pos', etc.

        Returns:
            The command type and arguments to be sent to the quadrotor. See `Command`.
        """
        command, args = self.state_machine.transition(ep_time, obs, info)
        return command, args
