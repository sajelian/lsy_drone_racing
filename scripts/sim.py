"""Simulate the competition as in the IROS 2022 Safe Robot Learning competition.sim.

Run as:

    $ python scripts/sim.py --config config/getting_started.yaml

Look for instructions in `README.md` and `edit_this.py`.
"""

from __future__ import annotations

import logging
import time
from functools import partial
from pathlib import Path

import fire
import numpy as np
import pybullet as p
import yaml
from munch import Munch, munchify
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import sync

from lsy_drone_racing.command import Command, apply_sim_command
from lsy_drone_racing.constants import FIRMWARE_FREQ
from lsy_drone_racing.env_modifiers import ActionTransformer, ObservationParser, Rewarder
from lsy_drone_racing.utils import load_controller
from lsy_drone_racing.wrapper import DroneRacingObservationWrapper

logger = logging.getLogger(__name__)


def simulate(
    config: str = "config/level/level3.yaml",
    controller: str = "controllers/ppo/ppo.py",
    controller_params: str = "models/ppo_l4_obs_gyro1Norm_rew_br50_act_2rel150_num_timesteps_30000000_time_07-21-16-06/params.yaml",
    n_runs: int = 5,
    gui: bool = True,
    terminate_on_lap: bool = True,
    log_level: str = "INFO",
    clock_time: bool = True,
) -> list[float]:
    """Evaluate the drone controller over multiple episodes.

    Args:
        config: The path to the configuration file.
        controller: The path to the controller module.
        controller_params: The path to the controller parameters.
        n_runs: The number of episodes.
        gui: Enable/disable the simulation GUI.
        terminate_on_lap: Stop the simulation early when the drone has passed the last gate.
        log_level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        clock_time: Synchronize the simulation with the real time.

    Returns:
        A list of episode times.
    """
    # Load configuration and check if firmare should be used.
    path = Path(config)
    assert path.exists(), f"Configuration file not found: {path}"
    with open(path, "r") as file:
        config = munchify(yaml.safe_load(file))
    # Overwrite config options
    config.quadrotor_config.gui = gui
    CTRL_FREQ = config.quadrotor_config["ctrl_freq"]
    CTRL_DT = 1 / CTRL_FREQ

    # Set the logging level
    assert log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    logger.setLevel(log_level)

    # Load the controller module
    path = Path(__file__).parents[1] / controller
    ctrl_class = load_controller(path)  # This returns a class, not an instance

    controller_args = {}
    extra_env_args = {}
    if controller == "controllers/ppo/ppo.py":
        # Load the controller parameters
        path = Path(__file__).parents[1] / controller_params
        assert path.exists(), f"Controller parameters file not found: {path}, and needed for PPO."
        with open(path, "r") as file:
            controller_args = yaml.safe_load(file)
            # TODO: dont hardcode the observation parser 
            extra_env_args["observation_parser"] = ObservationParser.from_yaml(
                n_gates=4, n_obstacles=4, file_path=controller_args["observation_parser"]
            )
            extra_env_args["action_transformer"] = ActionTransformer.from_yaml(controller_args["action_transformer"])
            extra_env_args["rewarder"] = Rewarder.from_yaml(controller_args["rewarder"])

    # Create environment.
    assert config.use_firmware, "Firmware must be used for the competition."
    pyb_freq = config.quadrotor_config["pyb_freq"]
    assert pyb_freq % FIRMWARE_FREQ == 0, "pyb_freq must be a multiple of firmware freq"
    # The env.step is called at a firmware_freq rate, but this is not as intuitive to the end
    # user, and so we abstract the difference. This allows ctrl_freq to be the rate at which the
    # user sends ctrl signals, not the firmware.
    config.quadrotor_config["ctrl_freq"] = FIRMWARE_FREQ
    env_func = partial(make, "quadrotor", **config.quadrotor_config)
    env = DroneRacingObservationWrapper(make("firmware", env_func, FIRMWARE_FREQ, CTRL_FREQ), **extra_env_args)

    # Create a statistics collection
    stats = {
        "ep_reward": 0,
        "collisions": 0,
        "collision_objects": set(),
        "violations": 0,
        "gates_passed": 0,
        "top_speed": 0,
        "top_angular_speed": 0,
    }
    ep_times = []

    # Run the episodes.
    for _ in range(n_runs):
        ep_start = time.time()
        done = False
        action = np.zeros(4)
        reward = 0

        p.resetDebugVisualizerCamera(
            cameraDistance=0.5,
            cameraYaw=0,
            cameraPitch=-45,
            cameraTargetPosition=[0, 0, 0],
            physicsClientId=env.pyb_client_id,
        )

        
        obs, info = env.reset()
        #logger.info(info)
        info["ctrl_timestep"] = CTRL_DT
        info["ctrl_freq"] = CTRL_FREQ
        lap_finished = False
        # obs = [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r]
        ctrl = ctrl_class(obs, info, verbose=config.verbose, **controller_args)
        gui_timer = p.addUserDebugText("", textPosition=[0, 0, 1], physicsClientId=env.pyb_client_id)
        i = 0

        top_speed = 0
        top_angular_speed = 0

        while not done:
            curr_time = i * CTRL_DT

            # Get the observation from the motion capture system
            # Compute control input.
            command_type, args = ctrl.compute_control(curr_time, env.observation_parser, reward, done, info)
            # Apply the control input to the drone. This is a deviation from the gym API as the
            # action is not applied in env.step()
            applied_transformed_action = None
            if command_type == Command.FULLSTATE:
                applied_transformed_action = [*(args[0].tolist()), args[3]]
            apply_sim_command(env, command_type, args)
            kwargs = {"applied_transformed_action": applied_transformed_action} if applied_transformed_action else {}
            obs, reward, done, info, action = env.step(sim_time=curr_time, action=action, **kwargs)
            # Update the controller internal state and models.

            follow_drone = True
            # We place the camera to follow the drone
            drone_pos = env.observation_parser.drone_pos
            drone_rpy = env.observation_parser.drone_rpy

            gui_timer = p.addUserDebugText(
                "Ep. time: {:.2f}s".format(curr_time),
                textPosition=[0, 0, 1.5] if not follow_drone else [drone_pos[0], drone_pos[1], drone_pos[2] + 0.2],
                textColorRGB=[1, 0, 0],
                lifeTime=3 * CTRL_DT,
                textSize=2.5,
                parentObjectUniqueId=0,
                parentLinkIndex=-1,
                replaceItemUniqueId=gui_timer,
                physicsClientId=env.pyb_client_id,
            )
            p.resetDebugVisualizerCamera(
                cameraDistance=0.5,
                cameraYaw=drone_rpy[2],
                cameraPitch=drone_rpy[1] - 45,
                cameraTargetPosition=drone_pos,
                physicsClientId=env.pyb_client_id,
            )

            ctrl.step_learn(action, obs, reward, done, info)
            # Add up reward, collisions, violations.
            stats["ep_reward"] += reward
            if info["collision"][1]:
                stats["collisions"] += 1
                stats["collision_objects"].add(info["collision"][0])
            stats["violations"] += "constraint_violation" in info and info["constraint_violation"]

            stats["top_speed"] = np.max([stats["top_speed"], np.linalg.norm(env.observation_parser.drone_speed)])
            stats["top_angular_speed"] = np.max(
                [stats["top_angular_speed"], np.linalg.norm(env.observation_parser.drone_angular_speed)]
            )


            # Synchronize the GUI.
            if clock_time:
                wall_clock_duration = time.time() - ep_start
                if wall_clock_duration > curr_time:
                    time.sleep((wall_clock_duration - curr_time) / 1000.0)
                
            i += 1
            # Break early after passing the last gate (=> gate -1) or task completion
            if terminate_on_lap and info["current_gate_id"] == -1:
                info["task_completed"], lap_finished = True, True

            if info["task_completed"]:
                logger.info("Task completed.")
                done = True
                lap_finished = True

        # Learn after the episode if the controller supports it
        ctrl.episode_learn()  # Update the controller internal state and models.
        log_episode_stats(stats, info, config, curr_time, lap_finished)
        ctrl.episode_reset()
        # Reset the statistics
        stats["ep_reward"] = 0
        stats["collisions"] = 0
        stats["collision_objects"] = set()
        stats["violations"] = 0
        stats["top_speed"] = 0
        stats["top_angular_speed"] = 0

        ep_times.append(curr_time if info["current_gate_id"] == -1 else None)

    # Close the environment
    env.close()
    return ep_times


def log_episode_stats(stats: dict, info: dict, config: Munch, curr_time: float, lap_finished: bool):
    """Log the statistics of a single episode."""
    stats["gates_passed"] = info["current_gate_id"]
    if stats["gates_passed"] == -1:  # The drone has passed the final gate
        stats["gates_passed"] = len(config.quadrotor_config.gates)
    if config.quadrotor_config.done_on_collision and info["collision"][1]:
        termination = "COLLISION"
    elif config.quadrotor_config.done_on_completion and info["task_completed"] or lap_finished:
        termination = "TASK COMPLETION"
    elif config.quadrotor_config.done_on_violation and info["constraint_violation"]:
        termination = "CONSTRAINT VIOLATION"
    else:
        termination = "MAX EPISODE DURATION"
    logger.info(
        (
            f"Flight time (s): {curr_time}\n"
            f"Reason for termination: {termination}\n"
            f"Gates passed: {stats['gates_passed']}\n"
            f"Total reward: {stats['ep_reward']}\n"
            f"Number of collisions: {stats['collisions']}\n"
            f"Number of constraint violations: {stats['violations']}\n"
            f"Top speed: {stats['top_speed']}\n"
            f"Top angular speed: {stats['top_angular_speed']}\n"
        )
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(simulate)
