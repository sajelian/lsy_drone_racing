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
import matplotlib.pyplot as plt
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

N_RUNS = 100
"""
TODO:
-comments, docstrings
-proper stucture maybe extra file and class etc
-read gate positions from config yaml
-run 100 runs and adjust opacity
"""

def plot_violins(fig, ax, lap_times):

    ax.violinplot(lap_times, showmedians=False, showextrema=False)
    medianprops = dict(linewidth=1.5, color='black')
    ax.boxplot(lap_times, showfliers=False, whis = (0, 100), capwidths=0.0, medianprops=medianprops)
    x = np.random.normal(1, 0.01, size=len(lap_times))
    ax.scatter(x, lap_times, alpha=0.2, color="blue")
    ax.set_ylabel("Laptime [s]")
    ax.set_xlabel(f"PPO-Agent \n (n={N_RUNS})")
    

def plot_2d_trajectories(fig, ax, trajectories, collisions):
    """Plot the trajectories of drone runs."""
    for trajectory_id, trajectory in enumerate(trajectories):
        trajectory = np.array(trajectory)
        ax.plot(
            trajectory[:, 0], trajectory[:, 1], color="firebrick" if collisions[trajectory_id] else "navy", alpha=0.1
        )


def plot_gates(fig, ax, gates):
    """Plot the gate configurations for a drone racing course."""
    edge_size: float = 0.525

    for i, gate in enumerate(gates):
        ax.text(gate[0], gate[1], str(i))
        ax.plot(
            [
                gate[0] - edge_size / 2 * np.cos(gate[-2]),
                gate[0] + edge_size / 2 * np.cos(gate[-2]),
            ],
            [
                gate[1] - edge_size / 2 * np.sin(gate[-2]),
                gate[1] + edge_size / 2 * np.sin(gate[-2]),
            ],
            linewidth=5,
            color="grey",
        )

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])



def simulate(
    config: str = "config/level/level3.yaml", #config/level/level3_straight.yaml",
    controller: str = "controllers/ppo/ppo.py",
    controller_params: str = "models/ppo_lt_obs_act_rew_beta_act_2rel_num_timesteps_100000_time_06-28-14-11/params.yaml",
    n_runs: int = N_RUNS,
    gui: bool = False,
    terminate_on_lap: bool = False,
    log_level: str = "INFO",
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
    }
    ep_times = []

    # Plotting
    fig, ax = plt.subplots()
    fig1, ax1 = plt.subplots()

    # TODO get gate pos and yaw from level config file
    gates = config.quadrotor_config["gates"]
    
    plot_gates(fig, ax, gates)

    trajectories = []
    collisions = []

    # Run the episodes.
    for _ in range(n_runs):
        trajectory = []
        collision = False

        ep_start = time.time()
        done = False
        action = np.zeros(4)
        reward = 0
        obs, info = env.reset()
        # logger.info(info)
        info["ctrl_timestep"] = CTRL_DT
        info["ctrl_freq"] = CTRL_FREQ
        lap_finished = False
        # obs = [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r]
        ctrl = ctrl_class(obs, info, verbose=config.verbose, **controller_args)
        gui_timer = p.addUserDebugText("", textPosition=[0, 0, 1], physicsClientId=env.pyb_client_id)
        i = 0
        while not done:
            curr_time = i * CTRL_DT
            gui_timer = p.addUserDebugText(
                "Ep. time: {:.2f}s".format(curr_time),
                textPosition=[0, 0, 1.5],
                textColorRGB=[1, 0, 0],
                lifeTime=3 * CTRL_DT,
                textSize=1.5,
                parentObjectUniqueId=0,
                parentLinkIndex=-1,
                replaceItemUniqueId=gui_timer,
                physicsClientId=env.pyb_client_id,
            )

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

            # append current drone xy to trajectory
            x, y = env.observation_parser.drone_pos[0:2]

            trajectory.append(np.array([x, y]))

            # Update the controller internal state and models.
            ctrl.step_learn(action, obs, reward, done, info)
            # Add up reward, collisions, violations.
            stats["ep_reward"] += reward
            if info["collision"][1]:
                collision = True
                stats["collisions"] += 1
                stats["collision_objects"].add(info["collision"][0])
            stats["violations"] += "constraint_violation" in info and info["constraint_violation"]

            # Synchronize the GUI.
            if config.quadrotor_config.gui:
                sync(i, ep_start, CTRL_DT)
            i += 1
            # Break early after passing the last gate (=> gate -1) or task completion
            if terminate_on_lap and info["current_gate_id"] == -1:
                info["task_completed"], lap_finished = True, True

            if info["task_completed"]:
                logger.info("Task completed.")
                done = True
                lap_finished = True

        # save run data
        collisions.append(collision)
        trajectories.append(trajectory)


        # Learn after the episode if the controller supports it
        ctrl.episode_learn()  # Update the controller internal state and models.
        log_episode_stats(stats, info, config, curr_time, lap_finished)
        ctrl.episode_reset()
        # Reset the statistics
        stats["ep_reward"] = 0
        stats["collisions"] = 0
        stats["collision_objects"] = set()
        stats["violations"] = 0
        ep_times.append(curr_time if info["current_gate_id"] == -1 else None)

    # Close the environment
    env.close()

    plot_2d_trajectories(fig, ax, trajectories, collisions)
    plot_violins(fig1, ax1, ep_times)
    
    plt.tight_layout()
    
    fig.savefig("trajectories.pdf") #TODO adapt this to save to different files
    fig1.savefig("laptimes.pdf")
    
    plt.show()

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
        )
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(simulate)
