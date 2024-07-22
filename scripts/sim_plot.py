"""Simulate the competition as in the IROS 2022 Safe Robot Learning competition.sim.

Run as:

    $ python scripts/sim.py --config config/getting_started.yaml

Look for instructions in `README.md` and `edit_this.py`.
"""

from __future__ import annotations

import csv
import logging
import time
from functools import partial
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybullet as p
import seaborn as sns
import yaml
from matplotlib.colors import ListedColormap
from munch import Munch, munchify
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import sync

from lsy_drone_racing.command import Command, apply_sim_command
from lsy_drone_racing.constants import FIRMWARE_FREQ
from lsy_drone_racing.env_modifiers import ActionTransformer, ObservationParser, Rewarder
from lsy_drone_racing.utils import load_controller
from lsy_drone_racing.wrapper import DroneRacingObservationWrapper

#TODO change fonts to Latex style
plt.rcParams.update({'font.size': 16})


logger = logging.getLogger(__name__)

N_RUNS = 50
N_AGENTS = 1


"""
TODO:
-comments, docstrings
-proper stucture maybe extra file and class etc
-multiple violins, mutiple bars.
"""


def plot_violins(ax, agents_lap_times, model_names):
    """Plot the lap times as violin plots."""
    colors_lst = ["royalblue", "darkorange", "gold", "darkorchid"]
    medianprops = dict(linewidth=1.5, color="black")
    for i, lap_times in enumerate(agents_lap_times):
        if len(lap_times) > 0:
            vp_plot = ax.violinplot(lap_times, vert=True, positions=[i + 1], showmedians=False, showextrema=False)
            for vp in vp_plot["bodies"]:
                vp.set_facecolor(colors_lst[i])
            ax.boxplot(
                lap_times,
                positions=[i + 1],
                vert=True,
                showfliers=False,
                whis=(0, 100),
                capwidths=0.0,
                medianprops=medianprops,
            )
            x = np.random.normal(i + 1, 0.01, size=len(lap_times))
            ax.scatter(x, lap_times, alpha=0.4, color=colors_lst[i])
            ax.set_ylabel("Laptime [s]")

    ax.set_xticks(np.arange(1, len(model_names) + 1), labels=model_names)


def plot_2d_trajectories(fig, ax, trajectories, collisions):
    """Plot the trajectories of drone runs."""
    plotted_collission = False
    plotted_success = False
    alpha = 0.3
    for trajectory_id, trajectory in enumerate(trajectories):
        trajectory = np.array(trajectory)
        if not plotted_collission and collisions[trajectory_id]:
            plotted_collission = True
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color="firebrick",
                label="collision",
                alpha=alpha,
                linewidth=2,
            )
        elif not plotted_success and not collisions[trajectory_id]:
            plotted_success = True
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color="navy",
                label="success",
                alpha=alpha,
                linewidth=2,
            )
        else:
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color="firebrick" if collisions[trajectory_id] else "navy",
                alpha=alpha,
                linewidth=2,
            )
        if collisions[trajectory_id]:
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color="black", marker="x", s=50, alpha=0.5)

    ax.legend()


def plot_gate_reached_percentage(ax, agents_gate_reached, model_names):
    """Plot the gate reached percentage of drone runs."""
    flatui = sns.color_palette("deep")
    my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())

    num_agents = len(agents_gate_reached)
    num_gates = len(agents_gate_reached[0])

    # Define the width of each bar
    bar_width = 0.8 / num_agents

    # Define colors for each agent (this can be expanded or modified as needed)
    colors_lst = plt.cm.get_cmap(my_cmap, num_agents).colors

    # Loop through each agent
    for agent_index, gate_reached in enumerate(agents_gate_reached):
        total_gate_sum = sum(gate_reached)

        # Calculate positions for bars for this agent
        positions = np.arange(num_gates) + agent_index * (bar_width)

        # Plot each gate's percentage reached
        percentages = [gate_i_reached_amount / total_gate_sum * 100 for gate_i_reached_amount in gate_reached]
        ax.barh(
            positions,
            percentages,
            bar_width - 0.05,
            label=f"{model_names[agent_index]}",
            color=colors_lst[agent_index],
        )

    # Set y-ticks to be in the center of the grouped bars
    ax.set_yticks(np.arange(num_gates) + (num_agents - 1) * bar_width / 2)
    ax.set_yticklabels([f"Gate {i}" for i in range(num_gates)])

    # Set labels and title
    ax.set_ylabel("Gate Passed")
    ax.set_xlabel(f"Percentage of Total Runs [%] \n (n={N_RUNS})")

    # Add a legend
    ax.legend()


def plot_parcour(fig, ax, gates, obstacles):
    """Plot the gate configurations for a drone racing course."""
    edge_size: float = 0.525

    for i, gate in enumerate(gates):
        ax.text(
            gate[0] + edge_size / 2 * np.cos(gate[-2]),
            gate[1] + edge_size / 2 * np.sin(gate[-2]),
            str(i),
            fontsize=15,
            fontweight="bold",
        )
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
    for obstacle in obstacles:
        ax.scatter(obstacle[0], obstacle[1], color="blue", marker="o", s=100)

    ax.set_aspect("equal")
    ax.set_xlabel(f"x [m] \n (n={N_RUNS})")
    ax.set_ylabel("y [m]")
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])


def simulate(
    model_name: str = "model_name",
    config: str = "config/level/level3.yaml",
    controller: str = "controllers/ppo/ppo.py",
    n_runs: int = N_RUNS,
    gui: bool = False,
    terminate_on_lap: bool = True,
    log_level: str = "INFO",
) -> list[float]:
    """Evaluate the drone controller over multiple episodes.

    Args:
        model_name: The name of the learned model file.
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
    controller_params = f"models/{model_name}/params.yaml"
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

    # Data lists for plotting
    ep_times = []
    trajectories = []
    collisions = []
    gate_reached = [0, 0, 0, 0, 0]

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
        # increment reached gate counter
        if info["current_gate_id"] == -1:
            gate_reached[4] += 1
        else:
            gate_reached[info["current_gate_id"]] += 1

        # Learn after the episode if the controller supports it
        ctrl.episode_learn()  # Update the controller internal state and models.
        log_episode_stats(stats, info, config, curr_time, lap_finished)
        ctrl.episode_reset()
        # Reset the statistics
        stats["ep_reward"] = 0
        stats["collisions"] = 0
        stats["collision_objects"] = set()
        stats["violations"] = 0
        if info["current_gate_id"] == -1:
            ep_times.append(curr_time)

        print(_)

    # Close the environment
    env.close()

    # Plotting
    fig, ax = plt.subplots()
    # fig1, ax1 = plt.subplots()
    # fig2, ax2 = plt.subplots()

    # TODO add this to trajectories plot
    OBSTACLES = config.quadrotor_config["obstacles"]
    GATES = config.quadrotor_config["gates"]

    plot_parcour(fig, ax, GATES, OBSTACLES)
    plot_2d_trajectories(fig, ax, trajectories, collisions)
    # if len(ep_times) > 0:
    #     plot_violins(fig1, ax1, ep_times)

    # plot_gate_reached_percentage(fig2, ax2, gates_reached)

    fig.savefig("trajectories.pdf", bbox_inches="tight")
    # fig1.savefig("laptimes.pdf", bbox_inches="tight")
    # fig2.savefig("gate_reached.pdf", bbox_inches="tight")

    return ep_times, gate_reached, trajectories, collisions


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

    READ = False
    RUN = True

    # models to iterate over
    # models = [
    #     "ppo_l4_obs_actNorm_rew_beta_act_2rel25_num_timesteps_2000000_time_07-05-14-32",
    #     "ppo_l4_obs_actNorm_rew_beta_act_2rel_num_timesteps_5000000_time_07-06-00-03",
    #     "ppo_l4_obs_actNorm_rew_beta_act_2rel100_num_timesteps_5000000_time_07-06-23-40",
    #     "ppo_l3_obs_act_rew_beta_act_2rel25_num_timesteps_5000000_time_07-01-22-47",
    # ]

    models = [
        "ppo_l4_obs_actNorm_rew_beta_act_2rel100_num_timesteps_5000000_time_07-06-23-40"
    ]
    
    # model names
    #model_names = ["$\gamma_0 = 0.025$", "$\gamma_0 = 0.05$", "$\gamma_0 = 0.1$", "$\gamma_0 = 0.025$ NonNorm"]
    model_names = ["$\gamma_0 = 0.1$"]

    # Lists for plotting
    agents_laptimes = []
    agents_gate_reached = []
    trajectories_lst = []
    collisions_lst = []


    # plotting
    laptimes_fig, laptimes_ax = plt.subplots()
    gate_reached_fig, gate_reached_ax = plt.subplots()
    trajectories_fig, trajectories_ax = plt.subplots()

    if RUN:
        # for i, model in enumerate(models):
        for i in range(N_AGENTS):
            print(f"Model Iteration: {i}")

            laptimes, gate_reached, trajectories, collisions = simulate(models[i])

            agents_laptimes.append(laptimes)
            agents_gate_reached.append(gate_reached)

            if i == 2:
                trajectories_lst = trajectories
                collisions_lst = collisions

        # safe to csv
        for i, laptimes in enumerate(agents_laptimes):
            np.savetxt(f"agent{i}_laptimes.csv", laptimes, delimiter=",")

        for i, gate_reached in enumerate(agents_gate_reached):
            np.savetxt(f"agent{i}_gate_reached.csv", gate_reached, delimiter=",")

    if READ:
        # read laptime data from csv file
        for i in range(N_AGENTS):
            with open(f"agent{i}_laptimes.csv", "r") as file:
                data = csv.reader(file, delimiter=",")

                # Convert the CSV data to a list of floats
                laptimes = []
                for row in data:
                    laptimes.extend([float(time) for time in row])

                # Append the list of floats to the agents_laptimes list
                agents_laptimes.append(laptimes)

        for i in range(N_AGENTS):
            with open(f"agent{i}_gate_reached.csv", "r") as file:
                data = csv.reader(file, delimiter=",")

                # Convert the CSV data to a list of floats
                gate_reached = []
                for row in data:
                    gate_reached.extend([float(time) for time in row])

                # Append the list of floats to the agents_laptimes list
                agents_gate_reached.append(gate_reached)

    # call plotting functions
    #plot_violins(laptimes_ax, agents_laptimes, model_names)
    #plot_gate_reached_percentage(gate_reached_ax, agents_gate_reached, model_names)
    #plot_parcour(trajectories_fig, trajectories_ax, GATES, OBSTACLE)
    #plot_2d_trajectories(trajectories_fig, trajectories_ax, trajectories_lst, collisions_lst)

    # safe figs
    #laptimes_fig.autofmt_xdate()
    #gate_reached_fig.autofmt_xdate()
    #laptimes_fig.savefig("laptimes.pdf", bbox_inches="tight")
    #gate_reached_fig.savefig("gate_reached.pdf", bbox_inches="tight")
    #trajectories_fig.savefig("trajectories.pdf", bbox_inches="tight")
    
    #for i, fig in enumerate(trajectories_figs):
    #    fig.savefig(f"agent{i}_trajectory.pdf", bbox_inches="tight")

    # show plot
    plt.tight_layout()
    plt.show()
