"""Example training script using the stable-baselines3 library."""

from __future__ import annotations

import datetime
import logging
from functools import partial
from pathlib import Path

import fire
import yaml
from munch import munchify
from safe_control_gym.utils.registration import make
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy

from lsy_drone_racing.constants import FIRMWARE_FREQ
from lsy_drone_racing.wrapper import DroneRacingWrapper

logger = logging.getLogger(__name__)


def create_race_env(
    level_path: Path,
    observation_parser_path: Path,
    rewarder_path: Path,
    action_transformer_path: Path,
    gui: bool = False,
    seed: int = 0,
    terminate_on_lap: bool = False,
) -> DroneRacingWrapper:
    """Create the drone racing environment."""
    # Load configuration and check if firmare should be used.
    assert level_path.exists(), f"Configuration file not found: {level_path}"
    with open(level_path, "r") as file:
        config = munchify(yaml.safe_load(file))
    # Overwrite config options
    config.quadrotor_config.gui = gui
    CTRL_FREQ = config.quadrotor_config["ctrl_freq"]
    # Create environment
    assert config.use_firmware, "Firmware must be used for the competition."
    pyb_freq = config.quadrotor_config["pyb_freq"]
    assert pyb_freq % FIRMWARE_FREQ == 0, "pyb_freq must be a multiple of firmware freq"
    config.quadrotor_config["ctrl_freq"] = FIRMWARE_FREQ
    env_factory = partial(make, "quadrotor", **config.quadrotor_config)
    firmware_env = make("firmware", env_factory, FIRMWARE_FREQ, CTRL_FREQ)
    return DroneRacingWrapper(
        firmware_env,
        terminate_on_lap=terminate_on_lap,
        observation_parser_path=observation_parser_path,
        rewarder_path=rewarder_path,
        action_transformer_path=action_transformer_path,
    )


def main(
    level: str = "config/level/blabla.yaml",
    controller_params: str = "models/ppo/params.yaml",
    gui: bool = True,
    seed: int = 0,
    n_eval: int = 10,
    terminate_on_lap: bool = False,
):
    """Create the environment, check its compatibility with sb3, and run a PPO agent."""
    project_path = Path(__file__).resolve().parents[2]

    level_path = project_path / level
    with open(controller_params, "r") as file:
        params = munchify(yaml.safe_load(file))

    observation_parser = params.observation_parser
    rewarder = params.rewarder
    action_transformer = params.action_transformer
    model_type = params.model_type

    observation_parser_path = project_path / observation_parser
    rewarder_path = project_path / rewarder
    action_transformer_path = project_path / action_transformer
    allowed_model_types = {"ppo": PPO, "sac": SAC}
    model_class = allowed_model_types[model_type]

    # Set level name and path
    level_name = level.split("/")[-1].split(".")[0]
    level_short_name = level_name[0] + level_name[-1]

    env = create_race_env(
        level_path=level_path,
        observation_parser_path=observation_parser_path,
        rewarder_path=rewarder_path,
        action_transformer_path=action_transformer_path,
        gui=gui,
        seed=seed,
        terminate_on_lap=terminate_on_lap,
    )

    observation_parser_shortname = env.observation_parser.get_shortname()
    rewarder_shortname = env.rewarder.get_shortname()
    action_transformer_shortname = env.action_transformer.get_shortname()
    date_now = datetime.datetime.now().strftime("%m-%d-%H-%M")

    logger.info(
        f"Evaluating on {level_short_name} level "
        + f"with {observation_parser_shortname}"
        + f" observation parser, {rewarder_shortname} rewarder,"
        + f" {action_transformer_shortname} action transformer. "
        + f"Time: {date_now}"
    )

    # Sanity check to ensure the environment conforms to the sb3 API
    # check_env(env)
    model = model_class.load(params.model_name)

    try:
        mean_reward, std_reward = evaluate_policy(
            model=model,
            env=env,
            n_eval_episodes=n_eval,
            deterministic=True,
        )
        logger.info(f"Mean reward: {mean_reward:.2f}, Std reward: {std_reward:.2f}")
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted. Saving model.")


if __name__ == "__main__":
    fire.Fire(main)
