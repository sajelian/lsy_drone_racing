"""Example training script using the stable-baselines3 library."""

from __future__ import annotations

import datetime
import logging
from functools import partial
from pathlib import Path
from typing import Any

import fire
import yaml
from munch import munchify
from safe_control_gym.utils.registration import make
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

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


class IncreaseDifficultyCallback(BaseCallback):
    def __init__(
        self,
        model: PPO,
        level_list: list[str],
        observation_parser_path: Path,
        rewarder_path: Path,
        action_transformer_path: Path,
        verbose: int = 0,
        increase_difficulty_reward: float = -1000,
        gui: bool = False,
        seed: int = 0,
        terminate_on_lap: bool = False,
        **kwargs: Any,
    ):
        super().__init__(verbose=verbose)
        self.model = model
        self.level_list = level_list
        self.observation_parser_path = observation_parser_path
        self.rewarder_path = rewarder_path
        self.action_transformer_path = action_transformer_path
        self.gui = gui
        self.seed = seed
        self.terminate_on_lap = terminate_on_lap
        self.increase_difficulty_reward = increase_difficulty_reward
        self.current_level = 0

    def _on_step(self) -> bool:
        last_mean_reward = self.parent.last_mean_reward
        logger.info(f"Last mean reward: {self.parent.last_mean_reward}")
        if last_mean_reward > self.increase_difficulty_reward and self.current_level < len(self.level_list) - 1:
            logger.info("Increasing difficulty since mean reward is above threshold.")
            self.current_level += 1
            self.model.set_env(
                create_race_env(
                    level_path=self.level_list[self.current_level],
                    observation_parser_path=self.observation_parser_path,
                    rewarder_path=self.rewarder_path,
                    action_transformer_path=self.action_transformer_path,
                    gui=self.gui,
                    seed=self.seed,
                    terminate_on_lap=self.terminate_on_lap,
                )
            )
            self.parent.env = create_race_env(
                level_path=self.level_list[self.current_level],
                observation_parser_path=self.observation_parser_path,
                rewarder_path=self.rewarder_path,
                action_transformer_path=self.action_transformer_path,
                gui=self.gui,
                seed=self.seed,
                terminate_on_lap=self.terminate_on_lap,
            )
        return True


def main(
    level: str = "config/level/blabla.yaml",
    level_list: str = None,
    observation_parser: str = "config/observation_parser/default.yaml",
    rewarder: str = "config/rewarder/default.yaml",
    action_transformer: str = "config/action_transformer/default.yaml",
    gui: bool = False,
    gui_eval: bool = False,
    log_level: int = logging.INFO,
    seed: int = 0,
    num_timesteps: int = 500_000,
    model_type: str = "ppo",
    terminate_on_lap: bool = False,
):
    """Create the environment, check its compatibility with sb3, and run a PPO agent."""
    logging.basicConfig(level=log_level)

    project_path = Path(__file__).resolve().parents[2]

    if level_list:
        level_list_path = project_path / level_list
        with open(level_list_path, "r") as file:
            level_list_config = munchify(yaml.safe_load(file))
        levels_list = [project_path / level_list_item for level_list_item in level_list_config.levels]
        logger.info(f"Training on levels: {levels_list}")
    else:
        levels_list = []

    level_path = project_path / levels_list[0] if level_list else project_path / level
    observation_parser_path = project_path / observation_parser
    rewarder_path = project_path / rewarder
    action_transformer_path = project_path / action_transformer
    allowed_model_types = {"ppo": PPO, "sac": SAC}
    assert model_type in allowed_model_types.keys(), f"Model type must be one of {allowed_model_types}"
    model_class = allowed_model_types[model_type]

    # Set level name and path
    if level_list:
        level_short_name = level_list
    else:
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
        f"Training {level_short_name} level "
        + f"with {observation_parser_shortname}"
        + f" observation parser, {rewarder_shortname} rewarder,"
        + f" {action_transformer_shortname} action transformer. "
        + f"Time: {date_now}"
    )

    train_name = f"{model_type}_" + "_".join(
        [
            level_short_name,
            "obs",
            observation_parser_shortname,
            "rew",
            rewarder_shortname,
            "act",
            action_transformer_shortname,
            "num_timesteps",
            str(num_timesteps),
            "time",
            date_now,
        ]
    )

    best_model_save_path = f"models/{train_name}/best_model"

    # We save the params as a yaml file for reproducibility
    Path(f"models/{train_name}").mkdir(parents=True, exist_ok=True)
    with open(f"models/{train_name}/params.yaml", "w") as file:
        yaml.dump(
            {
                "level": level,
                "observation_parser": observation_parser,
                "rewarder": rewarder,
                "action_transformer": action_transformer,
                "gui": gui,
                "seed": seed,
                "num_timesteps": num_timesteps,
                "model_name": f"{best_model_save_path}/best_model",
                "model_type": model_type,
            },
            file,
        )

    # Sanity check to ensure the environment conforms to the sb3 API
    # check_env(env)
    model = model_class(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        verbose=1,
        tensorboard_log="logs",
    )  # Train the agent

    eval_env = create_race_env(
        level_path=level_path,
        observation_parser_path=observation_parser_path,
        rewarder_path=rewarder_path,
        action_transformer_path=action_transformer_path,
        gui=gui_eval,
        seed=seed,
        terminate_on_lap=terminate_on_lap,
    )

    increase_difficulty = IncreaseDifficultyCallback(
        model=model,
        level_list=levels_list,
        observation_parser_path=observation_parser_path,
        rewarder_path=rewarder_path,
        action_transformer_path=action_transformer_path,
        gui=gui,
        seed=seed,
        terminate_on_lap=terminate_on_lap,
    )
    eval_callback = EvalCallback(
        eval_env,
        # callback_after_eval=increase_difficulty,
        best_model_save_path=best_model_save_path,
        log_path="./logs/",
        eval_freq=100_000,
        deterministic=True,
    )

    try:
        model.learn(
            total_timesteps=num_timesteps,
            progress_bar=True,
            tb_log_name=train_name,
            callback=eval_callback,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted. Saving model.")

    model.save(f"models/{train_name}/{train_name}")


if __name__ == "__main__":
    fire.Fire(main)
