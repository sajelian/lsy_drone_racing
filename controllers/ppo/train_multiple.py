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
    """Callback to increase the difficulty of the environment when the mean reward is above a threshold."""

    def __init__(
        self,
        threshold: float = 13.0,
        verbose: int = 0,
        **kwargs: Any,
    ):
        """Initialize the callback.

        Args:
            threshold: The threshold above which the mean reward should be to increase the difficulty.
            verbose: The verbosity level.
            **kwargs: Additional arguments.
        """
        super().__init__(verbose=verbose)
        self.threshold = threshold

    def _on_step(self) -> bool:
        last_mean_reward = self.parent.last_mean_reward
        logger.info(f"Last mean reward: {self.parent.last_mean_reward}")
        if last_mean_reward > self.threshold:
            logger.info("============================================================")
            logger.info(f"Increasing difficulty since mean reward is above threshold {self.threshold}.")
            logger.info("============================================================")
            return False
        else:
            return True


def main(
    levels: str,
    observation_parser: str = "config/observation_parser/default.yaml",
    rewarder: str = "config/rewarder/default.yaml",
    action_transformer: str = "config/action_transformer/default.yaml",
    gui: bool = False,
    gui_eval: bool = False,
    log_level: int = logging.INFO,
    seed: int = 0,
    num_timesteps: int = 500_000,
    eval_freq: int = 100_000,
    model_type: str = "ppo",
    terminate_on_lap: bool = False,
):
    """Create the environment, check its compatibility with sb3, and run a PPO agent."""
    logging.basicConfig(level=log_level)

    project_path = Path(__file__).resolve().parents[2]

    level_list_path = project_path / levels
    with open(level_list_path, "r") as file:
        levels_config = munchify(yaml.safe_load(file))

    level_list = [project_path / level_list_item for level_list_item in levels_config.levels]
    reward_thresholds = levels_config.reward_thresholds
    levels_shortname = levels_config.shortname
    logger.info("============================================================")
    logger.info(f"({levels_shortname}) Training on levels: {level_list} with reward thresholds: {reward_thresholds}")
    logger.info("============================================================")

    level_path = project_path / level_list[0]
    observation_parser_path = project_path / observation_parser
    rewarder_path = project_path / rewarder
    action_transformer_path = project_path / action_transformer
    allowed_model_types = {"ppo": PPO, "sac": SAC}
    assert model_type in allowed_model_types.keys(), f"Model type must be one of {allowed_model_types}"
    model_class = allowed_model_types[model_type]

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

    train_name = f"{model_type}_" + "_".join(
        [
            levels_shortname,
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
                "level": levels,
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
        threshold=reward_thresholds[0],
    )
    eval_callback = EvalCallback(
        eval_env,
        callback_after_eval=increase_difficulty,
        best_model_save_path=best_model_save_path,
        log_path="./logs/",
        eval_freq=eval_freq,
        deterministic=True,
    )

    try:
        for level_id in range(len(level_list)):
            model.learn(
                total_timesteps=num_timesteps,
                progress_bar=True,
                tb_log_name=train_name,
                callback=eval_callback,
            )

            if level_id < len(level_list) - 1:
                level_path = project_path / level_list[level_id + 1]
                reward_threshold = reward_thresholds[level_id + 1]

                model.set_env(
                    create_race_env(
                        level_path=level_path,
                        observation_parser_path=observation_parser_path,
                        rewarder_path=rewarder_path,
                        action_transformer_path=action_transformer_path,
                        gui=gui,
                        seed=seed,
                        terminate_on_lap=terminate_on_lap,
                    )
                )


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
                    threshold=reward_threshold,
                )
                eval_callback = EvalCallback(
                    eval_env,
                    callback_after_eval=increase_difficulty,
                    best_model_save_path=best_model_save_path,
                    log_path="./logs/",
                    eval_freq=eval_freq,
                    deterministic=True,
                )

            else:
                logger.info("============================================================")
                logger.info("Finished training on all levels.")
                logger.info("============================================================")

    except KeyboardInterrupt:
        logger.info("Training interrupted. Saving model.")

    model.save(f"models/{train_name}/{train_name}")


if __name__ == "__main__":
    fire.Fire(main)
