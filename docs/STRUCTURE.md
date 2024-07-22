## Training
- `controller/ppo/ppo_deploy.py`: contains a state machine triggering ppo after take off and stopping it after passing the last gate. Used by the `ppo.py` controller.
- `controller/ppo/ppo.py`: used directly by `sim.py` to control the drone. 

## Environment modifiers

The most important files are:

- `env_modifiers/observation_parser.py`: includes the `ObservationParser` base class and all the child classes based on it. There are a loooot of them, some need to be properly documented.
- `env_modifiers/action_transformer.py`: includes the `ActionTransformer` base class and all the child classes based on it.
- `env_modifiers/rewarder.py`: contains the rewarder `Rewarder`.

## Models
Models can be found under `models`. The name encodes the modules used for training. In the `params.yaml` of a model more details can be found.

## Configs

In configs, you can find subfolders containing all configurations ever tried (too many...) some of the levels required the modified version of
safe-control-gym to work, since it was modified to spawn drones at different locations.

## Other files

- `level_generator.py`: attempt to create levels on the run for curriculum training...abandoned.
- `speed_estimation.py`: used to create different speed estimation models. Not really used due to time constraints.
- `train_multiple.py`: used to train in a curriculum manner with a list of environments... not really used.

