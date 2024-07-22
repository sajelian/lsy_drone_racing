# Autonomous Drone Racing Project

https://github.com/user-attachments/assets/c573da5e-5e20-4102-ab30-57eaec114081

Check the original repository for further information.

[💾 Install](docs/INSTALL.md)

[👨‍💻 Code structure](docs/STRUCTURE.md)


## Difficulty levels
The complete problem is specified by a YAML file, e.g. [`getting_started.yaml`](config/getting_started.yaml)

The config folder contains settings for progressively harder scenarios:

|         Evaluation Scenario         | Constraints | Rand. Inertial Properties | Randomized Obstacles, Gates | Rand. Between Episodes |         Notes         |
| :---------------------------------: | :---------: | :-----------------------: | :-------------------------: | :--------------------: | :-------------------: |
| [`level0.yaml`](config/level0.yaml) |   **Yes**   |           *No*            |            *No*             |          *No*          |   Perfect knowledge   |
| [`level1.yaml`](config/level1.yaml) |   **Yes**   |          **Yes**          |            *No*             |          *No*          |       Adaptive        |
| [`level2.yaml`](config/level2.yaml) |   **Yes**   |          **Yes**          |           **Yes**           |          *No*          | Learning, re-planning |
| [`level3.yaml`](config/level3.yaml) |   **Yes**   |          **Yes**          |           **Yes**           |        **Yes**         |      Robustness       |
|                                     |             |                           |                             |                        |                       |
|              sim2real               |   **Yes**   |    Real-life hardware     |           **Yes**           |          *No*          |   Sim2real transfer   |

> **Note:** "Rand. Between Episodes" (governed by argument `reseed_on_reset`) states whether randomized properties and positions vary or are kept constant (by re-seeding the random number generator on each `env.reset()`) across episodes

