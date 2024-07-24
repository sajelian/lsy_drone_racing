## Installation

To run the LSY Autonomous Drone Racing project, you will need 3 main repositories:
- [safe-control-gym](https://github.com/utiasDSL/safe-control-gym/tree/lsy_drone_racing) - `lsy_drone_racing` branch: The drone simulator and gym environments

_UPDATE:_ use https://github.com/danielsanjosepro/safe-control-gym fork for levels that have array init states in the configuration, so that you can start drones at different locations.

- [pycffirmware](https://github.com/utiasDSL/pycffirmware) - `main` branch: A simulator for the on-board controller response of the drones we are using to accurately model their behavior
- [lsy_drone_racing](https://github.com/utiasDSL/lsy_drone_racing) - `main` branch: This repository contains the scripts to simulate and deploy the drones in the racing challenge

### Fork lsy_drone_racing

The first step is to fork the [lsy_drone_racing](https://github.com/utiasDSL/lsy_drone_racing) repository for your own group. This has two purposes: You automatically have your own repository with git version control, and it sets you up for taking part in the online competition and automated testing (see [competition](#the-online-competition)).

If you have never worked with GitHub before, see the [docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) on forking.

### Using conda/mamba

The following assumes that you have a functional installation of either [conda](https://conda.io/projects/conda/en/latest/index.html) or [mamba](https://mamba.readthedocs.io/en/latest/).

First, clone the new fork from your own account and create a new environment with Python 3.8 by running

```bash
mkdir -p ~/repos && cd repos
git clone https://github.com/<YOUR-USERNAME>/lsy_drone_racing.git
conda create -n drone python=3.8
conda activate drone
```

> **Note:** It is important you stick with **Python 3.8**. Yes, it is outdated. Yes, we'd also like to upgrade. However, there are serious issues beyond our control when deploying the code on the real drones with any other version.

Next, download the `safe-control-gym` and `pycffirmware` repositories and install them. Make sure you have your conda/mamba environment active!

```bash
cd ~/repos
# git clone -b lsy_drone_racing https://github.com/utiasDSL/safe-control-gym.git
git clone -b lsy_drone_racing https://github.com/danielsanjosepro/safe-control-gym.git
cd safe-control-gym
pip install .
```

> **Note:** If you receive an error installing safe-control-gym related to gym==0.21.0, run
> ```bash
>    pip install setuptools==65.5.0 pip==21 wheel==0.38.4
> ```
> first

```bash
cd ~/repos
git clone https://github.com/utiasDSL/pycffirmware.git
cd pycffirmware
git submodule update --init --recursive
sudo apt update
sudo apt install build-essential
conda install swig
./wrapper/build_linux.sh
```

Now you can install the lsy_drone_racing package in editable mode from the repository root

```bash
cd ~/repos/lsy_drone_racing
pip install --upgrade pip
pip install -e .
```

Finally, you can test if the installation was successful by running 

```bash
cd ~/repos/lsy_drone_racing
python scripts/sim.py
```

If everything is installed correctly, this opens the simulator and simulates a drone flying through four gates.
