# ALPHAMEPOL
This repository contains the implementation of the **ALPHAMEPOL** algorithm.

## Installation
In order to use this codebase you need to work with a Python version >= 3.6. Moreover, you need to have a working setup of Mujoco with a valid Mujco license. To setup Mujoco, have a look [here](http://www.mujoco.org/).
To avoid any conflict with your existing Python setup, and to keep this project self-contained, it is suggested to work in a virtual environment with virtualenv. To install virtualenv:
```bash
pip install --upgrade virtualenv
```
Create a virtual environment, activate it and install the requirements:
```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Unsupervised Pre-Training
To reproduce the *Unsupervised Pre-Training* experiments in the paper, run:
```bash
./scripts/exploration/[gridworld_with_slope.sh | multigrid.sh | ant.sh | minigrid.sh]
```

### Supervised Fine-Tuning
To reproduce the *Supervised Fine-Tuning* experiments, run:
```bash
./scripts/goal_rl/[gridworld_with_slope.sh | multigrid.sh | ant.sh | minigrid.sh]
```
By default, this will launch TRPO with ALPHAMEPOL initialization. To launch TRPO with a random initialization, simply omit the *policy_init* argument in the scripts.

Moreover, note that the scripts for the *GridWorld with Slope* and *MultiGrid* experiments have the argument ```num_goals = 50```, meaning that the training will be performed with one goal at a time. If you want to speed up the process, you can use several processes (ideally one for each goal), by passing as argument ```num_goals = 1``` and changing incrementally the seed. As regards the *Ant* and *MiniGrid* experiments, since the goals are predefined, you can also set the ```goal_index``` argument to specify a goal (from 0 to 7 and from 0 to 12 respectively).

## Results Visualization
Once launched, each experiment will log statistics in the *results* folder. You can visualize everything by launching tensorboard targeting that directory:
```bash
python -m tensorboard.main --logdir=./results --port 8080
```
and visiting the board at [http://localhost:8080](http://localhost:8080).