import sys, os
sys.path.append(os.getcwd() + '/src')
sys.path.append(os.getcwd() + '/src/envs/ant_maze')

import argparse
import torch.nn as nn
import numpy as np
import random

from envs.gridworld_continuous import GridWorldContinuous
from envs.multigrid import MultiGrid
from envs.discretizer import Discretizer
from maze_env import MazeEnv
from datetime import datetime
from algorithms.memento import memento
from policy import GaussianPolicy, train_supervised


parser = argparse.ArgumentParser(description='MEMENTO')

parser.add_argument('--num_workers', type=int, default=1,
                    help='How many parallel workers to use when collecting env trajectories and compute k-nn')
parser.add_argument('--env', type=str, required=True,
                    help='The MDP')
parser.add_argument('--sampling_dist', nargs='+', type=float, default=None,
                    help='The probability distribution with which the different environments are sampled')
parser.add_argument('--batch_dimension', type=int, default=1,
                    help='The number of trajectories on which compute the estimate of the entropy')
parser.add_argument('--use_percentile', type=int, default=0, choices=[0, 1],
                    help='Whether to update the policy using only the batches in the percentile or not')
parser.add_argument('--percentile', type=int, default=None,
                    help='The number of batches with the lowest entropies used to perform the policy update (or log)')
parser.add_argument('--baseline', type=int, default=0, choices=[0, 1],
                    help='Whether to use a baseline to make the estimation consistent')
parser.add_argument('--state_filter', nargs='+', type=int, default=None,
                    help='The subset of state indices returned by the environment on which to maximize entropy')
parser.add_argument('--zero_mean_start', type=int, default=0, choices=[0, 1],
                    help='Whether to make the policy start from a zero mean output')
parser.add_argument('--k', type=int, required=True,
                    help='The number of neighbors')
parser.add_argument('--kl_threshold', type=float, required=True,
                    help='The threshold after which the behavioral policy is updated')
parser.add_argument('--max_off_iters', type=int, metavar='max_off_iter', default=30,
                    help='The maximum number of off policy optimization steps')
parser.add_argument('--use_backtracking', type=int, default=1, choices=[0, 1],
                    help='Whether to use backtracking or not')
parser.add_argument('--backtrack_coeff', type=float, default=2,
                    help='Backtrack coefficient')
parser.add_argument('--max_backtrack_try', type=int, default=10,
                    help='Maximum number of backtracking try')
parser.add_argument('--learning_rate', type=float, required=True,
                    help='The learning rate')
parser.add_argument('--num_trajectories', type=int, required=True,
                    help='The batch of trajectories used in off policy optimization')
parser.add_argument('--trajectory_length', type=int, required=True,
                    help='The maximum length of each trajectory in the batch of trajectories used in off policy optimization')
parser.add_argument('--num_epochs', type=int, required=True,
                    help='The number of epochs(the total number of gradient updates)')
parser.add_argument('--optimizer', type=str, default='adam', choices=['rmsprop', 'adam'],
                    help='The optimizer to use')
parser.add_argument('--heatmap_every', type=int, default=10,
                    help='How many epochs to save a heatmap(if discretizer is defined).'
                         'Also the frequency at which policy weights are saved'
                         'Also the frequency at which full entropy is computed')
parser.add_argument('--heatmap_episodes', type=int, required=True,
                    help='The number of episodes on which the policy is run to compute the heatmap')
parser.add_argument('--heatmap_num_steps', type=int, required=True,
                    help='The number of steps per episode on which the policy is run to compute the heatmap')
parser.add_argument('--full_entropy_k', type=int, required=True,
                    help='The number of neighbors used to compute the full entropy')
parser.add_argument('--seed', type=int, default=None,
                    help='The random seed')
parser.add_argument('--tb_dir_name', type=str, default='memento',
                    help='The tensorboard directory under which the directory of this experiment is put')

args = parser.parse_args()

"""
Experiments specifications

    - env_create : callable that returns the target environment
    - discretizer_create : callable that returns a discretizer for the environment
    - hidden_sizes : hidden layer sizes
    - activation : activation function used in the hidden layers
    - log_std_init : log_std initialization for GaussianPolicy
    - state_filter : list of indices representing the set of features over which entropy is maximized
    - eps : epsilon factor to deal with knn aliasing

"""
exp_spec = {
    'GridWorld': {
        'env_create': lambda: GridWorldContinuous(),
        'discretizer_create': lambda env: Discretizer([[-env.dim, env.dim], [-env.dim, env.dim]], [20, 20]),
        'hidden_sizes': [300, 300],
        'activation': nn.ReLU,
        'log_std_init': -1.5,
        'eps': 0,
        'heatmap_interp': None,
        'heatmap_cmap': 'Blues',
        'heatmap_labels': ('Y', 'X'),
        'zero_mean_train_steps': 50,
        'batch_size': 5000
    },

    'MultiGrid': {
        'env_create': lambda: MultiGrid(),
        'discretizer_create': lambda env: Discretizer([[-env.dim, env.dim], [-env.dim, env.dim]], [20, 20]),
        'hidden_sizes': [300, 300],
        'activation': nn.ReLU,
        'log_std_init': -1.5,
        'eps': 0,
        'heatmap_interp': None,
        'heatmap_cmap': 'Blues',
        'heatmap_labels': ('Y', 'X'),
        'zero_mean_train_steps': 50,
        'batch_size': 5000
    },

    'AntMaze': {
        'env_create': lambda: MazeEnv(),
        'discretizer_create': lambda env: None,
        'hidden_sizes': [400, 300],
        'activation': nn.ReLU,
        'log_std_init': -0.5,
        'eps': 0,
        'state_filter': [0, 1]
    },
}

spec = exp_spec.get(args.env)

if spec is None:
    print("Experiment name not found. Available ones are: %s", ''.join(key for key in exp_spec))

env = spec['env_create']()
discretizer = spec['discretizer_create'](env)
state_filter = spec.get('state_filter')
eps = spec['eps']

zero_mean_train_steps = spec.get('zero_mean_train_steps')
batch_size = spec.get('batch_size')
if zero_mean_train_steps is None:
    zero_mean_train_steps = 100
    batch_size = 5000

def create_policy(is_behavioral=False, seed=np.random.randint(2**16-1)):
    
    policy = GaussianPolicy(
        num_features=env.num_features,
        hidden_sizes=spec['hidden_sizes'],
        action_dim=env.action_space.shape[0],
        activation=spec['activation'],
        log_std_init=spec['log_std_init'],
        seed=seed
    )

    if is_behavioral and args.zero_mean_start:
        env.seed(seed)
        policy = train_supervised(env, policy, train_steps=zero_mean_train_steps, batch_size=batch_size)

    return policy

out_path = os.path.join(os.path.dirname(__file__), "..", "..", "results", args.tb_dir_name, args.env + "_seed_" + str(args.seed) + "_" + datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
os.makedirs(out_path, exist_ok=True)

with open(os.path.join(out_path, 'log_info.txt'), 'w') as f:
    f.write("Run info:\n")
    f.write("-"*10 + "\n")

    for key, value in vars(args).items():
        f.write("{}={}\n".format(key, value))

    f.write("-"*10 + "\n")

    tmp_policy = create_policy()
    f.write(tmp_policy.__str__())
    f.write("\n")

    if args.seed is None:
        args.seed = np.random.randint(2**16-1)
        f.write("Setting random seed {}\n".format(args.seed))

memento(
    env=env,
    env_name=args.env,
    sampling_dist=args.sampling_dist,
    state_filter=state_filter,
    create_policy=create_policy,
    batch_dimension=args.batch_dimension,
    use_percentile=args.use_percentile,
    percentile=args.percentile,
    baseline=args.baseline,
    k=args.k,
    kl_threshold=args.kl_threshold,
    max_off_iters=args.max_off_iters,
    use_backtracking=args.use_backtracking,
    backtrack_coeff=args.backtrack_coeff,
    max_backtrack_try=args.max_backtrack_try,
    eps=eps,
    learning_rate=args.learning_rate,
    num_trajectories=args.num_trajectories,
    trajectory_length=args.trajectory_length,
    num_epochs=args.num_epochs,
    optimizer=args.optimizer,
    full_entropy_k=args.full_entropy_k,
    heatmap_every=args.heatmap_every,
    heatmap_discretizer=discretizer,
    heatmap_episodes=args.heatmap_episodes,
    heatmap_num_steps=args.heatmap_num_steps,
    heatmap_cmap=spec.get('heatmap_cmap'),
    heatmap_labels=spec.get('heatmap_labels'),
    heatmap_interp=spec.get('heatmap_interp'),
    seed=args.seed,
    out_path=out_path,
    num_workers=args.num_workers
)