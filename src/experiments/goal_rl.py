import sys, os
sys.path.append(os.getcwd() + '/src')
sys.path.append(os.getcwd() + '/src/envs/ant_maze')
import argparse
import torch
import torch.nn as nn
import numpy as np
import gym

from datetime import datetime
from envs.wrappers import CustomRewardEnv
from envs.gridworld_continuous import GridWorldContinuous
from envs.multigrid import MultiGrid
from maze_env import MazeEnv
import envs.minigrid
from gym_minigrid.wrappers import ImgObsWrapper
from algorithms.trpo import trpo
from policy import GaussianPolicy, Encoder, ValueFunction


parser = argparse.ArgumentParser(description='Supervised Fine-Tuning - TRPO')

parser.add_argument('--num_workers', type=int, default=1,
                    help='How many parallel workers to use when collecting samples')
parser.add_argument('--env', type=str, required=True,
                    help='The MDP')
parser.add_argument('--config', type=int, default=0,
                    help='The index indicating the configuration (environment) on which the training will be done')
parser.add_argument('--policy_init', type=str, default=None,
                    help='Path to the weights for custom policy initialization.')
parser.add_argument('--num_epochs', type=int, required=True,
                    help='The number of training epochs')
parser.add_argument('--num_goals', type=int, default=1,
                    help='The number of the goal positions that will be randomly sampled (used only with GridWorld)')
parser.add_argument('--goal_index', type=int, default=None,
                    help='The index of the goal to be used among the ones pre-specified in the env. config.')
parser.add_argument('--batch_size', type=int, required=True,
                    help='The batch size')
parser.add_argument('--traj_len', type=int, required=True,
                    help='The maximum length of a trajectory')
parser.add_argument('--gamma', type=float, default=0.995,
                    help='The discount factor')
parser.add_argument('--lambd', type=float, default=0.98,
                    help='The GAE lambda')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='The optimizer used for the critic, either adam or lbfgs')
parser.add_argument('--critic_lr', type=float, default=1e-2,
                    help='Learning rate for critic optimization')
parser.add_argument('--critic_reg', type=float, default=1e-3,
                    help='Regularization coefficient for critic optimization')
parser.add_argument('--critic_iters', type=int, default=5,
                    help='Number of critic full updates')
parser.add_argument('--critic_batch_size', type=int, default=64,
                    help='Mini batch in case of adam optimizer for critic optimization')
parser.add_argument('--cg_iters', type=int, default=10,
                    help='Conjugate gradient iterations')
parser.add_argument('--cg_damping', type=float, default=0.1,
                    help='Conjugate gradient damping factor')
parser.add_argument('--kl_thresh', type=float, required=True,
                    help='KL threshold')
parser.add_argument('--seed', type=int, default=None,
                    help='The random seed')
parser.add_argument('--tb_dir_name', type=str, default='goal_rl',
                    help='The tensorboard directory under which the directory of this experiment is put')

args = parser.parse_args()

"""
Sparse reward functions
"""
def grid_goal1(s, r, d, i, goal_position):
    if np.linalg.norm(s - goal_position) <= 1e-1:
        return 1, True
    else:
        return 0, False

def grid_goal2(s, r, d, i, goal_position):
    if np.linalg.norm(s - goal_position) <= 1e-1:
        return 1, True
    else:
        return 0, False

def grid_multi_goal(s, r, d, i, goal_position):
    if np.linalg.norm(s - goal_position) <= 1e-1:
        return 1, True
    else:
        return 0, False

def antmaze_multi_goal(s, r, d, i, goal_position):
    if s[0] >= goal_position[0]:
        return 1, True
    else:
        return 0, False

"""
Experiments specifications
    - env_create : callable that returns the target environment
    - hidden_sizes : hidden layer sizes
    - activation : activation function used in the hidden layers
    - log_std_init : log_std initialization for GaussianPolicy
    - goal_positions: if not specified, <num_goals> random goal positions will be sampled
"""
exp_spec = {
    'GridGoal1': {
        'env_create': lambda: CustomRewardEnv(GridWorldContinuous(configuration=1), grid_goal1),
        'hidden_sizes': [300, 300],
        'activation': nn.ReLU,
        'log_std_init': -1.5,
        'goal_positions': np.array([0., 0.5], dtype=np.float32),
    },

    'GridGoal2': {
        'env_create': lambda: CustomRewardEnv(GridWorldContinuous(configuration=1), grid_goal2),
        'hidden_sizes': [300, 300],
        'activation': nn.ReLU,
        'log_std_init': -1.5,
        'goal_positions': np.array([0.2, -0.9], dtype=np.float32),
    },

    'GridSlope': {
        'env_create': lambda: CustomRewardEnv(GridWorldContinuous(configuration=args.config), grid_multi_goal),
        'hidden_sizes': [300, 300],
        'activation': nn.ReLU,
        'log_std_init': -1.5,
        'goal_positions': None,
    },

    'MultiGrid': {
        'env_create': lambda: CustomRewardEnv(MultiGrid(configuration=args.config), grid_multi_goal),
        'hidden_sizes': [300, 300],
        'activation': nn.ReLU,
        'log_std_init': -1.5,
        'goal_positions': None,
    },

    'AntMaze': {
        'env_create': lambda: CustomRewardEnv(MazeEnv(structure_id=args.config), antmaze_multi_goal),
        'hidden_sizes': [400, 300],
        'activation': nn.ReLU,
        'log_std_init': -0.5,
        'goal_positions': np.array([[2.75, 0], [3, 0], [3.25, 0], [3.5, 0], [3.75, 0], [4, 0], [4.25, 0], [4.5, 0]], dtype=np.float32), # Only adverse config.
    },

    'MiniGrid': {
        'env_create': lambda: ImgObsWrapper(gym.make('MiniGrid-CustomSimpleCrossing-18x18-v0')) if args.config == 0 else ImgObsWrapper(gym.make('MiniGrid-CustomDoor-10x10-v0')),
        'goal_positions': np.array([[[2, 1], [4, 3], [9, 3], [14, 3], [12, 5], [9, 6], [4, 8], [14, 8], [7, 10], [16, 10], [4, 13], [9, 13], [14, 13]], [[6, 8], [8, 1], [4, 3], [8, 4], [4, 6], [7, 6], [3, 8], [5, 1], [6, 3], [3, 4], [7, 4], [5, 5], [5, 7]]], dtype=np.int32)[args.config],
    }
}

spec = exp_spec.get(args.env)

if spec is None:
    print("Experiment name not found. Available ones are: {}".format(', '.join(key for key in exp_spec)))
    exit()

env = spec['env_create']()

# Create a policy
if args.env == 'MiniGrid':
    policy = Encoder(
        env=env,
        seed=args.seed
    )
else:
    policy = GaussianPolicy(
        num_features=env.num_features,
        hidden_sizes=spec['hidden_sizes'],
        action_dim=env.action_space.shape[0],
        activation=spec['activation'],
        log_std_init=spec['log_std_init'],
        seed=args.seed
    )

# Create a critic
if args.env == 'MiniGrid':
    vfunc = ValueFunction(env, args.seed)
else:
    hidden_sizes = [64, 64]
    hidden_activation = nn.ReLU
    layers = []
    for i in range(len(hidden_sizes)):
        if i == 0:
            layers.extend([
                nn.Linear(env.num_features, hidden_sizes[i]),
                hidden_activation()
            ])
        else:
            layers.extend([
                nn.Linear(hidden_sizes[i-1], hidden_sizes[i]),
                hidden_activation()
            ])

    layers.append(nn.Linear(hidden_sizes[i], 1))
    vfunc = nn.Sequential(*layers)

    for module in vfunc:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)


if args.policy_init is not None:
    kind = 'ALPHAMEPOLInit'
    policy.load_state_dict(torch.load(args.policy_init))
else:
    kind = 'RandomInit'


parent_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", args.tb_dir_name, args.env + "_config_" + str(args.config) + "_" + datetime.now().strftime('%Y_%m_%d_%H_%M'))
out_path = os.path.join(parent_dir, args.env + "_seed_" + str(args.seed) + "_" + kind + "_" + datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
os.makedirs(out_path, exist_ok=True)


with open(os.path.join(out_path, 'log_info.txt'), 'w') as f:
    f.write("Run info:\n")
    f.write("-"*10 + "\n")

    for key, value in vars(args).items():
        f.write("{}={}\n".format(key, value))

    f.write("-"*10 + "\n")

    f.write(policy.__str__())
    f.write("-"*10 + "\n")
    f.write(vfunc.__str__())

    f.write("\n")

    if args.seed is None:
        args.seed = np.random.randint(2**16-1)
        f.write("Setting random seed {}\n".format(args.seed))

trpo(
    env_maker=spec['env_create'],
    env_name=args.env,
    env_config=args.config,
    policy_init=args.policy_init,
    num_epochs=args.num_epochs,
    num_goals=args.num_goals,
    goal_index=args.goal_index,
    goal_positions=spec['goal_positions'],
    batch_size=args.batch_size,
    traj_len=args.traj_len,
    gamma=args.gamma,
    lambd=args.lambd,
    vfunc=vfunc,
    policy=policy,
    optimizer=args.optimizer,
    critic_lr=args.critic_lr,
    critic_reg=args.critic_reg,
    critic_iters=args.critic_iters,
    critic_batch_size=args.critic_batch_size,
    cg_iters=args.cg_iters,
    cg_damping=args.cg_damping,
    kl_thresh=args.kl_thresh,
    num_workers=args.num_workers,
    out_path=out_path,
    seed=args.seed
)