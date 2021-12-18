import sys, os
sys.path.append(os.getcwd() + '/src')

import numpy as np
import collections
import random
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import scipy
import scipy.stats
import scipy.special
import time
import io
from tabulate import tabulate

import gym
import envs.minigrid
from gym_minigrid.wrappers import ImgObsWrapper

from sklearn.neighbors import NearestNeighbors
from torch.utils import tensorboard
from utils.dtypes import float_type, int_type



def get_heatmap(training_envs, sampling_dist, policy, discretizer, num_episodes, num_steps, batch_dimension, cmap, interp, labels):

    if training_envs is not None:
        env = training_envs[0]
        if env.__class__.__name__ == 'ImgObsWrapper':
            positions = []
            env = training_envs[0]
            pos = np.zeros((env.width, env.height))
            for _ in range(num_episodes):
                s = env.reset()
                for t in range(num_steps):
                    a = policy.predict(s).numpy()
                    s, _, done, _ = env.step(a)
                    pos[env.front_pos[0], env.front_pos[1]] += 1
            pos = pos / num_episodes
            positions.append(pos)

            env = training_envs[1]
            pos = np.zeros((env.width, env.height))
            for _ in range(num_episodes):
                s = env.reset()
                for t in range(num_steps):
                    a = policy.predict(s).numpy()
                    s, _, done, _ = env.step(a)
                    pos[env.front_pos[0], env.front_pos[1]] += 1
            pos = pos / num_episodes
            positions.append(pos)

            plt.close()
            minigrid_heatmap = plt.figure(figsize=plt.figaspect(0.5))
            plot_titles = ['SimpleCrossing', 'Door']
            for j in range(len(positions)):
                ax = minigrid_heatmap.add_subplot(1, 2, j+1)
                ax.set_title(plot_titles[j])
                sns.heatmap(positions[j], ax=ax, cmap='Blues', annot=True, fmt=".1f")
            return None, 0, minigrid_heatmap
        else:
            raise("Error getting the heatmap: the environment has not been correctly specified.")


def collect_particles_and_compute_knn(training_envs, sampling_dist, policy, random_encoder, num_trajectories, trajectory_length, num_batches, batch_dimension, state_filter, k, num_workers):
    env = training_envs[0]
    random_feats = torch.zeros((num_trajectories, trajectory_length + 1, env.num_features), dtype=float_type)
    states = torch.zeros((num_trajectories, trajectory_length + 1, 7, 7, 3), dtype=float_type)
    actions = torch.zeros((num_trajectories, trajectory_length, 1), dtype=torch.int32)
    effective_trajectory_lengths = torch.zeros((num_trajectories, 1), dtype=torch.int32)
    distances = torch.zeros((num_batches, batch_dimension*trajectory_length, k+1), dtype=float_type)
    indices = torch.zeros((num_batches, batch_dimension*trajectory_length, k+1), dtype=torch.long)
    sampled_env_batched = torch.zeros((num_batches), dtype=torch.int32)

    # Noise param.
    lower, upper = 0, 0.001
    mu, sigma = 0.001, 0.001
    
    for trajectory in range(num_trajectories):
        
        if (trajectory % batch_dimension == 0) or (num_batches == 1):
            
            if len(training_envs) > 1: # Multiple environments
                sampled_config = np.random.choice(np.arange(0,len(training_envs)), p=sampling_dist)
                env = training_envs[sampled_config]
                index = 0 if trajectory == 0 else trajectory // batch_dimension
                sampled_env_batched[index] = sampled_config

        s = env.reset()

        obs = torch.tensor(s, dtype=float_type)
        obs = obs.unsqueeze(0).transpose(1, 3).transpose(2, 3)
        random_feat = random_encoder(obs)[0,:,0,0].detach().numpy()
        # Add some noise to avoid aliasing
        random_noise = scipy.stats.truncnorm.rvs((lower-mu) / sigma, (upper-mu) / sigma, loc=mu, scale=sigma, size=random_feat.shape[0])
        random_feat = random_feat + random_noise
        #print(random_feat.shape) # (64,)

        for t in range(trajectory_length):
            states[trajectory, t] = torch.from_numpy(s)
            random_feats[trajectory, t] = torch.from_numpy(random_feat)

            a = policy.predict(s).numpy()

            actions[trajectory, t] = torch.from_numpy(a)

            new_s, _, done, _ = env.step(a)

            s = new_s

            obs = torch.tensor(s, dtype=float_type)
            obs = obs.unsqueeze(0).transpose(1, 3).transpose(2, 3)
            random_feat = random_encoder(obs)[0,:,0,0].detach().numpy()
            # Add some noise to avoid aliasing
            random_noise = scipy.stats.truncnorm.rvs((lower-mu) / sigma, (upper-mu) / sigma, loc=mu, scale=sigma, size=random_feat.shape[0])
            random_feat = random_feat + random_noise

            """ if done:
                break """

        random_feats[trajectory, t] = torch.from_numpy(random_feat)
        states[trajectory, t+1] = torch.from_numpy(s)
        effective_trajectory_lengths[trajectory] = t+1

    start_index = -batch_dimension
    end_index = 0
    for batch in range(num_batches):
        next_states = None
        start_index += batch_dimension
        end_index += batch_dimension
        traj_real_len = effective_trajectory_lengths[start_index].item()
        current_batch_states = random_feats[start_index:end_index, :, :]
        for traj_batch in range(batch_dimension):
            traj_next_states = current_batch_states[traj_batch, 1:traj_real_len+1, :].reshape(-1, env.num_features)
            if next_states is None:
                next_states = traj_next_states
            else:
                next_states = torch.cat([next_states, traj_next_states], dim=0)
        
        if state_filter is not None:
            # Filter particle features over which entropy is maximized
            next_states = next_states[:, state_filter]

        # Fit knn for the batch of collected particles
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean', algorithm='auto', n_jobs=num_workers)
        nbrs.fit(next_states)
        distances[batch], indices[batch] = torch.tensor(nbrs.kneighbors(next_states))

    return states, actions, effective_trajectory_lengths, distances, indices, sampled_env_batched


def compute_importance_weights(behavioral_policy, target_policy, states, actions, batch_dimension, trajectory_lengths):
    # Initialize to None for the first concat
    importance_weights = None

    # Compute the importance weights
    # build iw vector incrementally from trajectory particles
    for trajectory in range(batch_dimension):
        trajectory_length = trajectory_lengths[trajectory][0].item()

        traj_states = states[trajectory, :trajectory_length]
        traj_actions = actions[trajectory, :trajectory_length]

        traj_particle_target_log_p = target_policy.get_log_p(traj_states, traj_actions)
        traj_particle_behavior_log_p = behavioral_policy.get_log_p(traj_states, traj_actions)

        traj_particle_iw = torch.exp(torch.cumsum(traj_particle_target_log_p - traj_particle_behavior_log_p, dim=0))

        if importance_weights is None:
            importance_weights = traj_particle_iw
        else:
            importance_weights = torch.cat([importance_weights, traj_particle_iw], dim=0)

    # Normalize the weights
    importance_weights = importance_weights / torch.sum(importance_weights)
    return importance_weights


def compute_entropy(behavioral_policy, target_policy, states, actions, num_batches, batch_dimension, use_percentile, percentile, percentile_indices, baseline, trajectory_lengths, distances, indices, biased_full_entropy, batch_entropies, is_policy_update, k, G, B, ns, eps):
    entropies = torch.zeros((num_batches), dtype=float_type)
    entropies_with_no_grad = torch.zeros((num_batches), dtype=float_type, requires_grad=False)
    if use_percentile:
        percentile_entropies = torch.zeros((percentile), dtype=float_type)
        percentile_entropies_with_no_grad = torch.zeros((percentile), dtype=float_type, requires_grad=False)
    start_index = -batch_dimension
    end_index = 0
    for batch in range(num_batches):
        start_index += batch_dimension
        end_index += batch_dimension
        importance_weights = compute_importance_weights(behavioral_policy, target_policy, states[start_index:end_index], actions[start_index:end_index], batch_dimension, trajectory_lengths)
        # Compute objective function
        # compute weights sum for each particle
        weights_sum = torch.sum(importance_weights[indices[batch, :, :-1]], dim=1)
        # compute volume for each particle
        volumes = (torch.pow(distances[batch, :, k], ns) * torch.pow(torch.tensor(np.pi), ns/2)) / G
        # compute entropy
        n_sample = batch_dimension * trajectory_lengths.size()[0]
        entropies_with_no_grad[batch] = - (1 / n_sample) * torch.sum(torch.log((k / (n_sample*volumes + eps)) + eps)) + B
        entropies[batch] = - torch.sum((weights_sum / k) * torch.log((weights_sum / (volumes + eps)) + eps)) + B

    if (num_batches != 1) and not is_policy_update:
        with torch.no_grad():
            # Compute the biased full entropy, i.e. the mean entropy of all the batches (for log purposes only)
            biased_full_entropy.append(torch.mean(entropies).item())
            # Save the entropies of the mini-batches (for log purposes only)
            batch_entropies.append(entropies.tolist())
    
    if (use_percentile) and (num_batches != 1):

        ALPHA = percentile / num_batches
        
        sorted_entropies = torch.sort(entropies).values
        sorted_entropies_with_no_grad = torch.sort(entropies_with_no_grad).values

        with torch.no_grad():
            # Get the indices in order to retrieve the batches of the sampled environments
            percentile_indices.append(list(np.array(torch.sort(entropies).indices[:percentile])))
            
        percentile_entropies = sorted_entropies[:percentile]
        percentile_entropies_with_no_grad = sorted_entropies_with_no_grad[:percentile]
    
        # Add the baseline, if required
        if baseline and is_policy_update:
            # VaR baseline:
            empirical_var = sorted_entropies_with_no_grad[percentile]
            return percentile_entropies * (1 - empirical_var)

        return percentile_entropies

    if (not use_percentile) and (num_batches != 1):
        # Perform bootstrapping
        random_indexes = np.random.choice(range(num_batches), percentile, replace=False)
        return entropies[random_indexes]
        
    return entropies


def compute_kl(behavioral_policy, target_policy, states, actions, num_trajectories, trajectory_lengths, indices, k, eps):
    importance_weights = compute_importance_weights(behavioral_policy, target_policy, states, actions, num_trajectories, trajectory_lengths)

    weights_sum = torch.sum(importance_weights[indices[:, :-1]], dim=1)

    # Compute KL divergence between behavioral and target policy
    N = importance_weights.shape[0]
    kl = (1 / N) * torch.sum(torch.log(k / (N * weights_sum) + eps))

    numeric_error = torch.isinf(kl) or torch.isnan(kl)

    # Minimum KL is zero
    # NOTE: do not remove epsilon factor
    kl = torch.max(torch.tensor(0.0), kl)

    return kl, numeric_error


def log_epoch_statistics(writer, log_file, log_entropies, log_gradients, csv_file_1, csv_file_2, epoch,
                         loss, entropy, batch_entropies, sampled_env, grads, num_off_iters, execution_time, full_entropy, biased_full_entropy, perc_sampled_env,
                         heatmap_image, heatmap_entropy, backtrack_iters, backtrack_lr):
    # Log to Tensorboard
    writer.add_scalar("Loss", loss, global_step=epoch)
    writer.add_scalar("CVaR Entropy", entropy, global_step=epoch)
    writer.add_scalar("Execution time", execution_time, global_step=epoch)
    writer.add_scalar("Number off-policy iteration", num_off_iters, global_step=epoch)
    if biased_full_entropy is not None:
        writer.add_scalar("Biased Full Entropy", biased_full_entropy, global_step=epoch)
    if full_entropy is not None:
        writer.add_scalar(f"Full Entropy:", full_entropy, global_step=epoch)

    if heatmap_image is not None:
        # Log heatmap to tensorboard
        writer.add_figure(f"Heatmap", heatmap_image, global_step=epoch)

    # Log the sampled env. inside the percentile
    if perc_sampled_env != '':
        env_in_perc = perc_sampled_env.split('-')
        env_in_perc = [0 if el == 'None' else int(el) for el in env_in_perc]
        sampled_env_fig, ax = plt.subplots()
        ax.bar([x for x in range(len(env_in_perc))], env_in_perc)
        writer.add_figure(f'Sampled env. in perc.', sampled_env_fig, global_step=epoch)
        plt.clf()

    table = []
    fancy_float = lambda f : f"{f:.3f}"
    table.extend([
        ["Epoch", epoch],
        ["Execution time (s)", fancy_float(execution_time)],
        ["CVaR Entropy", fancy_float(entropy)],
        ["Off-policy iters", num_off_iters]
    ])

    if biased_full_entropy is not None:
        table.extend([
            ["Biased Full Entropy", fancy_float(biased_full_entropy)]
        ])

    if heatmap_image is not None:
        if isinstance(heatmap_entropy, int):
            table.extend([
                ["Heatmap entropy", fancy_float(heatmap_entropy)]
            ])
        elif isinstance(heatmap_entropy, list) and (len(heatmap_entropy) == 2):
            table.extend([
                ["Heatmap entropy (config. 0)", fancy_float(heatmap_entropy[0])],
                ["Heatmap entropy (config. 1)", fancy_float(heatmap_entropy[1])]
            ])

    if backtrack_iters is not None:
        table.extend([
            ["Backtrack iters", backtrack_iters],
            ["Backtrack learning rate: ", fancy_float(backtrack_lr)]
        ])

    fancy_grid = tabulate(table, headers="firstrow", tablefmt="fancy_grid", numalign='right')

    # Log to csv file 1
    csv_file_1.write(f"{epoch},{loss},{entropy},{full_entropy},{num_off_iters},{execution_time},{perc_sampled_env},{biased_full_entropy}\n")
    csv_file_1.flush()

    # Log to csv file 2
    if heatmap_image is not None:
        csv_file_2.write(f"{epoch},{heatmap_entropy}\n")
        csv_file_2.flush()

    # Log to stdout and log file
    log_file.write(fancy_grid)
    log_file.flush()

    # Log entropies to csv file 4
    if batch_entropies is not None:
        writer = csv.writer(log_entropies, delimiter=',', lineterminator='\n',)
        writer.writerow([epoch, '-'.join(map(str, sampled_env.tolist()))] + batch_entropies)
        log_entropies.flush()
        batch_entropies = []

    # Log gradients
    if grads is not None:
        parsed_grads = ' '.join(str(e.item()) for e in grads)
        log_gradients.write(parsed_grads + '\n')
        log_gradients.flush()

    print(fancy_grid)


def log_off_iter_statistics(writer, csv_file_3, epoch, global_off_iter,
                            num_off_iter, entropy, kl, lr):
    # Log to csv file 3 to see what's going on during off policy optimization
    csv_file_3.write(f"{epoch},{num_off_iter},{entropy},{kl},{lr}\n")
    csv_file_3.flush()

    # Also log to tensorboard
    writer.add_scalar("Off policy iter Entropy", entropy, global_step=global_off_iter)
    writer.add_scalar("Off policy iter KL", kl, global_step=global_off_iter)


def get_grads(named_parameters):
    ave_grads = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n) and (p.grad is not None):
            ave_grads.append(p.grad.mean())
    return ave_grads


def policy_update(optimizer, behavioral_policy, target_policy, states, actions, num_batches, batch_dimension, use_percentile, percentile, percentile_indices, baseline, trajectory_lengths, distances, indices, biased_full_entropy, batch_entropies, k, G, B, ns, eps):
    optimizer.zero_grad()

    # Maximize entropy <-> minimize loss
    loss = - torch.mean(compute_entropy(behavioral_policy, target_policy, states, actions, num_batches, batch_dimension, use_percentile, percentile, percentile_indices, baseline, trajectory_lengths, distances, indices, biased_full_entropy, batch_entropies, True, k, G, B, ns, eps))

    numeric_error = torch.isinf(loss) or torch.isnan(loss)

    loss.backward()
    grads = get_grads(target_policy.named_parameters())
    optimizer.step()

    return loss, numeric_error, grads


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


def alphamepol_visual(env, sampling_dist, state_filter, create_policy, batch_dimension, use_percentile, percentile, baseline, k, kl_threshold, max_off_iters,
          use_backtracking, backtrack_coeff, max_backtrack_try, eps,
          learning_rate, num_trajectories, trajectory_length, num_epochs, optimizer, full_entropy_k,
          heatmap_every, heatmap_discretizer, heatmap_episodes, heatmap_num_steps,
          heatmap_cmap, heatmap_interp, heatmap_labels, seed, out_path, num_workers):

    assert num_trajectories % batch_dimension == 0, "Please provide a number of trajectories that can be equally split among batches."
    assert num_trajectories >= batch_dimension, "Please provide a number of trajectories which is greater or equal than the batch dimension."
    num_batches = num_trajectories // batch_dimension
    if percentile is not None:
        assert num_batches >= percentile, "Please check if the number of batches is greater or equal than the required percentile."

    # Indices of the batches corresponding to the percentile
    percentile_indices = []
    # Compute also the biased full entropy, i.e. the mean entropy of all the batches
    biased_full_entropy = []
    # List used to log the entropy of all the mini-batches at each epoch
    batch_entropies = []
    
    training_envs = None

    if (sampling_dist is not None) and (len(sampling_dist) > 1):
        if env.__class__.__name__ == 'ImgObsWrapper': # MiniGrid Wrapper
            training_envs = []
            custom_simplecrossing_env = gym.make('MiniGrid-CustomSimpleCrossing-18x18-v0')
            custom_door_env = gym.make('MiniGrid-CustomDoor-10x10-v0')
            training_envs.append(ImgObsWrapper(custom_simplecrossing_env))
            training_envs.append(ImgObsWrapper(custom_door_env))
            for e in training_envs:
                e.seed(seed)
    elif sampling_dist is None:
        training_envs = []
        env.seed(seed)
        training_envs.append(env)
    elif len(sampling_dist) == 1:
        training_envs = []
        env.seed(seed)
        training_envs.append(env)
        sampling_dist = [1]
    else:
        raise("The specified sampling distribution is not valid.")

    if training_envs is None:
        raise("No environments have been found.")

    if seed is not None:
        # Seed everything
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    if sampling_dist is None:
        # Set a uniform sampling distribution
        sampling_dist = [1/len(training_envs) for _ in training_envs]
    if training_envs is not None:
        if len(sampling_dist) != len(training_envs):
            print("The sampled configuration does not exist. Check the length of the sampling distribution argument.")
            exit()

    # Create writer for tensorboard
    writer = tensorboard.SummaryWriter(out_path)

    # Create a behavioral, a target policy and a tmp policy used to save valid target policies (those with kl <= kl_threshold) during off policy opt
    behavioral_policy = create_policy(is_behavioral=True, seed=seed)
    target_policy = create_policy()
    last_valid_target_policy = create_policy()
    target_policy.load_state_dict(behavioral_policy.state_dict())
    last_valid_target_policy.load_state_dict(behavioral_policy.state_dict())

    # Create the random encoder
    random_encoder = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
    
    for param in random_encoder.parameters():
        param.requires_grad = False
    
    random_encoder.apply(weights_init)

    # Set optimizer
    if optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(target_policy.parameters(), lr=learning_rate)
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(target_policy.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError

    # Fixed constants
    ns = len(state_filter) if (state_filter is not None) else env.num_features
    B = np.log(k) - scipy.special.digamma(k)
    full_B = np.log(full_entropy_k) - scipy.special.digamma(full_entropy_k)
    G = scipy.special.gamma(ns/2 + 1)

    # Create log files
    log_file = open(os.path.join((out_path), 'log_file.txt'), 'a', encoding="utf-8")

    # Create file to log the gradients of the loss
    log_gradients = open(os.path.join((out_path), 'log_gradients.txt'), 'a', encoding="utf-8")

    csv_file_1 = open(os.path.join(out_path, f"{env.__class__.__name__}.csv"), 'w')
    csv_file_1.write(",".join(['epoch', 'loss', 'cvar_entropy', 'full_entropy', 'num_off_iters','execution_time','perc_sampled_env', 'biased_full_entropy']))
    csv_file_1.write("\n")

    if heatmap_discretizer is not None or env.__class__.__name__ == 'MazeEnv' or env.__class__.__name__ == 'ImgObsWrapper':
        # Some kind of 2d discretizer is defined on this environment
        csv_file_2 = open(os.path.join(out_path, f"{env.__class__.__name__}-heatmap.csv"), 'w')
        csv_file_2.write(",".join(['epoch', 'average_entropy']))
        csv_file_2.write("\n")
    else:
        csv_file_2 = None

    csv_file_3 = open(os.path.join(out_path, f"{env.__class__.__name__}_off_policy_iter.csv"), "w")
    csv_file_3.write(",".join(['epoch', 'off_policy_iter', 'cvar_entropy', 'kl', 'learning_rate']))
    csv_file_3.write("\n")

    csv_file_4 = open(os.path.join(out_path, f"{env.__class__.__name__}_entropies.csv"), "w")
    first_part_list = ['epoch', 'configurations']
    list_of_batches = ['batch_' + str(j) for j in range(num_batches)]
    log_entropies_list = first_part_list + list_of_batches
    csv_file_4.write(",".join(log_entropies_list))
    csv_file_4.write("\n")


    # At epoch 0 do not optimize, just log stuff for the initial policy
    epoch = 0
    t0 = time.time()

    # Full entropy
    states, actions, effective_trajectory_lengths, distances, indices, sampled_env_batched = \
        collect_particles_and_compute_knn(training_envs, sampling_dist, behavioral_policy, random_encoder, num_trajectories,
                                          trajectory_length, 1, num_trajectories, state_filter, full_entropy_k, num_workers)

    with torch.no_grad():
        full_entropies = compute_entropy(behavioral_policy, behavioral_policy, states, actions,
                                    1, num_trajectories, 0, num_batches, percentile_indices, baseline, effective_trajectory_lengths,
                                    distances, indices, biased_full_entropy, batch_entropies, False, full_entropy_k, G, full_B, ns, eps)
        full_entropy = full_entropies[0]

    # Indices of the batches corresponding to the percentile
    percentile_indices = []
    # Entropy
    states, actions, effective_trajectory_lengths, distances, indices, sampled_env_batched = \
        collect_particles_and_compute_knn(training_envs, sampling_dist, behavioral_policy, random_encoder, num_trajectories,
                                        trajectory_length, num_batches, batch_dimension, state_filter, k, num_workers)

    with torch.no_grad():
        entropies = compute_entropy(behavioral_policy, behavioral_policy, states, actions,
                                  num_batches, batch_dimension, use_percentile, percentile, percentile_indices, baseline, effective_trajectory_lengths,
                                  distances, indices, biased_full_entropy, batch_entropies, False, k, G, B, ns, eps)
        # CVaR entropy
        if use_percentile:
            entropy = torch.mean(entropies)
        else:
            entropy = torch.mean(torch.sort(entropies).values[:percentile])

    execution_time = time.time() - t0
    
    loss = - entropy

    perc_sampled_env = ""
    if use_percentile:
        # Compute the occurrences of the sampled environments
        percentile_sampled_env_batched = torch.zeros((percentile), dtype=torch.int32)
        for batch in range(percentile):
            percentile_sampled_env_batched[batch] = sampled_env_batched[percentile_indices[0][batch]]

        occurrences = collections.Counter(np.array(percentile_sampled_env_batched))
        perc_sampled_env = ""
        for c in range(len(sampling_dist)):
            perc_sampled_env += str(occurrences.get(c))
            if c < len(sampling_dist)-1:
                perc_sampled_env += "-"

    # Heatmap
    if heatmap_discretizer is not None or env.__class__.__name__ == 'MazeEnv' or env.__class__.__name__ == 'ImgObsWrapper':
        _, heatmap_entropy, heatmap_image = \
            get_heatmap(training_envs, sampling_dist, behavioral_policy, heatmap_discretizer, heatmap_episodes, heatmap_num_steps, batch_dimension, heatmap_cmap, heatmap_interp, heatmap_labels)
    else:
        heatmap_entropy = None
        heatmap_image = None

    # Save initial policy
    torch.save(behavioral_policy.state_dict(), os.path.join(out_path, f"{epoch}-policy"))

    # Log statistics for the initial policy
    log_epoch_statistics(
            writer=writer, log_file=log_file, log_entropies=csv_file_4, log_gradients=log_gradients, csv_file_1=csv_file_1, csv_file_2=csv_file_2,
            epoch=epoch,
            loss=loss,
            entropy=entropy,
            batch_entropies=batch_entropies[-1] if len(batch_entropies) > 0 else None,
            sampled_env=sampled_env_batched,
            grads=None,
            execution_time=execution_time,
            num_off_iters=0,
            full_entropy=full_entropy,
            biased_full_entropy=biased_full_entropy[-1] if len(biased_full_entropy) > 0 else None,
            perc_sampled_env=perc_sampled_env,
            heatmap_image=heatmap_image,
            heatmap_entropy=heatmap_entropy,
            backtrack_iters=None,
            backtrack_lr=None
        )

    # Main Loop
    global_num_off_iters = 0

    if use_backtracking:
        original_lr = learning_rate

    # Variables used to check convergence
    entropies_over_ten_epochs = [entropy]
    old_mean_entropy_over_ten_epochs = 0
    switched_to_obj_pos = False
    transition_counter = 0

    while epoch < num_epochs:
        t0 = time.time()

        # Off policy optimization
        kl_threshold_reached = False
        last_valid_target_policy.load_state_dict(behavioral_policy.state_dict())
        num_off_iters = 0

        # Collect particles to optimize off policy
        states, actions, effective_trajectory_lengths, distances, indices, sampled_env_batched = \
                collect_particles_and_compute_knn(training_envs, sampling_dist, behavioral_policy, random_encoder, num_trajectories,
                                                  trajectory_length, num_batches, batch_dimension, state_filter, k, num_workers)
        
        if use_backtracking:
            learning_rate = original_lr

            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            backtrack_iter = 1
        else:
            backtrack_iter = None
        
        # Indices of the batches corresponding to the percentile
        percentile_indices = []

        while not kl_threshold_reached:
            # Optimize policy
            loss, numeric_error, grads = policy_update(optimizer, behavioral_policy, target_policy, states, actions, num_batches, batch_dimension, use_percentile, percentile, percentile_indices, baseline, effective_trajectory_lengths, distances, indices, biased_full_entropy, batch_entropies, k, G, B, ns, eps)
            entropy = - loss.detach().numpy()

            if not numeric_error:
                with torch.no_grad():
                    kl, kl_numeric_error = compute_kl(behavioral_policy, target_policy, states,
                                                        actions, num_trajectories, effective_trajectory_lengths,
                                                        indices.reshape(num_batches*batch_dimension*trajectory_length, k+1), k, eps)
                kl = kl.numpy()

                if not kl_numeric_error:
                    if kl <= kl_threshold:
                        # Valid update
                        last_valid_target_policy.load_state_dict(target_policy.state_dict())
                        num_off_iters += 1
                        global_num_off_iters += 1

                        # Log statistics for this off policy iteration
                        log_off_iter_statistics(writer, csv_file_3, epoch, global_num_off_iters, num_off_iters - 1, entropy, kl, learning_rate)           
                    else:
                        if use_backtracking:
                            # We are here because we need to perform one last update
                            if not backtrack_iter == max_backtrack_try:
                                target_policy.load_state_dict(last_valid_target_policy.state_dict())

                                learning_rate = original_lr / (backtrack_coeff ** backtrack_iter)

                                for param_group in optimizer.param_groups:
                                    param_group['lr'] = learning_rate

                                backtrack_iter += 1
                                continue
                else:
                    # Do not accept the update, set exit condition to end the epoch
                    kl_threshold_reached = True
            else:
                # Set exit condition to end the epoch
                kl_threshold_reached = True
            
            if use_backtracking and backtrack_iter > 1:
                # Just perform at most 1 step using backtracking
                kl_threshold_reached = True

            if num_off_iters == max_off_iters:
                # Set exit condition also if the maximum number of off policy opt iterations has been reached
                kl_threshold_reached = True
            
            if kl_threshold_reached:
                # Indices of the batches corresponding to the percentile
                percentile_indices = []
                # In case of successful off-policy optimization, compute entropy of new policy
                if not numeric_error:
                    with torch.no_grad():
                        entropies = compute_entropy(last_valid_target_policy, last_valid_target_policy, states, actions,
                                                    num_batches, batch_dimension, use_percentile, percentile, percentile_indices, baseline, effective_trajectory_lengths,
                                                    distances, indices, biased_full_entropy, batch_entropies, False, k, G, B, ns, eps)
                        # CVaR entropy
                        if use_percentile:
                            entropy = torch.mean(entropies)
                        else:
                            entropy = torch.mean(torch.sort(entropies).values[:percentile])

                    perc_sampled_env = ""
                    if use_percentile:
                        # Compute the occurrences of the sampled environments
                        percentile_sampled_env_batched = torch.zeros((percentile), dtype=torch.int32)
                        for batch in range(percentile):
                            percentile_sampled_env_batched[batch] = sampled_env_batched[percentile_indices[0][batch]]

                        occurrences = collections.Counter(np.array(percentile_sampled_env_batched))
                        perc_sampled_env = ""
                        for c in range(len(sampling_dist)):
                            perc_sampled_env += str(occurrences.get(c))
                            if c < len(sampling_dist)-1:
                                perc_sampled_env += "-"
               
                if np.isnan(entropy) or np.isinf(entropy):
                    print("Aborting because final entropy is nan or inf...")
                    print("There is most likely a problem in knn aliasing. Use a higher k.")
                    exit()
                else:
                    # End of epoch, prepare statistics to log
                    epoch += 1

                    # Update behavioral policy
                    behavioral_policy.load_state_dict(last_valid_target_policy.state_dict())
                    target_policy.load_state_dict(last_valid_target_policy.state_dict())

                    loss = - entropy.numpy()
                    entropy = entropy.numpy()
                    execution_time = time.time() - t0

                    if epoch % heatmap_every == 0:
                        # Heatmap
                        if heatmap_discretizer is not None or env.__class__.__name__ == 'MazeEnv' or env.__class__.__name__ == 'ImgObsWrapper':
                            _, heatmap_entropy, heatmap_image = \
                                get_heatmap(training_envs, sampling_dist, behavioral_policy, heatmap_discretizer, heatmap_episodes, heatmap_num_steps, batch_dimension, heatmap_cmap, heatmap_interp, heatmap_labels)
                        else:
                            heatmap_entropy = None
                            heatmap_image = None

                        # Indices of the batches corresponding to the percentile
                        percentile_indices = []
                        # Full entropy
                        states, actions, effective_trajectory_lengths, distances, indices, sampled_env_batched = \
                            collect_particles_and_compute_knn(training_envs, sampling_dist, behavioral_policy, random_encoder, num_trajectories,
                                                                trajectory_length, 1, num_trajectories, state_filter, full_entropy_k, num_workers)

                        with torch.no_grad():
                            full_entropies = compute_entropy(behavioral_policy, behavioral_policy, states, actions,
                                                            1, num_trajectories, 0, num_batches, percentile_indices, baseline, effective_trajectory_lengths,
                                                            distances, indices, biased_full_entropy, batch_entropies, False, full_entropy_k, G, full_B, ns, eps)
                        
                            full_entropy = full_entropies[0]

                        # Save policy
                        torch.save(behavioral_policy.state_dict(), os.path.join(out_path, f"{epoch}-policy"))

                    else:
                        heatmap_entropy = None
                        heatmap_image = None

                    # Log statistics for this epoch
                    log_epoch_statistics(
                        writer=writer, log_file=log_file, log_entropies=csv_file_4, log_gradients=log_gradients, csv_file_1=csv_file_1, csv_file_2=csv_file_2,
                        epoch=epoch,
                        loss=loss,
                        entropy=entropy,
                        batch_entropies=batch_entropies[-1] if len(batch_entropies) > 0 else None,
                        sampled_env=sampled_env_batched,
                        grads=grads,
                        execution_time=execution_time,
                        num_off_iters=num_off_iters,
                        full_entropy=full_entropy,
                        biased_full_entropy=biased_full_entropy[-1] if len(biased_full_entropy) > 0 else None,
                        perc_sampled_env=perc_sampled_env,
                        heatmap_image=heatmap_image,
                        heatmap_entropy=heatmap_entropy,
                        backtrack_iters=backtrack_iter,
                        backtrack_lr=learning_rate
                    )