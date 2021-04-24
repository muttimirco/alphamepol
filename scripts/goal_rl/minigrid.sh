python src/experiments/goal_rl.py --env MiniGrid \
    --config 1 --num_epochs 200 --batch_size 7500 \
    --traj_len 150 --kl_thresh 1e-4 --policy_init ./pretrained/minigrid