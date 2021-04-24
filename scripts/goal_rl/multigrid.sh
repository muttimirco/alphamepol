python src/experiments/goal_rl.py --env MultiGrid \
    --config 0 --num_epochs 100 --num_goals 50 --batch_size 12000 \
    --traj_len 400 --kl_thresh 1e-4 --policy_init ./pretrained/multigrid