python src/experiments/goal_rl.py --env AntMaze \
    --config 1 --num_epochs 100 --batch_size 40000 \
    --traj_len 400 --kl_thresh 1e-2 --policy_init ./pretrained/ant