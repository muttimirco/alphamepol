python src/experiments/alphamepol.py --env AntMaze \
    --sampling_dist 0.8 0.2 --batch_dimension 5 --use_percentile 1 \
    --percentile 6 --k 500 --kl_threshold 15 --learning_rate 0.00001 --num_trajectories 150 \
    --trajectory_length 400 --num_epoch 400 --heatmap_every 50 --heatmap_episodes 150 \
    --heatmap_num_steps 400 --full_entropy_k 500