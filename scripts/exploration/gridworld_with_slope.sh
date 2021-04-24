python src/experiments/memento.py --env GridWorld \
    --sampling_dist 0.8 0.2 --batch_dimension 5 --use_percentile 1 \
    --percentile 14 --k 30 --kl_threshold 15 \
    --learning_rate 0.00001 --num_trajectories 200 --trajectory_length 400 \
    --num_epoch 150 --zero_mean_start 1 --heatmap_every 10 --heatmap_episodes 200 \
    --heatmap_num_steps 400 --full_entropy_k 30