python src/experiments/alphamepol_visual.py --env MiniGrid \
    --sampling_dist 0.8 0.2 --batch_dimension 5 --use_percentile 1 \
    --percentile 6 --k 50 --kl_threshold 15 --learning_rate 0.00001 --num_trajectories 100 \
    --trajectory_length 150 --num_epoch 300 --heatmap_every 50 --heatmap_episodes 100 \
    --heatmap_num_steps 150 --full_entropy_k 50