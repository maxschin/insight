cd cleanrl
python train_policy_atari.py --wandb-project-name insight --env-id PongNoFrameskip-v4 --run-name benchmark-ng-reg-weight-1e-3-Pong-seed1 --ng True --reg_weight 1e-3 --seed 1 --load_cnn True --track True --cuda False --total-timesteps 1000
