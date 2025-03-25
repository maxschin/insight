#!/bin/bash
# Change to the cleanrl directory
cd cleanrl

# Define the game name
GAME="PongNoFrameskip-v4"

# Define an array of reward functions for Pong
#reward_funcs=("default" "close_but_no_hit_rf" "opposite_of_enemy_rf" "random_rf" "up_and_down_rf")
reward_funcs=("close_but_no_hit_rf" "opposite_of_enemy_rf" "random_rf" "up_and_down_rf")

# Define an associative array for timesteps per reward function
declare -A timesteps
timesteps["default"]=10000000
timesteps["close_but_no_hit_rf"]=5000000
timesteps["opposite_of_enemy_rf"]=5000000
timesteps["random_rf"]=50000
timesteps["up_and_down_rf"]=5000000

# Iterate over each reward function
for reward in "${reward_funcs[@]}"; do
    total_steps=${timesteps[$reward]}
    if [ "$reward" == "default" ]; then
      echo "Running experiment for game: $GAME with no reward function and total timesteps: $total_steps"
      python train_policy_atari.py --game="$GAME" --total-timesteps "$total_steps"
    else
      echo "Running experiment for game: $GAME with reward function: $reward and total timesteps: $total_steps"
      python train_policy_atari.py --game="$GAME" --reward_function="$reward" --total-timesteps "$total_steps"
    fi
done
