#!/bin/bash
# Change to the cleanrl directory
cd cleanrl

# Define an ordered array of games
#games=("Pong" "Freeway" "Seaquest")
games=("Pong")

# Define an associative array where each key is a game and the value is a space-separated list of reward functions.
declare -A reward_funcs
#reward_funcs["Pong"]="default basic_rf close_but_no_hit_rf opposite_of_enemy_rf random_rf"
reward_funcs["Pong"]="default"
#reward_funcs["Freeway"]="default basic_rf stay_in_middle_rf random_rf"
#reward_funcs["Seaquest"]="default basic_rf stick_to_surface_rf random_rf"

# Iterate over each game
for game in "${games[@]}"; do
  # Get the reward functions for this game
  rewards=${reward_funcs[$game]}
  
  # Iterate over each reward function for this game
  for reward in $rewards; do
    # Handle empty reward function by setting it to an empty argument
    if [ "$reward" == "default" ]; then
      echo "Running experiment for game: $game with no reward function"
      python train_then_distill_policy_hackatari.py --game="$game"
    else
      echo "Running experiment for game: $game with reward function: $reward"
      python train_then_distill_policy_hackatari.py --game="$game" --reward_function="$reward"
    fi
  done
done
