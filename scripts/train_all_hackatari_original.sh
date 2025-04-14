#!/bin/bash
# Change to the cleanrl/train directory
cd cleanrl/train

# List of games (exactly as specified)
games=("PongNoFrameskip-v4" "SeaquestNoFrameskip-v4" "MsPacmanNoFrameskip-v4" "SpaceInvadersNoFrameskip-v4" "FreewayNoFrameskip-v4")

# Reward configurations for Pong with custom reward functions.
declare -A reward_configs_Pong=(
    ["default"]=10000000
    ["close_but_no_hit_rf"]=10000000
    ["opposite_of_enemy_rf"]=10000000
    ["random_rf"]=10000000
    ["up_and_down_rf"]=10000000
)

# Default reward configuration for all other games.
declare -A reward_configs_default=(
    ["default"]=10000000
)

# Map each game to its corresponding reward configuration associative array.
declare -A reward_config_names=(
    ["PongNoFrameskip-v4"]="reward_configs_default" # change to reward_configs_Pong if you want to train Pong with different reward functions
    ["SeaquestNoFrameskip-v4"]="reward_configs_default"
    ["MsPacmanNoFrameskip-v4"]="reward_configs_default"
    ["SpaceInvadersNoFrameskip-v4"]="reward_configs_default"
    ["FreewayNoFrameskip-v4"]="reward_configs_default"
)

# Iterate over every game.
for game in "${games[@]}"; do
    echo "=== Processing game: $game ==="
    
    # Determine the reward configuration for the current game.
    config_array_name=${reward_config_names[$game]}
    # Create a nameref to the appropriate associative array (Bash 4.3+ required)
    declare -n config="$config_array_name"
    
    # Iterate over each reward function defined in the associative array.
    for reward in "${!config[@]}"; do
        timesteps=${config[$reward]}
        timesteps=10
        # Determine the reward parameter to pass:
        if [ "$reward" == "default" ]; then
            echo "Training $game using default reward function with timesteps: $timesteps"
            python train_policy_atari.py --game="$game" --total-timesteps "$timesteps"
        else
            echo "Training $game using custom reward function '$reward' with timesteps: $timesteps"
            python train_policy_atari.py --game="$game" --total-timesteps "$timesteps" --reward_function="$reward"
        fi
    done
    echo ""  # Newline for clarity between games.
done

echo "All game and reward function combinations processed."
