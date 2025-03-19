import os

def save_equations(expression_list, output_folder, run_name):
    """
    Save a list of equations to a text file in the specified output folder.

    Parameters:
    - expression_list (list): List of string equations to save.
    - output_folder (str): Path to the folder where the file will be saved.
    - run_name (str): A name to include in the filename, e.g., identifying this run.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Define the output filename
    file_name = f"{run_name}_equations.txt"
    file_path = os.path.join(output_folder, file_name)
    
    # Write each equation to the file, one per line
    with open(file_path, 'w') as f:
        for equation in expression_list:
            f.write(equation + "\n\n")
    
    print(f"Equations saved to: {file_path}")


def get_reward_func_path(game, reward_func_name):
    """
    Build and validate the full file path for a reward function.
    
    The expected file structure is:
        reward_functions/<game>/<reward_func_name>.py
    
    Parameters:
        game (str): Name of the game directory under 'rewards'.
        reward_func_name (str): Base name of the reward function file (without .py).
    
    Returns:
        str: The absolute path to the reward function file.
    
    Raises:
        ValueError: If the game or reward_func_name contains invalid characters.
        FileNotFoundError: If the game directory or reward function file does not exist.
    """
    base_dir = "reward_functions"
    
    # Validate that game and reward_func_name don't contain path separators
    for part, name in [(game, "game"), (reward_func_name, "reward_func_name")]:
        if os.path.sep in part or "/" in part or "\\" in part:
            raise ValueError(f"Invalid {name!r}. It should not contain path separators.")

    # ensure lower case game
    game = game.lower()

    # Construct the game directory path and validate it exists.
    game_path = os.path.join(base_dir, game)
    if not os.path.isdir(game_path):
        raise FileNotFoundError(f"Game directory does not exist: {game_path}")
    
    # Construct the full file path and validate the file exists.
    file_name = reward_func_name + ".py"
    file_path = os.path.join(game_path, file_name)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Reward function file does not exist: {file_path}")
    
    # Return the absolute path
    return os.path.abspath(file_path)
