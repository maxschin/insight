from ocatari.ram.pong import *

SCORING = False


def reward_function(self) -> float:
    reward = 0.0
    game_objects = self.objects
    SCREEN_HEIGHT = 210
    margin = 2  # pixels: defines the 'edge' region

    # Identify the player's paddle.
    player = None
    for obj in game_objects:
        if isinstance(obj, Player):
            player = obj
            break

    if player is None:
        return reward

    # Initialize state tracking if necessary.
    if not hasattr(self, 'prev_player_y'):
        self.prev_player_y = player.y  # store previous y position
        # Also store last movement direction: +1 means moving down, -1 means moving up.
        self.prev_direction = 0  

    # Compute the vertical movement (delta)
    delta_y = player.y - self.prev_player_y

    # Reward continuous movement: no movement is penalized.
    if delta_y == 0:
        reward -= 0.1  # small penalty for idling
    else:
        reward += 0.1  # small bonus for moving

    # Check for proper behavior at the edges:
    # When near the top, the paddle should be moving down (delta_y > 0)
    if player.y <= margin:
        if delta_y > 0:
            reward += 1.0  # bonus for switching to move down at the top
        else:
            reward -= 1.0  # penalty for not moving down at the top

    # When near the bottom, the paddle (using its bottom edge) should be moving up.
    if (player.y + player.h) >= (SCREEN_HEIGHT - margin):
        if delta_y < 0:
            reward += 1.0  # bonus for switching to move up at the bottom
        else:
            reward -= 1.0  # penalty for not moving up at the bottom

    # Update state for next time step.
    self.prev_player_y = player.y

    return reward
