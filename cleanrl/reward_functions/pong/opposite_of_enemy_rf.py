from ocatari.ram.pong import *

SCORING = False


def reward_function(self) -> float:
    
    global SCORING
    reward = 0.0

    game_objects = self.objects

    SCREEN_WIDTH = 160
    SCREEN_HEIGHT = 210
        
    # Define categories for easy identification
    player = None
    enemy = None
    ball = None

    # Classify objects by type
    for obj in game_objects:
        if isinstance(obj, Player):
            player = obj
        elif isinstance(obj, Enemy):
            enemy = obj
        elif isinstance(obj, Ball):
            ball = obj

    if player and enemy:
        player_center = player.y + player.h / 2
        enemy_center = enemy.y + enemy.h / 2
        
        mirror_player_center = SCREEN_HEIGHT - enemy_center
        
        difference = abs(player_center - mirror_player_center)
        normalized_difference = difference / SCREEN_HEIGHT
    
        reward += 1.0 - normalized_difference

    return reward