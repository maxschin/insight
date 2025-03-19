from ocatari.ram.pong import *
import random

SCORING = False


def reward_function(self) -> float:
    
    global SCORING
    reward = 0.0

    game_objects = self.objects

    SCREEN_WIDTH = 160
        
    # Define categories for easy identification
    player = None
    opponent = None
    ball = None

    # Classify objects by type
    for obj in game_objects:
        if isinstance(obj, Player):
            player = obj
        elif isinstance(obj, Enemy):
            enemy = obj
        elif isinstance(obj, Ball):
            ball = obj

    
    reward = random.uniform(-1, 1)

    return reward
