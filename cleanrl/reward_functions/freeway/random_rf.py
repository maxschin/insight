from ocatari.ram.freeway import *
import random

STARTING_Y = None
SCREEN_HEIGHT = 210

def reward_function(self) -> float:
    global STARTING_Y
    reward = 0.0
    game_objects = self.objects

    # Define categories
    player = None
    vehicles = []

    # Classify objects
    for obj in game_objects:
        if isinstance(obj, Player):
            player = obj
        elif isinstance(obj, Vehicle):
            vehicles.append(obj)
    
    reward = random.uniform(-1, 1)
    
    return reward
