from ocatari.ram.freeway import *

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

    if player:
        if STARTING_Y is None:
            STARTING_Y = player.y

        progress = STARTING_Y - player.y
        normalized_progress = progress / STARTING_Y
        reward += normalized_progress * 3

        for vehicle in vehicles:
            if (player.x <= vehicle.x + vehicle.w) and (player.x + player.w > vehicle.x) and \
               (player.y <= vehicle.y + vehicle.h) and (player.y + player.h > vehicle.y):
                reward -= 3.0  

        if player.y <= 20:  #assuming the goal/top is at 20
            reward += 100 

    return reward
