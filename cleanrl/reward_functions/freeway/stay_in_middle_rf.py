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

        distance_to_45 = abs(player.y-50)
        reward -= distance_to_45

        for vehicle in vehicles:
            if (player.x <= vehicle.x + vehicle.w) and (player.x + player.w > vehicle.x) and \
               (player.y <= vehicle.y + vehicle.h) and (player.y + player.h > vehicle.y):
                reward -= 30.0  

        if 40 < player.y < 50 :
            reward += 10

    return reward
