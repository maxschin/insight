from ocatari.ram.pong import *

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

    if ball:
# reward for scoring
        if ball.x <= 0 and SCORING:
            reward += 1.0
            SCORING = True

# reward for opponent scoring
        if (ball.x + ball.w) >= SCREEN_WIDTH and not SCORING:
            reward = -1.0
            SCORING = True

    if SCORING and (10 < ball.x < SCREEN_WIDTH - 10):
            SCORING = False

    return reward
