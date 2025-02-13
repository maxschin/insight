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
            #reward = -1.0
            SCORING = True

# reward which is higher if the paddel is closer to the ball when it crosses
        if ball.x > player.x:
            ball_center_y = ball.y + ball.h / 2
            vertical_diff = abs(player.y - ball_center_y)
            MAX_VERTICAL_DIFF = 210 #screen_height
            scale = 0.01
            alignment_reward = scale * max(0, (1 - (vertical_diff / MAX_VERTICAL_DIFF)))
            reward += alignment_reward

        if SCORING and (10 < ball.x < SCREEN_WIDTH - 10):
            SCORING = False

    return reward
