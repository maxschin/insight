from ocatari.core import OCAtari
import random

from tqdm import tqdm
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import pandas as pd
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("--game", type=str, default="Pong")
parser.add_argument("--frames", type=int, default=10000)

args = parser.parse_args()

if not os.path.isdir("batch_training/" + args.game):
    os.mkdir("batch_training/" + args.game)

env = OCAtari(args.game, mode="both", hud=True, render_mode="rgb_array")
env.metadata['render_fps'] = 30
observation, info = env.reset()

game_nr = 0
turn_nr = 0
dataset = {"INDEX": [], "RAM": [], "VIS": [], "HUD": []}
frames = []

recorded_frames = []

for i in tqdm(range(args.frames)):
    action = random.randint(0, env.nb_actions-1)
    obs, reward, terminated, truncated, info = env.step(action)

    frame = env.render()
    recorded_frames.append(frame)

    step = f"{'%0.5d' % (game_nr)}_{'%0.5d' % (turn_nr)}"
    dataset["INDEX"].append(step)

    # dataset["OBS"].append(obs.flatten().tolist())
    dataset["VIS"].append(
        [x for x in sorted(env.objects_v, key=lambda o: str(o))])
    dataset["RAM"].append(
        [x for x in sorted(env.objects, key=lambda o: str(o)) if x.hud == False])
    dataset["HUD"].append(
        [x for x in sorted(env.objects, key=lambda o: str(o)) if x.hud == True])
    turn_nr = turn_nr + 1

    # if a game is terminated, restart with a new game and update turn and game counter
    if terminated or truncated:
        observation, info = env.reset()
        turn_nr = 0
        game_nr = game_nr + 1

env.close()

clip = ImageSequenceClip(recorded_frames, fps=30)
clip.write_videofile("batch_training/" + args.game + "/" + args.game + ".mp4")

df = pd.DataFrame(dataset, columns=['INDEX', 'RAM', 'HUD', 'VIS'])
df.to_csv("batch_training/" + args.game + "/" + args.game + ".csv", index=False)