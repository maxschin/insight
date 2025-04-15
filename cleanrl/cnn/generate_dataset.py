import os
import sys
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

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

target_dir = os.path.join(SRC, "batch_training/" + args.game)
if not os.path.isdir(target_dir):
    os.mkdir(target_dir)

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
video_file_path = os.path.join(target_dir, args.game + ".mp4")
clip.write_videofile(video_file_path)

df = pd.DataFrame(dataset, columns=['INDEX', 'RAM', 'HUD', 'VIS'])
csv_file_path = os.path.join(target_dir, args.game + ".csv")
df.to_csv(csv_file_path, index=False)
