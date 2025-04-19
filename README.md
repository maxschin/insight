
# Not so INSIGHTful?

This repository builds on and is a fork of the [repository](https://github.com/ins-rl/insight) for the [INSIGHT](https://arxiv.org/abs/2403.12451) end-to-end explainable reinforcement learning framework. Using [OC_Atari](https://github.com/k4ntz/OC_Atari) and [HackAtari](https://github.com/k4ntz/HackAtari), we evaluated both the perceptual capabilities and the suitability of the INSIGHT framework for LLM-based explanations.

## Installation
We assume that you have a working installation of ```python3``` on your system. The code is tested for version 3.12 on Ubuntu 22.04 to be changed at your own peril. If you want to avoid the installation but have ```podman``` installed, you can run the most important scripts directly inside containers. See below.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install opencv-python-headless
```
If you want to run the object detection with SAM-Track as the authors did in the original paper, you might also have to install the following dependencies:
```bash
pip install groundingdino-py
```
## General usage
You can run the core script to pre-train the CNN, train the end-to-end INSIGHT agents, and then evaluate them with different HackAtari modifications either locally or inside a container. As of now, this runs the script for Pong, SpaceInvaders, MsPacman, Seaquest, and Freeway. The selection of games can be configured inside```scripts/pre_train_cnn_all.sh``` and ```scripts/train_all_hackatari_original.sh```.

### Run core script locally
```bash
bash scripts/pre_train_cnn_all.sh
bash scripts/train_all_hackatari_original.sh
python cleanrl/evaluate_trained_agents.py
```
### Run core script in container (recommended)
This assumes that you have ```podman``` set up on your system and, if you have an NVIDIA GPU, also that you have enabled CDI support. The relevant directories will be mounted such that results are saved locally.
```bash
bash scripts/docker_pre_train.sh
bash scripts/docker_run.sh --original --all
bash scripts/docker_eval.sh
```
A few notes on the flags of the ```scripts/docker_run.sh``` file:
- ```--all```: if this flag set, it runs the training for all games and their respective reward functions and training durations. If it is omitted, it runs the training only for Pong.
- ```--original```: this flags specifies if an end-to-end, pixel-to-action INSIGHT agent is trained as in the original paper or (if not set) an INSIGHT agent that works with the object-centric representation provided by OC_Atari
- ```--d```: if set, the container is executed in detached mode

If you want to change the games, reward functions, or training durations used, follow the breadcrumbs inside the file!

### Run other scripts locally
Generally, it is recommended to run these scripts from inside ```./cleanrl/```:
```bash
cd cleanrl
```
If you want to pre-train the CNN for any specific game, run the following series of commands. You should be able to substitue any farama Atari [game-string](https://ale.farama.org/environments/), although we've only tested for ```*NoFrameskip-v4``` type environments:
```bash
python cnn/generate_dataset.py --game=PongNoFrameskip-v4
python cnn/segment_video.py --game=PongNoFrameskip-v4
python cnn/transform_data.py --game=PongNoFrameskip-v4
python train/train_cnn_reorder.py --game=PongNoFrameskip-v4
```
If you want train an agent for different games or with different parameters individually using the end-to-end original approach, you can use (check the file for flag-options):
```bash
python train/train_policy_atari.py --game PongNoFrameskip-v4 --reward_function random_rf
```
*Important:* To train for a game, you need to first run pre-training for this game!

Alternatively, you can also train directly on OC_Atari object-centric representations of the environments. In this case, no pre-training is necessary although results will likely not be satisfactory in terms of EQL-agent performance:
```bash
python train/train_policy_ocatari.py --game PongNoFrameskip-v4 --reward_function random_rf
```
A further experimental training script where the EQL-agent is trained after the neural agent has finished training can be run (without any guarantees!) using:
```bash
python train/train_then_distill_policy_hackatari.py
```
## Results
Any agent training produces the following outputs, which can be navigated to via the run name (printed during training)
- *Video recordings*: During and after training both the EQL-agent and the neural agent are recorded, which can be found in ```./cleanrl/ppoeql_ocatari_videos/RUNNAME/```
- *EQL-equations*: After training the polynomials are extracted and stored in a text file which can be found at ```./cleanrl/equations/RUNNAME.txt```
- *Trained agent*: The agents themseles (both the EQL and neural agent) are saved during training at ```./cleanrl/models/agents/RUNNAME_final.pth```

## Object-detection benchmarking
An explanation of how the benchmarking of the object detection works, can be found in the README in `cleanrl/benchmark_object_detection`

## LLM evaluation
One of the central aims of the project was to evaluate the validity and usefulness of their approach to explainability. The authors suggested that the equations extracted from their agent can be effectively verbalized/explained by an LLM. To test this claim, we retrained a Pong agent using several different reward functions and then asked the LLM to use the equations the infer the reward function this agent was optimized for. We also evaluated the regular Pong agent in OOD environments which we manipulated using HackAtari and asked the LLM to predict its performance and behavior using the provided policy. Prompts, results, equations, etc. for these experiments can be found [here](https://drive.google.com/drive/folders/1yi9JkR5QRicLOnu9eG90hjm6KU-g8Tse).

## Contact
For any questions regarding the CNN-pre-training or benchmarking, reach out to [Alexander Doll](https://github.com/alexanderdoll) and for any questions regarding the LLM evaluation or the agent training, reach out to [Max Schindler](https://github.com/maxschin/).

## Acknowledgement
We thank our supervisor [Quentin Delfosse](https://github.com/k4ntz/) for his guidance and encouragement!

## License
This project is open sourced under [MIT License](LICENSE).
