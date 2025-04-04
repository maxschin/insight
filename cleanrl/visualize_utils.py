# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
from distutils.util import strtobool
import matplotlib.pyplot as plt
import torch
import cv2
from tqdm import tqdm

def images_to_video(folder_path, output_path, fps):
    # Get all the image file names in the folder and sort them numerically
    filenames = [os.path.join(folder_path, f"{i}.png") for i in range(100)]
    # filenames.sort()

    # Read the first image to get the width and height of the video
    frame = cv2.imread(filenames[0])
    h, w, layers = frame.shape
    size = (w, h)

    # Create a video writing object using the XVID codec and MP4V container
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    # Add image to video
    for filename in filenames:
        img = cv2.imread(filename)
        out.write(img)

    out.release()


def visual_for_videos(envs, od_model, next_obs, device, args, run_name):
     print('save frames')
     os.makedirs(os.path.join('ppoeql_stack_cnn_out_frames', run_name), exist_ok=True)
     input_path = os.path.join('ppoeql_stack_cnn_out_frames', run_name, 'test')

     # Use tqdm in a loop
     for n in tqdm(range(100), desc="Processing"): # The desc parameter is used to set the progress bar description
         with torch.no_grad():
             action = envs.action_space.sample()
         next_obs, reward, done, _, info = envs.step(action)
         next_obs = torch.Tensor(next_obs).to(device)
         vis_obs = next_obs[0, 0, :, :, :]
         cors = od_model(next_obs / 255.0)[0, :]
         vis_obs = vis_obs.cpu().numpy()
         size = vis_obs.shape
         plt.figure(figsize=(10, 10))
         plt.axis('off')
         plt.imshow(vis_obs.astype('uint8'))
         for i in range(256):
             cors[i * args.obj_vec_length] = cors[i * args.obj_vec_length] * size[0]
             cors[i * args.obj_vec_length + 1] = cors[i * args.obj_vec_length + 1] * size[1]
             plt.text(cors[i * args.obj_vec_length + 1], cors[i * args.obj_vec_length], str(i + 1), fontsize=36, color='green')
         plt.savefig(os.path.join('ppoeql_stack_cnn_out_frames', run_name, 'test', f'{n}.png'))
         plt.close()

     output_path = input_path + '_seg.mp4'
     fps = 20
     images_to_video(input_path, output_path, fps)
     return

def visual_for_ocatari_agent_videos(envs, agent, device, args, output_folder, n_step=200, actor="neural", label="test"):
    output_path = os.path.join(output_folder, f"{actor}_{label}.mp4")
    
    # Handle different Gym API versions
    result = envs.reset()
    obs = result[0] if isinstance(result, tuple) else result
    obs = torch.Tensor(obs).to(device)

    first_frame = envs.render()
    height, width, channels = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20  
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for n in tqdm(range(n_step), desc="Shooting Video"):
        frame = envs.render()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)

        action, _, _, _, _, _ = agent.get_action_and_value(obs, actor=actor)
        obs_result = envs.step(action.cpu().numpy())

        if isinstance(obs_result, tuple):  # Ensure unpacking matches return structure
            obs = obs_result[0]  # Extract only the observation
        else:
            obs = obs_result

        obs = torch.Tensor(obs).to(device)

    video_writer.release()
    return output_path

def visual_for_agent_videos(envs, agent, next_obs, device, args, run_name, input_path, n_step=200, threshold=0.8):
    # Use tqdm in a loop
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    for n in tqdm(range(n_step), desc="Processing"): # The desc parameter is used to set the progress bar description
        with torch.no_grad():
            action, logprob, _, value,_,prob = agent.get_action_and_value(next_obs, threshold=threshold)
            cors = agent.network(next_obs / 255.0, threshold=threshold)[0, :].cpu()
        next_obs, reward, done, _, info = envs.step(action)
        next_obs = torch.Tensor(next_obs).to(device)
        if args.gray:
            vis_obs = next_obs[0, 0, :, :]
        else:
            vis_obs = next_obs[0, 0, :, :, :]
        vis_obs = vis_obs.cpu().numpy()
        size = vis_obs.shape
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(vis_obs.astype('uint8'))
        for i in range(256):
            cors[i * args.obj_vec_length] = cors[i * args.obj_vec_length] * size[0]
            cors[i * args.obj_vec_length + 1] = cors[i * args.obj_vec_length + 1] * size[1]
            plt.text(cors[i * args.obj_vec_length + 1], cors[i * args.obj_vec_length], str(i + 1), fontsize=36, color='green')
        plt.savefig(os.path.join(input_path, f'{n}.png'))
        plt.close()

    output_path = input_path + '_seg.mp4'
    fps = 20
    return images_to_video(input_path, output_path, fps)

def visual(agent, next_obs,args,update,run_name):
    # state original shape: n_batch, n_frame, height, width, n_channel
    record_path = os.path.join('ppoeql_stack_cnn_out_frames', run_name,'record')
    vis_obs = next_obs[0,0,:,:,:]
    cors = agent.network(next_obs / 255.0)[0,:]
    vis_obs = vis_obs.cpu().numpy()
    size = vis_obs.shape
    plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.imshow(vis_obs.astype('uint8'))
    for i in range(256):
        cors[i*9] = cors[i*9]*size[0]
        cors[i*9+1] = cors[i*9+1]*size[1]
        plt.text(cors[i*9+1], cors[i*9], str(i+1), fontsize=36,color='green')
    # vis_obs = cv2.cvtColor(vis_obs, cv2.COLOR_GRAY2RGB)
    plt.savefig(os.path.join(record_path, f'{update}.png'))
    # plt.show()
    plt.close()
    output_path = record_path+'_seg.mp4'
    return
