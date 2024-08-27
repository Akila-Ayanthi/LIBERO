import argparse
import sys
import os
import setproctitle

# TODO: find a better way for this?
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import hydra
import json
import numpy as np
import pprint
import time
import torch
import wandb
import yaml
import multiprocessing
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import cv2
import copy

from easydict import EasyDict
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoModel, pipeline, AutoTokenizer, logging
from pathlib import Path

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from libero.libero.utils.time_utils import Timer
from libero.libero.utils.video_utils import VideoWriter
from libero.lifelong.algos import *
from libero.lifelong.datasets import get_dataset, SequenceVLDataset, GroupedTaskDataset
from libero.lifelong.metric import (
    evaluate_loss,
    evaluate_success,
    raw_obs_to_tensor_obs,
)
from libero.lifelong.utils import (
    control_seed,
    safe_device,
    torch_load_model,
    NpEncoder,
    compute_flops,
)

from libero.lifelong.main import get_task_embs
from libero.lifelong.models.bc_transformer_policy import PerturbationAttention

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils

import time


benchmark_map = {
    "libero_10": "LIBERO_10",
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
}

algo_map = {
    "base": "Sequential",
    "er": "ER",
    "ewc": "EWC",
    "packnet": "PackNet",
    "multitask": "Multitask",
}

policy_map = {
    "bc_rnn_policy": "BCRNNPolicy",
    "bc_transformer_policy": "BCTransformerPolicy",
    "bc_vilt_policy": "BCViLTPolicy",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--experiment_dir", type=str, default="experiments")
    # for which task suite
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["libero_10", "libero_spatial", "libero_object", "libero_goal"],
    )
    parser.add_argument("--task_id", type=int, nargs='+', required=True)
    # method detail
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["base", "er", "ewc", "packnet", "multitask"],
    )
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        choices=["bc_rnn_policy", "bc_transformer_policy", "bc_vilt_policy"],
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ep", type=int)
    parser.add_argument("--load_task", type=int)
    parser.add_argument("--device_id", type=int)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--folder", type=int, help="folder with the model to load")
    # parser.add_argument('--save_dir',  type=str, required=True)
    args = parser.parse_args()
    args.device_id = "cuda:" + str(args.device_id)
    args.save_dir = f"{args.experiment_dir}_saved"

    if args.algo == "multitask":
        assert args.ep in list(
            range(0, 50, 5)
        ), "[error] ep should be in [0, 5, ..., 50]"
    else:
        assert args.load_task in list(
            range(10)
        ), "[error] load_task should be in [0, ..., 9]"
    return args


def main():

    # Set the process name to 'python'
    setproctitle.setproctitle('python')

    args = parse_args()
    # e.g., experiments/LIBERO_SPATIAL/Multitask/BCRNNPolicy_seed100/

    experiment_dir = os.path.join(
        args.experiment_dir,
        f"{benchmark_map[args.benchmark]}/"
        + f"{algo_map[args.algo]}/"
        + f"{policy_map[args.policy]}_seed{args.seed}",
    )

    # find the checkpoint
    experiment_id = 0
    for path in Path(experiment_dir).glob("run_*"):
        if not path.is_dir():
            continue
        try:
            # folder_id = int(str(path).split("run_")[-1])
            folder_id = args.folder
            print("folder id", folder_id)
            if folder_id > experiment_id:
                experiment_id = folder_id
        except BaseException:
            pass
    if experiment_id == 0:
        print(f"[error] cannot find the checkpoint under {experiment_dir}")
        sys.exit(0)

    run_folder = os.path.join(experiment_dir, f"run_{experiment_id:03d}")
    print("run folder", run_folder)
    try:
        if args.algo == "multitask":
            model_path = os.path.join(run_folder, f"multitask_model_ep{args.ep}.pth")
            sd, cfg, previous_mask = torch_load_model(
                model_path, map_location=args.device_id
            )
        else:
            model_path = os.path.join(run_folder, f"task{args.load_task}_model.pth")
            sd, cfg, previous_mask = torch_load_model(
                model_path, map_location=args.device_id
            )
    except:
        print(f"[error] cannot find the checkpoint at {str(model_path)}")
        sys.exit(0)

    cfg.folder = '/datasets/work/d61-csirorobotics/source/LIBERO' #/home/dis023/dgx #get_libero_path("datasets")
    cfg.bddl_folder = get_libero_path("bddl_files")
    cfg.init_states_folder = get_libero_path("init_states")

    cfg.device = args.device_id
    algo = safe_device(eval(algo_map[args.algo])(10, cfg), cfg.device)
    algo.policy.previous_mask = previous_mask

    if cfg.lifelong.algo == "PackNet":
        algo.eval()
        for module_idx, module in enumerate(algo.policy.modules()):
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                weight = module.weight.data
                mask = algo.previous_masks[module_idx].to(cfg.device)
                weight[mask.eq(0)] = 0.0
                weight[mask.gt(args.task_id + 1)] = 0.0
                # we never train norm layers
            if "BatchNorm" in str(type(module)) or "LayerNorm" in str(type(module)):
                module.eval()

    algo.policy.load_state_dict(sd)

    if not hasattr(cfg.data, "task_order_index"):
        cfg.data.task_order_index = 0

    # get the benchmark the task belongs to
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    descriptions = [benchmark.get_task(i).language for i in range(10)]
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    tsne_features_agent, tsne_features_eye, tsne_labels, tsne_features_text, tsne_features_temporal, tsne_gmm_features = [], [], [], [], [], []


    directory = args.benchmark + '_visualisations'
    if not os.path.exists(directory):
        os.makedirs(directory)


    for task_id in args.task_id:
        task = benchmark.get_task(task_id)
        

    # ======================= start evaluation ============================

        print(f"===============Evaluating on task {task_id} with description {descriptions[task_id]}==============================")
        print("len of features", len(tsne_features_agent), len(tsne_labels))

        # 1. evaluate dataset loss
        try:
            dataset, shape_meta = get_dataset(
                dataset_path=os.path.join(
                    cfg.folder, benchmark.get_task_demonstration(task_id)
                ),
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=True,
                seq_len=cfg.data.seq_len,
            )
            dataset = GroupedTaskDataset(
                [dataset], task_embs[task_id : task_id + 1]
            )
        except:
            print(
                f"[error] failed to load task {task_id} name {benchmark.get_task_names()[task_id]}"
            )
            sys.exit(0)

        algo.eval()

        test_loss = 0.0

        # 2. evaluate success rate
        if args.algo == "multitask":
            save_folder = os.path.join(
                args.save_dir,
                f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_ep{args.ep}_on{task_id}.stats",
            )
        else:
            save_folder = os.path.join(
                args.save_dir,
                f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_load{args.load_task}_on{task_id}.stats",
            )

        video_folder_agent = os.path.join(
            args.save_dir,
            f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_load{args.load_task}_on{task_id}_agent_videos",
        )

        video_folder_eye = os.path.join(
                    args.save_dir,
                    f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_load{args.load_task}_on{task_id}_eye_videos",
                )


        with Timer() as t, VideoWriter(video_folder_agent, args.save_videos) as video_writer1, VideoWriter(video_folder_eye, args.save_videos) as video_writer2:
            env_args = {
                "bddl_file_name": os.path.join(
                    cfg.bddl_folder, task.problem_folder, task.bddl_file
                ),
                "camera_heights": cfg.data.img_h,
                "camera_widths": cfg.data.img_w,
            }

            env_num = 5  #20
            env = SubprocVectorEnv(
                [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
            )
            env.reset()
            env.seed(cfg.seed)
            algo.reset()

            init_states_path = os.path.join(
                cfg.init_states_folder, task.problem_folder, task.init_states_file
            )
            init_states = torch.load(init_states_path)
            print("init states", init_states.shape)
            indices = np.arange(env_num) % init_states.shape[0]
            print("indices", indices)
            init_states_ = init_states[indices]
            print("init state", init_states_)

            # Assuming init_states_ is a NumPy array
            init_states_1 = np.array(init_states_)  # Convert to NumPy array if it's not already

            # Check if all rows are the same
            first_state = init_states_1[0]
            all_same = np.all(np.all(init_states_1 == first_state, axis=1))
            print("All initial states are the same:", all_same)

            dones = [False] * env_num
            steps = 0
            obs = env.set_init_state(init_states_)
            task_emb = benchmark.get_task_emb(task_id)

            num_success = 0
            for _ in range(5):  # simulate the physics without any actions
                env.step(np.zeros((env_num, 7)))

            with torch.no_grad():
                while steps < cfg.eval.max_steps:
                    steps += 1

                    data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
                    actions = algo.policy.get_action(data)
                    obs, reward, done, info = env.step(actions)
                    # print("obs", obs)
                    copy_obs = copy.deepcopy(obs)
                    # video_writer.append_vector_obs(
                    #     obs, dones, camera_name="agentview_image"
                    # )


                    if steps%1==0:
                        images = ['agentview', 'eye_in_hand']
                        for image_ in images:
                            encoder_name = image_+'_rgb'
                            perturb_attn = PerturbationAttention(algo.policy.image_encoders[encoder_name]["encoder"], device=cfg.device)

                            # data = algo.policy.preprocess_input(data, train_mode=False)
                            attn_weights = perturb_attn(data, algo.policy, encoder_name, 'image_encoder')
                            # print("attention weights", attn_weights.shape)

                            for i in range(env_num):                            
                                attn_map = attn_weights[i, 0]  # Shape now (128, 128)
                                attn_map = np.flipud(attn_map)
                                # print("map", attn_map.shape)
                                # print("min max", np.min(attn_map), np.max(attn_map))

                                attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map))

                                # print("min max", np.min(attn_map), np.max(attn_map))

                                image = data["obs"][encoder_name][i].permute(1, 2, 0).detach().cpu().numpy()
                                image = np.flipud(image)
                                # image = image/255.0

                                # print(image.dtype, np.min(image), np.max(image))  # Should be float and in [0, 1] range
                                # print(attn_map.dtype, np.min(attn_map), np.max(attn_map))  # Should be float and in [0, 1] range
                                # Create a figure and axis to plot the image
                                fig, ax = plt.subplots(figsize=(8, 8))

                                # Display the original image
                                ax.imshow(image, interpolation='bilinear')

                                # Overlay the attention map
                                ax.imshow(attn_map, cmap='jet', alpha=0.5)

                                # Remove the axes for better visualization
                                ax.axis('off')

                                # Save the combined image
                                image_path = os.path.join(directory,f'attention_overlay_loadtask{args.load_task}_on_{task_id}_{steps}.png')
                                # plt.savefig(image_path, bbox_inches='tight', pad_inches=0)

                                fig.canvas.draw()
                                overlay_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                                overlay_image = overlay_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                                plt.close(fig)

                                # Flip the image vertically (if needed) to match the original format
                                overlay_image = np.flipud(overlay_image)

                                # Normalize to [0, 1] range if required by the video writer (assuming normalization is needed)
                                overlay_image = overlay_image / 255.0

                                # print("copy obs", copy_obs[0].shape)

                                # for key, value in copy_obs[0].items():
                                #     if isinstance(value, np.ndarray):
                                #         print(f"Key: {key}, Shape: {value.shape}")
                                #     else:
                                #         print(f"Key: {key}, Type: {type(value)}")
                                
                                image_name = image_+'_image'
                                # print("copy_obs", copy_obs)

                                copy_obs[i][image_name] = overlay_image
                                # print(f"copy obs {i}---> {copy_obs[i]}")

                            # Append the observation to the video writer
                            if image_name=='agentview_image':
                                video_writer1.append_vector_obs(copy_obs, dones, camera_name="agentview_image")
                            else:
                                video_writer2.append_vector_obs(copy_obs, dones, camera_name="eye_in_hand_image")


                        
                    # check whether succeed
                    for k in range(env_num):
                        dones[k] = dones[k] or done[k]
                    if all(dones):
                        break

               
                for k in range(env_num):
                    num_success += int(dones[k])

            success_rate = num_success / env_num
            env.close()

            eval_stats = {
                "loss": test_loss,
                "success_rate": success_rate,
            }

            # os.system(f"mkdir -p {args.save_dir}")
            # torch.save(eval_stats, save_folder)
        print(
            f"[info] finish for ckpt at {run_folder} in {t.get_elapsed_time()} sec for rollouts"
        )
        print(f"Results are saved at {save_folder}")
        print(test_loss, success_rate)


    


    
def attention_maps(data, text, img_name, policy, task, args, module):

    directory = args.benchmark + '_visualisations'
    if not os.path.exists(directory):
        os.makedirs(directory)

    image = data['obs'][img_name]

    if module == 'image_encoder':
        features = policy.image_encoders[img_name]['encoder'](image, returnt='feats', langs=text)
        heatmap = torch.mean(features[0], dim=0).squeeze()  # Average along channels
        heatmap = heatmap.detach().cpu().numpy()
        heatmap = cv2.resize(heatmap, (image.shape[3], image.shape[2]))  # Resize to (width, height)
    else:
        data = policy.preprocess_input(data, train_mode=False)
        temporal_f = policy(data, returnt='feats')
        attn_weights = policy.temporal_transformer.attention_output[3].mean(dim=1)
        heatmap = attn_weights[0]
        heatmap = heatmap.detach().cpu().numpy()

    image_np = image[0].permute(1, 2, 0).detach().cpu().numpy()
    if image_np.dtype != np.uint8:
        image_np = np.uint8(255 * image_np / np.max(image_np))

    heatmap = np.uint8(255 * heatmap / (np.max(heatmap) + 1e-8))
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.resize(heatmap_colored, (image_np.shape[1], image_np.shape[0]))

    # # Adjust blending ratio if needed
    overlayed_image = cv2.addWeighted(image_np, 0.6, heatmap_colored, 0.4, 0)
    overlayed_image = np.flipud(overlayed_image)

    plt.figure(figsize=(10, 5))
    plt.imshow(overlayed_image)
    plt.title(img_name)
    file_path = os.path.join(directory, f"{img_name}_loaded_task{args.load_task}_on_{task}_{module}_.png")
    plt.savefig(file_path)
    # Define different blending ratios

    # blend_ratios = [0.2, 0.4, 0.6, 0.8]

    # # Plot each blended image in a subplot
    # plt.figure(figsize=(15, 10))  # Adjust the figure size as needed

    # for i, alpha in enumerate(blend_ratios):
    #     overlayed_image = cv2.addWeighted(image_np, alpha, heatmap_colored, 1 - alpha, 0)
    #     overlayed_image = np.flipud(overlayed_image)

    #     plt.subplot(2, 2, i + 1)  # Adjust subplot grid size as needed
    #     plt.imshow(overlayed_image)
    #     plt.title(f'Alpha {alpha:.1f}')

    # plt.suptitle(f"{img_name} attention maps with different blending ratios")
    # plt.savefig(f"{img_name}_loaded_task{args.load_task}_on_{task}_{module}_subplots.png")
    # plt.show()


if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn'
    if multiprocessing.get_start_method(allow_none=True) != "spawn":  
        multiprocessing.set_start_method("spawn", force=True)

    main()
