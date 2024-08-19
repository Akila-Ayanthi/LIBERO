import argparse
import sys
import os

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

    cfg.folder = '/datasets/work/d61-csirorobotics/source/LIBERO' #get_libero_path("datasets")
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

        video_folder = os.path.join(
            args.save_dir,
            f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_load{args.load_task}_on{task_id}_videos",
        )


        with Timer() as t, VideoWriter(video_folder, args.save_videos) as video_writer:
            env_args = {
                "bddl_file_name": os.path.join(
                    cfg.bddl_folder, task.problem_folder, task.bddl_file
                ),
                "camera_heights": cfg.data.img_h,
                "camera_widths": cfg.data.img_w,
            }

            env_num = 5
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
            indices = np.arange(env_num) % init_states.shape[0]
            init_states_ = init_states[indices]
            # print("init state", init_states_)

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
                    video_writer.append_vector_obs(
                        obs, dones, camera_name="agentview_image"
                    )

                    #plot attention maps
                    # if steps==1:
                    #     attention_maps(data, data["task_emb"], 'agentview_rgb', algo.policy, task_id, args, 'image_encoder')
                    #     attention_maps(data, data["task_emb"], 'eye_in_hand_rgb', algo.policy, task_id, args, 'image_encoder')
                    #     attention_maps(data, data["task_emb"], 'agentview_rgb', algo.policy, task_id, args, 'temp_transformer')
                    #     attention_maps(data, data["task_emb"], 'eye_in_hand_rgb', algo.policy, task_id, args, 'temp_transformer')

                    if steps%10==0:
                        perturb_attn = PerturbationAttention(algo.policy.image_encoders['agentview_rgb']["encoder"], device=cfg.device)

                        # data = algo.policy.preprocess_input(data, train_mode=False)
                        attn_weights = perturb_attn(data)
                        print("attention weights", attn_weights.shape)

                        attn_map = attn_weights[0, 0]  # Shape now (128, 128)
                        attn_map = np.flipud(attn_map)
                        print("map", attn_map.shape)
                        print("min max", np.min(attn_map), np.max(attn_map))

                        attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map))

                        print("min max", np.min(attn_map), np.max(attn_map))

                        image = data["obs"]["agentview_rgb"][0].permute(1, 2, 0).detach().cpu().numpy()
                        image = np.flipud(image)
                        # image = image/255.0

                        print(image.dtype, np.min(image), np.max(image))  # Should be float and in [0, 1] range
                        print(attn_map.dtype, np.min(attn_map), np.max(attn_map))  # Should be float and in [0, 1] range
                        # Create a figure and axis to plot the image
                        fig, ax = plt.subplots(figsize=(8, 8))

                        # Display the original image
                        ax.imshow(image, interpolation='bilinear')

                        # Overlay the attention map
                        ax.imshow(attn_map, cmap='jet', alpha=0.5)

                        # Remove the axes for better visualization
                        ax.axis('off')

                        # Save the combined image
                        plt.savefig(f'attention_overlay_loadtask{args.load_task}_on_{task_id}_{steps}.png', bbox_inches='tight', pad_inches=0)
                        plt.close(fig)

                        # # # Normalize the attention weights to range [0, 255]
                        # # attn_map = np.uint8(255 * attn_map / np.max(attn_map))
                        # attn_map = np.uint8(255 * (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map)))
                        # # attn_map = 1-attn_map

                        # # Apply a complementary colormap
                        # # cmap = plt.get_cmap('viridis')  # Example colormap
                        # # attn_map_colored = cmap(1 - attn_map)  # Invert color map

                        # # Convert to 8-bit image format if needed
                        # # heatmap = (attn_map_colored * 255).astype(np.uint8)
                        # # total_sum = np.sum(attn_map)
                        # # attn_map_normalized = attn_map / total_sum
                        # # attn_map_normalized = 255 * attn_map_normalized
                        # # attn_map = np.uint8(attn_map_normalized)
                        # # attn_map = np.uint8(255*attn_map)


                        # # # Apply a color map (e.g., JET color map)
                        # heatmap = cv2.applyColorMap(attn_map, cv2.COLORMAP_JET)
                        # # heatmap = np.flipud(heatmap)

                        # # # Display the heatmap using matplotlib
                        # plt.imshow(heatmap)
                        # plt.title("Attention Heatmap")
                        # plt.axis('off')
                        # plt.savefig(f"Attention_{task_id}__.png")

                        # image_np = data["obs"]["agentview_rgb"][0].permute(1, 2, 0).detach().cpu().numpy()
                        # if image_np.dtype != np.uint8:
                        #     image_np = np.uint8(255 * image_np / np.max(image_np))
                        #     # image_np = np.uint8(255 * (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np)))
                        # # print("image shape", image_np.shape)

                        # # heatmap = np.uint8(255 * attn_map / (np.max(attn_map) + 1e-8))
                        # # print("heatmap", heatmap.shape)
                        # # heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                        # # heatmap_colored = cv2.resize(heatmap_colored, (image_np.shape[1], image_np.shape[0]))

                        # # # # # Adjust blending ratio if needed
                        # overlayed_image = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)
                        # overlayed_image = np.flipud(overlayed_image)

                        # plt.figure(figsize=(10, 5))
                        # plt.imshow(overlayed_image)
                        # plt.title("Heat map")
                        # # file_path = os.path.join(directory, f"{img_name}_loaded_task{args.load_task}_on_{task}_{module}_.png")
                        # plt.savefig(f"Heatmap_{task_id}__.png")

                    # agent_view_f, eye_view_f, text_f, temporal_f = tsne_feats(data, algo.policy)
                    # tsne_features_agent.append(agent_view_f.cpu().numpy())
                    # tsne_features_eye.append(eye_view_f.cpu().numpy())
                    # tsne_features_text.append(text_f.cpu().numpy())
                    # tsne_features_temporal.append(temporal_f.cpu().numpy())

                    # means_reshaped = tsne_gmm(data, algo.policy)
                    # tsne_gmm_features.append(means_reshaped.cpu().numpy())

                    # tsne_labels.append([task_id])



                    # check whether succeed
                    for k in range(env_num):
                        dones[k] = dones[k] or done[k]
                    if all(dones):
                        break

                # attention_maps(data, data["task_emb"], 'agentview_rgb', algo.policy, task_id, args, 'image_encoder')
                # attention_maps(data, data["task_emb"], 'eye_in_hand_rgb', algo.policy, task_id, args, 'image_encoder')
                # attention_maps(data, data["task_emb"], 'agentview_rgb', algo.policy, task_id, args, 'temp_transformer')
                # attention_maps(data, data["task_emb"], 'eye_in_hand_rgb', algo.policy, task_id, args, 'temp_transformer')
                
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


    # Convert lists to numpy arrays
    # agent_features = np.concatenate(tsne_features_agent, axis=0)
    # eye_features = np.concatenate(tsne_features_eye, axis=0)
    # print("len text features", len(tsne_features_text))
    # text_features = np.concatenate(tsne_features_text, axis=0)
    # print("text features after concat", text_features.shape)
    # temporal_features = np.concatenate(tsne_features_temporal, axis=0)
    # gmm_features = np.concatenate(tsne_gmm_features, axis=0)
    # tsne_labels_ = np.array([np.full(env_num, element) for element in tsne_labels])
    # targets = np.concatenate(tsne_labels_, axis=0)


    # tsne = TSNE(n_components=2, random_state=0)
    # agent_features_tsne = tsne.fit_transform(agent_features)

    # tsne = TSNE(n_components=2, random_state=0)
    # eye_features_tsne = tsne.fit_transform(eye_features)

    # tsne = TSNE(n_components=2, random_state=0)
    # text_features_tsne = tsne.fit_transform(text_features)
    # print("text features for tsne", text_features_tsne.shape)

    # tsne = TSNE(n_components=2, random_state=0)
    # temporal_features_tsne = tsne.fit_transform(temporal_features)

    # tsne = TSNE(n_components=2, random_state=0)
    # gmm_features_tsne = tsne.fit_transform(gmm_features)

    # colormap = cm.get_cmap('tab10')
    # unique_targets = np.unique(targets)
    # colors = [colormap(i) for i in range(len(unique_targets))]

    # print("targets length:", len(targets))
    # print("agent_features_tsne length:", len(agent_features_tsne))

    # print("Min and Max of TSNE components:", agent_features_tsne.min(), agent_features_tsne.max())

    # plt.figure(figsize=(10, 8))
    # for i in range(len(unique_targets)):
    #     plt.scatter(agent_features_tsne[targets == unique_targets[i], 0], 
    #                 agent_features_tsne[targets == unique_targets[i], 1], 
    #                 color=colors[i], label=str(unique_targets[i]), s=10)


    # plt.legend()
    # # title = 't-SNE Task'+str(model.task-1)+'_epoch'+str(epoch)+' new classes'
    # plt.title('Agent view Encoder')
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    # file_path = os.path.join(directory, f'tsne_agent_before_projection_{args.algo}_loadtask_{str(args.load_task)}_{args.benchmark}.png')
    # plt.savefig(file_path)

    # plt.figure(figsize=(10, 8))
    # for i in range(len(unique_targets)):
    #     plt.scatter(eye_features_tsne[targets == unique_targets[i], 0], 
    #                 eye_features_tsne[targets == unique_targets[i], 1], 
    #                 color=colors[i], label=str(unique_targets[i]), s=10)


    # plt.legend()
    # # title = 't-SNE Task'+str(model.task-1)+'_epoch'+str(epoch)+' new classes'
    # plt.title('Eye view Encoder')
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    # file_path = os.path.join(directory, f'tsne_eye_before_projection_{args.algo}_loadtask_{str(args.load_task)}_{args.benchmark}.png')
    # plt.savefig(file_path)

    # plt.figure(figsize=(10, 8))
    # for i in range(len(unique_targets)):
    #     plt.scatter(text_features_tsne[targets == unique_targets[i], 0], 
    #                 text_features_tsne[targets == unique_targets[i], 1], 
    #                 color=colors[i], label=str(unique_targets[i]), s=10)


    # plt.legend()
    # # title = 't-SNE Task'+str(model.task-1)+'_epoch'+str(epoch)+' new classes'
    # plt.title('Language Descriptions')
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')

    # plt.xlim(-150, 150)  # Example limits; adjust based on your data
    # plt.ylim(-150, 150)  # Example limits; adjust based on your data

    # file_path = os.path.join(directory, f'tsne_language_{args.algo}_loadtask_{str(args.load_task)}_{args.benchmark}__.png')
    # plt.savefig(file_path)


    # plt.figure(figsize=(10, 8))
    # for i in range(len(unique_targets)):
    #     plt.scatter(temporal_features_tsne[targets == unique_targets[i], 0], 
    #                 temporal_features_tsne[targets == unique_targets[i], 1], 
    #                 color=colors[i], label=str(unique_targets[i]), s=10)


    # plt.legend()
    # # title = 't-SNE Task'+str(model.task-1)+'_epoch'+str(epoch)+' new classes'
    # plt.title('Temporal Transformer')
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    # file_path = os.path.join(directory, f'tsne_temporal_transformer_{args.algo}_loadtask_{str(args.load_task)}_{args.benchmark}.png')
    # plt.savefig(file_path)

    # plt.figure(figsize=(10, 8))
    # for i in range(len(unique_targets)):
    #     plt.scatter(gmm_features_tsne[targets == unique_targets[i], 0], 
    #                 gmm_features_tsne[targets == unique_targets[i], 1], 
    #                 color=colors[i], label=str(unique_targets[i]), s=10)


    # plt.legend()
    # # title = 't-SNE Task'+str(model.task-1)+'_epoch'+str(epoch)+' new classes'
    # plt.title('GMM')
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    # file_path = os.path.join(directory, f'gmm_{args.algo}_loadtask_{str(args.load_task)}_{args.benchmark}.png')
    # plt.savefig(file_path)


def tsne_feats(data, policy):
    agent_view_x = data["obs"]['agentview_rgb']
    eye_view_x = data['obs']['eye_in_hand_rgb']
    agent_view_f = policy.image_encoders['agentview_rgb']['encoder'](agent_view_x, returnt='feats', langs=data["task_emb"])
    eye_view_f = policy.image_encoders['eye_in_hand_rgb']['encoder'](eye_view_x, returnt='feats', langs=data["task_emb"])

    task_emb = policy.language_encoder(data)
    # print("task_emb", task_emb.shape)
    
    b_dim = agent_view_f.shape[0]

    data = policy.preprocess_input(data, train_mode=False)
    temporal_f = policy(data, returnt='feats')
    temporal_f = torch.squeeze(temporal_f)
    

    # agent_view_f = agent_view_f.squeeze(0)
    # eye_view_f = eye_view_f.squeeze(0)

    # flattened_agent_tensor = agent_view_f.view(-1)  # Shape: [128 * 16 * 16] = [32768] 
    # flattened_eye_tensor = eye_view_f.view(-1)  # Shape: [128 * 16 * 16] = [32768] 

    # flattened_agent_tensor_np = flattened_agent_tensor.detach().cpu().numpy().reshape(1, -1)  # Shape: [1, 32768]
    # flattened_eye_tensor_np = flattened_eye_tensor.detach().cpu().numpy().reshape(1, -1)


    flattened_agent_tensor = agent_view_f.view(b_dim, -1) # Shape: [B, 32768]
    flattened_eye_tensor = eye_view_f.view(b_dim, -1)  # Shape: [B, 32768]

    return flattened_agent_tensor, flattened_eye_tensor, task_emb, temporal_f
    # return flattened_agent_tensor_np, flattened_eye_tensor_np


def tsne_gmm(data, policy):
    policy.eval()
    data = policy.preprocess_input(data, train_mode=False)
    x = policy.spatial_encode(data)
    x = policy.temporal_encode(x)
    # print("temp x", x.shape)

    # Forward pass through GMMHead
    # x is the input tensor
    means, stds, logits = policy.policy_head.forward_fn(x)
    # print("means shape", means.shape)


    # t-SNE on the means of the GMM components
    means_reshaped = means.view(-1, policy.policy_head.num_modes * policy.policy_head.output_size)
    # print("means rehspaed", means_reshaped.shape)

    return means_reshaped
    
    # tsne = TSNE(n_components=2, random_state=0)
    # tsne_results = tsne.fit_transform(means_reshaped.detach().cpu().numpy())

    # # Plotting t-SNE results
    # plt.figure(figsize=(10, 8))
    # plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=logits.detach().cpu().numpy().argmax(axis=1), cmap='tab10')
    # plt.title('t-SNE plot of GMM components')
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    # plt.colorbar()
    # file_path = os.path.join(directory, f'tsne_temporal_transformer_{args.algo}_loadtask_{str(args.load_task)}_{args.benchmark}.png')
    # plt.savefig(file_path)    




    
# def attention_maps(data, policy):
#     agent_view_x = data["obs"]['agentview_rgb'][0]
#     eye_view_x = data['obs']['eye_in_hand_rgb'][0]
#     agent_view_f = policy.image_encoders['agentview_rgb']['encoder'](agent_view_x, returnt='feats', langs=data["task_emb"])
#     eye_view_f = policy.image_encoders['eye_in_hand_rgb']['encoder'](eye_view_x, returnt='feats', langs=data["task_emb"])

#     agent_view_heatmap = torch.mean(agent_view_f, dim=1).squeeze()  # Average along channels
#     agent_view_heatmap = agent_view_heatmap.detach().cpu().numpy()

#     eye_view_heatmap = torch.mean(eye_view_f, dim=1).squeeze()  # Average along channels
#     eye_view_heatmap = eye_view_heatmap.detach().cpu().numpy()

#     agent_view_heatmap = cv2.resize(agent_view_heatmap, (agent_view_x.shape[2], agent_view_x.shape[1]))  # Resize to input image size
#     eye_view_heatmap = cv2.resize(eye_view_heatmap, (eye_view_x.shape[2], eye_view_x.shape[1]))  # Resize to input image size

#     agent_view_heatmap = np.uint8(255 * agent_view_heatmap)  # Normalize to 0-255
#     agent_view_heatmap = cv2.applyColorMap(agent_view_heatmap, cv2.COLORMAP_JET)  # Apply a color map

#     overlayed_agent_image = cv2.addWeighted(agent_view_x, 0.6, agent_view_heatmap, 0.4, 0)  # Blend heatmap with the original image

#     eye_view_heatmap = np.uint8(255 * eye_view_heatmap)  # Normalize to 0-255
#     eye_view_heatmap = cv2.applyColorMap(eye_view_heatmap, cv2.COLORMAP_JET)  # Apply a color map

#     overlayed_eye_image = cv2.addWeighted(eye_view_x, 0.6, eye_view_heatmap, 0.4, 0)  # Blend heatmap with the original image

#     # Create a new figure
#     plt.figure(figsize=(10, 5))  # Set the figure size

#     # Plot the first image in the first subplot
#     plt.subplot(1, 2, 1)  # (1 row, 2 columns, 1st subplot)
#     plt.imshow(overlayed_agent_image)
#     plt.title("Agent View Image")
#     plt.axis('off')  # Turn off axis labels

#     # Plot the second image in the second subplot
#     plt.subplot(1, 2, 2)  # (1 row, 2 columns, 2nd subplot)
#     plt.imshow(overlayed_eye_image)
#     plt.title("Eye View Image")
#     plt.axis('off')  # Turn off axis labels

#     # Show the figure
#     plt.show()

# # def a ttention_maps(data, text, img_name, policy, task, args, module):

#     image = data['obs'][img_name]

#     if module == 'image_encoder':
#         features = policy.image_encoders[img_name]['encoder'](image, returnt='feats', langs=text)

#         # Average along channels to get a single-channel heatmap
#         heatmap = torch.mean(features[0], dim=0).squeeze()  # Average along channels
#         heatmap = heatmap.detach().cpu().numpy()

#         # Resize the heatmap to match the original image size
#         heatmap = cv2.resize(heatmap, (image.shape[3], image.shape[2]))  # Resize to (width, height)

#     else:
#         data = policy.preprocess_input(data, train_mode=False)
#         temporal_f = policy(data, returnt='feats')
#         print("layers in tranformer", len(policy.temporal_transformer.layers))
#         attn_weights = policy.temporal_transformer.attention_output[3].mean(dim=1)
#         heatmap = attn_weights[0]
#         heatmap = heatmap.detach().cpu().numpy()

#     # Convert image to numpy array and ensure it's in the correct format
#     image_np = image[0].permute(1, 2, 0).detach().cpu().numpy()  # Convert to (height, width, channels)
    
#     # Normalize the image to 0-255 if it's not already in uint8 format
#     if image_np.dtype != np.uint8:
#         image_np = np.uint8(255 * image_np / np.max(image_np))
    
#     # Normalize and apply color map to the heatmap
#     heatmap = np.uint8(255 * heatmap / np.max(heatmap))
#     heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply a color map

#     # Resize heatmap_colored to match the original image dimensions
#     heatmap_colored = cv2.resize(heatmap_colored, (image_np.shape[1], image_np.shape[0]))  # Resize to (width, height)

#     # Blend the heatmap with the original image
#     overlayed_image = cv2.addWeighted(image_np, 0.6, heatmap_colored, 0.4, 0)  # Blend heatmap with the original image
#     overlayed_image = np.flipud(overlayed_image)
#     # Plot and save the overlayed image
#     plt.figure(figsize=(10, 5))  # Set the figure size
#     plt.imshow(overlayed_image)
#     plt.title(img_name)
#     plt.savefig(f"{img_name}_loaded_task{args.load_task}_on_{task}_{module}_.png")
    
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
