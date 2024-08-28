#!/bin/bash
#SBATCH --job-name=lib-cl
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128g
#SBATCH --account=OD-236362
#SBATCH --cpus-per-task=8

# # Get the GPU ID allocated by SLURM
# GPU_ID=$(echo $SLURM_JOB_GPUS | cut -d',' -f1)

# # Set the environment variables for CUDA
# export CUDA_VISIBLE_DEVICES=$GPU_ID
# export MUJOCO_EGL_DEVICE_ID=$GPU_ID

# # Debugging to ensure correct device ID
# echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# echo "MUJOCO_EGL_DEVICE_ID: $MUJOCO_EGL_DEVICE_ID"


# python libero/lifelong/main.py benchmark_name=LIBERO_SPATIAL lifelong=er seed=100
python libero/lifelong/main.py benchmark_name=LIBERO_OBJECT lifelong=base seed=200 policy=bc_transformer_policy_mheads

# python libero/lifelong/evaluate.py --benchmark libero_spatial --task_id 9 --algo er --policy bc_transformer_policy --seed 10000 --load_task 0 --device_id 0

# python libero/lifelong/evaluate_wtsne.py --benchmark libero_spatial --task_id 0 --algo er --policy bc_transformer_policy --seed 10000 \
#     --load_task 0 --device_id 0 --folder 1 --save_videos

# python libero/lifelong/evaluate_attn.py --benchmark libero_object --task_id 0 --algo er --policy bc_transformer_policy --seed 10000 \
#     --load_task 4 --device_id 0 --folder 5 --save-videos
