from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv, DemoRenderEnv
# from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
import os
from libero.libero.utils.time_utils import Timer
from libero.libero.utils.video_utils import VideoWriter
# import init_path
from libero.libero import benchmark, get_libero_path
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
import multiprocessing


import h5py

if multiprocessing.get_start_method(allow_none=True) != "spawn":  
        multiprocessing.set_start_method("spawn", force=True)

benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_object" # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()

# retrieve a specific task
task_id = 2
task = task_suite.get_task(task_id)
task_name = task.name
print("the task name is:", task_name)
task_description = task.language
# print("bddl file", get_libero_path("bddl_files"))
bddl_file_path = '/scratch3/dis023/LIBERO/libero/libero/bddl_files'
task_bddl_file = os.path.join(bddl_file_path, task.problem_folder, task.bddl_file)
print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
      f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")


video_folder_agent = f"{task_suite_name}_{task_id}_demo_40_videos"

with Timer() as t, VideoWriter(video_folder_agent, save_video=True) as video_writer:

    # step over the environment
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 128,
        "camera_widths": 128,
        "has_renderer": True,
        "has_offscreen_renderer": False,
    }
    env = DemoRenderEnv(**env_args)
    # print(env)
    # env_num = 1  #20
    # env = SubprocVectorEnv(
    #     [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
    # )

    # env = VisualizationWrapper(env.env)

    env.reset()
    env.seed(0)
    init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
    init_state_id = [40]
    print("init states is:", init_states.shape)
    env.set_init_state(init_states[init_state_id])
    # env.sim.set_state_from_flattened(init_states[init_state_id])
    print("env render", env.renderer)
    print("the task_bddl_file is", task_bddl_file)


    dataset_path = "/datasets/work/d61-csirorobotics/source/LIBERO/libero_object/pick_up_the_salad_dressing_and_place_it_in_the_basket_demo.hdf5"

    data = h5py.File(dataset_path)["data"]["demo_40"]["actions"]
    print("data is:", data.shape)

    for action in data:
        obs, reward, done, info = env.step(action)
        env.render()
        # video_writer.append_vector_obs(
        #         obs, done, camera_name="agentview_image"
        #     )

    env.close()