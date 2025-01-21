import time
import uuid
import json
import numpy as np

from pathlib import Path
from a_star import A_star
from opti_process import Opti_Process

def baseline_start_end(start, goal, num_points):
    return np.linspace(start, goal, num_points + 2)

def main(config):
    """
    Parameters:

    ==========================================================
    cutting_plane_factor [1, X]: Length of the cutting planes
    ------------------
    A factor that multiplies the average distance,
    which is calculated from the individual segments,
    by the found baseline path whether A*, baseline approach Start-Goal etc.
    In pre-processing, excessively long cutting planes should be adjusted to the size of the map
    ==========================================================

    ==========================================================
    epsilon [0, X]: Downsampling size
    ------------------
    The maximum distance tolerance for simplification.
    Epsilon = 0 No simplification will occur.
    Larger Epsilon value will result in a more simplified line, with fewer points.
    ==========================================================

    ==========================================================
    tau [0.1, 1]: Smoothness of the clothoids
    ------------------
    Highly smooth curves with tight bends, smaller tau.
    Straighter curves with less curvature, larger tau.
    Highly smooth curves: 0.1 to 0.3
    Moderately smooth curves: 0.3 to 0.5
    Less smooth curves: 0.5 to 1.0
    ==========================================================

    ==========================================================
    pop_size: Population size
    ==========================================================

    ==========================================================
    n_gen: Number of generations
    ==========================================================

    ==========================================================
    algo: CMAES | DE | PSO | GA?
    ==========================================================

    ==========================================================
    Note: Baseline approach
    num_points [5, X] - specifies how many control points will be set for optimization
    similar to epsilon.
    ==========================================================
    """

    run_name = config["run_name"]
    
    astar = A_star()

    ## Load the heightmap
    
    heightmap = np.load(config["heightmap"])
    heightmap = astar.downsample_heightmap(heightmap, config["res"][0], config["res"][1])

    # Define start and goal points
    start = config["start"]
    goal = config["goal"]

    ## Run A*

    start_time = time.time()
    path1 = astar.astar_heightmap(
        heightmap, start, goal, astar.euclidean_distance, astar.height_difference_cost
    )
    end_time = time.time()
    print(f"Time taken to find the path: {end_time - start_time:.4f} seconds")

    ### Run A* path optimization
    
    save_folder = Path(f"results/{run_name}/")
    save_folder.mkdir(exist_ok=True, parents=True)

    optiprocess = Opti_Process(path1, heightmap, basepath_approach="A*", save_path=f"results/{run_name}/", seed=config["seed"])
    optiprocess.pre_process_astar_path(
        cutting_plane_factor=config["cutting_plane_factor"], epsilon=config["epsilon"]
    )
    optiprocess.run_optimization(
        algo=config["algo"],
        pop_size=config["pop_size"],
        gen_size=config["gen_size"],
        tau=config["tau"],
    )
    optiprocess.plot_fitness_progress(save=False, vis=False)
    optiprocess.post_process_path(save=False, vis=False)
    optiprocess.plot_result_path(
        PATH=f"./results/{run_name}/path_plot.png", save=True, vis=False
    )
    fit_orig, fit_optim = optiprocess.compare_paths()

    with open(f"./results/{run_name}/results.json", 'w') as file:
        json.dump({
            "fitness_A*_path": fit_orig,
            "fitness_A*_optimised_path": fit_optim
        }, file, indent=4)

    print(f"[INFO] Results saved.", flush=True)


if __name__ == "__main__":
    """
    Definition of several configuration examples and main configurations for the optimization algorithm.
    """
    
    conf_example = {
        "slovenia_example": {
            "heightmap": "./heightmaps/slovenia_2000_6000_heightmap_wo_norm.npy",
            "start" : (175,0),
            "goal": (160, 290),
            "res": (200, 400)
        },
        "austria_example":{
            "heightmap": "./heightmaps/austria_2000_6000_heightmap_wo_norm.npy",
            "start" : [50, 140],
            "goal": [140, 460],
            "res": [200, 600]
        },
        "italy_example":{
            "heightmap": "./heightmaps/italia_4000_6000_heightmap_wo_norm.npy",
            "start" : [130, 160],
            "goal": [225, 425],
            "res": [400, 600]
        },
        "opti_algos": {
           0: "CMAES",
           1: "PSO",
           2: "DE"
        }
    }

    main_conf = {
        "heightmap": conf_example["slovenia_example"]["heightmap"],
        "res": conf_example["slovenia_example"]["res"],
        "start": conf_example["slovenia_example"]["start"],
        "goal": conf_example["slovenia_example"]["goal"],
        "algo": conf_example["opti_algos"][0],
        "cutting_plane_factor": 1,
        "epsilon": 1,
        "pop_size": 60,
        "gen_size": 50,
        "tau": 0.4,
        "seed": 10,
        "run_name": uuid.uuid4().hex
    }
    
    main(main_conf)