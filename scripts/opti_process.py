import numpy as np
import matplotlib.pyplot as plt

from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

from opti_problem import GetHistory, PathProblem
from clothoids.create_clothoids import clothoids_from_control_points

class Opti_Process(object):
    def __init__(self, path, heightmap: np.ndarray, basepath_approach: str, save_path, seed: int):
        self.arr_len = None
        self.arr_xl = None
        self.arr_xu = None
        self.avg_size = None
        self.res = None
        self.optimized_path = None
        self.tau = None
        self.cutting_plane_factor = None
        self.algo = None
        self.basepath_approach = basepath_approach
        self.path_to_optimize = path
        self.heightmap = heightmap
        self.problem = None
        
        self.width = heightmap.shape[1]
        self.height = heightmap.shape[0]
        self.save_path = save_path
        self.seed = seed
    
    def initialize(self, algo: str, pop_size: int, gen_size: int, tau: float):
        self.tau = tau
        self.algo = algo

        # Create the problem object
        problem = PathProblem(self.arr_len, self.arr_xl, self.arr_xu, self.path_to_optimize, self.heightmap, tau)
        self.problem = problem

    def run_optimization(self, algo: str, pop_size: int, gen_size: int, tau: float):
        """
        Runs an optimization algorithm on a given problem instance.

        Args:
            algo: The name of the optimization algorithm to use (e.g., "CMAES", "DE", "PSO").
            pop_size: The size of the population used in the optimization algorithm.
            gen_size: The number of generations to run the optimization for.
            tau: A parameter used in the optimization algorithm.

        Returns:
            None
        """

        ### add 1
        # compute clothoids_from_control_points
        c_np = clothoids_from_control_points(self.path_to_optimize, tau=tau)
        # save file 
        np.save(self.save_path + "astar_path.npy", c_np)
        ### 

        self.initialize(algo, pop_size, gen_size, tau)
        # self.tau = tau
        # self.algo = algo

        # # Create the problem object
        # problem = PathProblem(self.arr_len, self.arr_xl, self.arr_xu, self.path_to_optimize, self.heightmap, tau)
        # self.problem = problem

        if algo == "CMAES":
            algorithm = CMAES(restarts=20, pop_size = pop_size, restart_from_best=True)
        elif algo == "DE":
            algorithm = DE(restarts=20, pop_size = pop_size, restart_from_best=True)
        elif algo == "PSO":
            algorithm = PSO(restarts=20, pop_size = pop_size, restart_from_best=True)
        elif algo == "GA":
            algorithm = GA(restarts=20, pop_size = pop_size, restart_from_best=True)
        else:
            SystemExit("Wrong choice of algorithm!")

        self.res = minimize(
            self.problem,
            algorithm,
            ('n_gen', gen_size),
            seed=self.seed,
            verbose=True,
            callback=GetHistory(),
            save_history=True
        )

        print("Best solution found: F = %s" % (abs(self.res.F)), flush=True)

    def point_line_distance(self, point: list, start: list, end: list) -> float:
        """
        Calculates the perpendicular distance between a point and a line segment in 2D space.

        Args:
            point: A list representing the coordinates of the point (x, y).
            start: A list representing the coordinates of the starting point of the line segment (x1, y1).
            end: A list representing the coordinates of the ending point of the line segment (x2, y2).

        Returns:
            The perpendicular distance between the point and the line segment.
        """
        if start == end:
            return np.sqrt((point[0] - start[0])**2 + (point[1] - start[1])**2)
        
        n = np.abs((end[0] - start[0]) * (start[1] - point[1]) - (start[0] - point[0]) * (end[1] - start[1]))
        d = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        return n / d

    def douglas_peucker(self, points: list, epsilon: float):
        """
        Implements the Douglas-Peucker algorithm for line simplification.

        Args:
            points: A list of points (x, y) representing the original line.
            epsilon: The maximum distance tolerance for simplification.

        Returns:
            A simplified list of points representing the line.
        """
        dmax = 0
        index = 0
        for i in range(1, len(points) - 1):
            d = self.point_line_distance(points[i], points[0], points[-1])
            if d > dmax:
                index = i
                dmax = d

        if dmax > epsilon:
            results = self.douglas_peucker(points[:index+1], epsilon) + self.douglas_peucker(points[index:], epsilon)[1:]
        else:
            results = [points[0], points[-1]]

        return results
    
    def pre_process_astar_path(self, cutting_plane_factor: int, epsilon: float):
        """
        Pre-processes a path for A* pathfinding by simplifying it and generating bounding constraints.

        Args:
            cutting_plane_factor: A factor that determines the distance of the cutting planes from the path.
            epsilon: The maximum distance tolerance for the Douglas-Peucker simplification.

        Returns:
            None
        """
        original_path = self.path_to_optimize
        self.cutting_plane_factor = cutting_plane_factor
        
        # This parameter affects the degree of simplification
        simplified_points = self.douglas_peucker(original_path, epsilon)
        print(f"Number of original points: {len(original_path)} -> number of downsampled points: {len(simplified_points)}", flush=True)
        
        low_res_path = simplified_points
        self.path_to_optimize = np.array(low_res_path)

        arr_xl = []
        arr_xu = []
        sizes = []
        heightmap = self.heightmap

        for i in range(len(low_res_path)-2):
            A = np.array(low_res_path[i])
            B = np.array(low_res_path[i+1])
            AB = B - A
            size_AB = np.linalg.norm(AB)
            sizes.append(size_AB)

        self.avg_size = np.mean(sizes)

        # Note: Not optimized version yet, hovewer works
        for i in range(len(low_res_path)-2):
            A = np.array(low_res_path[i])
            B = np.array(low_res_path[i+1])
            AB = B - A
            
            size_AB = self.avg_size * cutting_plane_factor
            # normal vector (perpendicular to AB)
            normal_vector = np.array([-AB[1], AB[0]])
            normal_vector = normal_vector / np.linalg.norm(normal_vector)
            
            end_point_negative = B - size_AB * normal_vector
            end_point_positive = B + size_AB * normal_vector

            if end_point_negative[0] > heightmap.shape[0]:
                t = (heightmap.shape[0] - B[0]) / normal_vector[0]
                x = B[1] + t * normal_vector[1]
                C = [heightmap.shape[0],x]
                BC = B - C
                size_AB_negative = np.linalg.norm(BC)
            else:
                size_AB_negative = size_AB
            
            if end_point_positive[0] < 0:
                t = (0 - B[0]) / normal_vector[0]
                x = B[1] + t * normal_vector[1]
                C = [0,x]
                BC = B - C
                size_AB_positive = np.linalg.norm(BC)
            else:
                size_AB_positive = size_AB

            end_point_negative = B - size_AB_negative * normal_vector
            end_point_positive = B + size_AB_positive * normal_vector

            if end_point_positive[1] > heightmap.shape[1]:
                t = (heightmap.shape[1] - B[1]) / normal_vector[1]
                y = B[0] + t * normal_vector[0]
                C = [y,heightmap.shape[1]]
                BC = B - C
                size_AB_positive = np.linalg.norm(BC)

            elif end_point_negative[1] > heightmap.shape[1]:
                t = (heightmap.shape[1] - B[1]) / normal_vector[1]
                y = B[0] + t * normal_vector[0]
                C = [y,heightmap.shape[1]]
                BC = B - C
                size_AB_negative = np.linalg.norm(BC)

            if end_point_positive[1] < 0:
                t = (0 - B[1]) / normal_vector[1]
                y = B[0] + t * normal_vector[0]
                C = [y,0]
                BC = B - C
                size_AB_positive = np.linalg.norm(BC)

            elif end_point_negative[1] < 0:
                t = (0 - B[1]) / normal_vector[1]
                y = B[0] + t * normal_vector[0]
                C = [y,0]
                BC = B - C
                size_AB_negative = np.linalg.norm(BC)
            
            arr_xl.append(-size_AB_negative)
            arr_xu.append(size_AB_positive)

        self.arr_xl = np.array(arr_xl)
        self.arr_xu = np.array(arr_xu)
        self.arr_len = self.arr_xl.size
    
    def pre_process_baseline_path(self, cutting_plane_factor: int):
        """
        Pre-processes a baseline approach of pathfinding by generating bounding constraints.

        Args:
            cutting_plane_factor: A factor that determines the distance of the cutting planes from the path.

        Returns:
            None
        """

        self.cutting_plane_factor = cutting_plane_factor

        arr_xl = []
        arr_xu = []
        sizes = []

        path = self.path_to_optimize
        heightmap = self.heightmap

        for i in range(len(path)-2):
            A = np.array(path[i])
            B = np.array(path[i+1])
            AB = B - A
            size_AB = np.linalg.norm(AB)
            sizes.append(size_AB)

        self.avg_size = np.mean(sizes)

        # Note: Not optimized version yet, hovewer works
        for i in range(len(path)-2):
            A = np.array(path[i])
            B = np.array(path[i+1])
            AB = B - A
            
            size_AB = self.avg_size * cutting_plane_factor
            # normal vector (perpendicular to AB)
            normal_vector = np.array([-AB[1], AB[0]])
            normal_vector = normal_vector / np.linalg.norm(normal_vector)
            
            end_point_negative = B - size_AB * normal_vector
            end_point_positive = B + size_AB * normal_vector

            if end_point_negative[0] > heightmap.shape[0]:
                t = (heightmap.shape[0] - B[0]) / normal_vector[0]
                x = B[1] + t * normal_vector[1]
                C = [heightmap.shape[0],x]
                BC = B - C
                size_AB_negative = np.linalg.norm(BC)
            else:
                size_AB_negative = size_AB
            
            if end_point_positive[0] < 0:
                t = (0 - B[0]) / normal_vector[0]
                x = B[1] + t * normal_vector[1]
                C = [0,x]
                BC = B - C
                size_AB_positive = np.linalg.norm(BC)
            else:
                size_AB_positive = size_AB

            end_point_negative = B - size_AB_negative * normal_vector
            end_point_positive = B + size_AB_positive * normal_vector

            if end_point_positive[1] > heightmap.shape[1]:
                t = (heightmap.shape[1] - B[1]) / normal_vector[1]
                y = B[0] + t * normal_vector[0]
                C = [y,heightmap.shape[1]]
                BC = B - C
                size_AB_positive = np.linalg.norm(BC)

            elif end_point_negative[1] > heightmap.shape[1]:
                t = (heightmap.shape[1] - B[1]) / normal_vector[1]
                y = B[0] + t * normal_vector[0]
                C = [y,heightmap.shape[1]]
                BC = B - C
                size_AB_negative = np.linalg.norm(BC)

            if end_point_positive[1] < 0:
                t = (0 - B[1]) / normal_vector[1]
                y = B[0] + t * normal_vector[0]
                C = [y,0]
                BC = B - C
                size_AB_positive = np.linalg.norm(BC)

            elif end_point_negative[1] < 0:
                t = (0 - B[1]) / normal_vector[1]
                y = B[0] + t * normal_vector[0]
                C = [y,0]
                BC = B - C
                size_AB_negative = np.linalg.norm(BC)
        
            arr_xl.append(-size_AB_negative)
            arr_xu.append(size_AB_positive)

        self.arr_xl = np.array(arr_xl)
        self.arr_xu = np.array(arr_xu)
        self.arr_len = self.arr_xl.size
    
    def find_perpendicular_points(self, A: list, B: list, t_point: list, num_points=3):
        """
        Finds points along a line perpendicular to the line segment AB, passing through the point t_point.

        Args:
            A: The starting point of the line segment.
            B: The ending point of the line segment.
            t_point: The parameter t for the point on the line segment AB.
            num_points: The number of points to generate on the perpendicular line.

        Returns:
            A tuple containing:
                - A list of points on the perpendicular line.
                - The point on the perpendicular line corresponding to the given t_point.
        """

        AB = B - A

        heightmap = self.heightmap

        size_AB = self.avg_size * self.cutting_plane_factor
        normal_vector = np.array([-AB[1], AB[0]])
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        
        # Note: Not optimized version yet, hovewer works
        end_point_negative = B - size_AB * normal_vector
        end_point_positive = B + size_AB * normal_vector

        if end_point_negative[0] > heightmap.shape[0]:
            t = (heightmap.shape[0] - B[0]) / normal_vector[0]
            x = B[1] + t * normal_vector[1]
            C = [heightmap.shape[0],x]
            BC = B - C
            size_AB_negative = np.linalg.norm(BC)
        else:
            size_AB_negative = size_AB
        
        if end_point_positive[0] < 0:
            t = (0 - B[0]) / normal_vector[0]
            x = B[1] + t * normal_vector[1]
            C = [0,x]
            BC = B - C
            size_AB_positive = np.linalg.norm(BC)
        else:
            size_AB_positive = size_AB

        end_point_negative = B - size_AB_negative * normal_vector
        end_point_positive = B + size_AB_positive * normal_vector

        if end_point_positive[1] > heightmap.shape[1]:
            t = (heightmap.shape[1] - B[1]) / normal_vector[1]
            y = B[0] + t * normal_vector[0]
            C = [y,heightmap.shape[1]]
            BC = B - C
            size_AB_positive = np.linalg.norm(BC)

        elif end_point_negative[1] > heightmap.shape[1]:
            t = (heightmap.shape[1] - B[1]) / normal_vector[1]
            y = B[0] + t * normal_vector[0]
            C = [y,heightmap.shape[1]]
            BC = B - C
            size_AB_negative = np.linalg.norm(BC)

        if end_point_positive[1] < 0:
            t = (0 - B[1]) / normal_vector[1]
            y = B[0] + t * normal_vector[0]
            C = [y,0]
            BC = B - C
            size_AB_positive = np.linalg.norm(BC)

        elif end_point_negative[1] < 0:
            t = (0 - B[1]) / normal_vector[1]
            y = B[0] + t * normal_vector[0]
            C = [y,0]
            BC = B - C
            size_AB_negative = np.linalg.norm(BC)
                
        # Parametric line: X = B + t * normal_vector
        # t for points
        t_points = np.linspace(-size_AB_negative, size_AB_positive, num_points)
        points = B + t_points[:, np.newaxis] * normal_vector
        # t for search point
        point = B + t_point * normal_vector
        
        return points, point

    def find_perpendicular_point(self, A, B, t_point):
        """
        Finds a point on a line perpendicular to the line segment AB, passing through the point t_point.

        Args:
            A: The starting point of the line segment.
            B: The ending point of the line segment.
            t_point: The parameter t for the point on the line segment AB.

        Returns:
            The coordinates of the point on the perpendicular line.
        """
        # direction vector AB
        AB = B - A
        # normal vector (perpendicular to AB)
        normal_vector = np.array([-AB[1], AB[0]])
        normal_vector = normal_vector / np.linalg.norm(normal_vector)

        new_point = B + t_point * normal_vector
        return  np.round(new_point)
    
    def clip_point(self, new_point):
        """
        Clips a point to ensure it stays within the bounds of a given heightmap.

        Args:
            new_point: The coordinates of the point to clip.

        Returns:
            The clipped coordinates.
        """
        return [int(np.clip(new_point[0], 0, self.height-1)), int(np.clip(new_point[1], 0, self.width-1))]
        
    def post_process_path(self, save=False, vis=False):
        """
        Post-processes the optimized path obtained from the optimization algorithm.

        Args:
            save: A flag indicating whether to save the visualization as a PNG image (default: False).
            vis: A flag indicating whether to display the visualization (default: False).

        Returns:
            None
        """
        data = self.res.X
        path = self.path_to_optimize
        
        # add start point 
        new_path = [path[0]]
        # compute the rest of the points
        for i in range(len(path)-2):
            A = np.array(path[i])
            B = np.array(path[i+1])
            t_point = data[i]
            new_point = self.find_perpendicular_point(A, B, t_point)
            new_point = self.clip_point(new_point)
            new_path.append(new_point)

        new_path.append(path[-1])
        new_path = np.array(new_path)

        c_np_o = clothoids_from_control_points(new_path, tau=self.tau)

        ### add 2
        # compute clothoids_from_control_points
        # save file
        np.save(self.save_path + "optimized_astar_path.npy", c_np_o)
        ### 


    def plot_fitness_progress(self, PATH=None, save=False, vis=False):
        """
        Plots the fitness score progress across generations.

        Args:
            PATH (str, optional): The file path where the plot will be saved. Defaults to None.
            save (bool, optional): If True, saves the plot to a file. Defaults to False.
            vis (bool, optional): If True, displays the plot. Defaults to False.

        Returns:
            None
        """
        val_avg = self.res.algorithm.callback.data["avg"]
        val_min = self.res.algorithm.callback.data["best"]

        # Create the plot with improved formatting and customization
        plt.figure(figsize=(8, 5))

        ax = plt.axes()

        # Ensure x-axis starts from 0
        plt.xlim(0, len(val_avg))

        # Add grid lines for better readability
        plt.grid(True)

        # Set a descriptive title
        plt.title(f"Fitness Score Across Generations {self.algo}", fontsize=16)

        # Plot the fitness values with red color and solid line style
        ax.plot(np.arange(len(val_avg)), val_avg, color='red', linestyle='-', label="f. avg" )
        ax.plot(np.arange(len(val_min)), val_min, color='blue', linestyle='-',label="f. min")

        plt.legend()
        plt.xlabel("Generations", fontsize=14)
        plt.ylabel("Fitness Score", fontsize=14)
        plt.xticks(np.arange(len(val_avg))) 
        # Adjust tick label font size for better readability
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()

        if save:
            if PATH:
                plt.savefig(PATH)

        if vis:
            plt.show()

    def plot_result_path(self, PATH=None, save=False, vis=False, path_to_plot=None):
        """
        Plots the resulting path on a terrain map.

        Args:
            PATH (str, optional): The file path where the plot will be saved. Defaults to None.
            save (bool, optional): If True, saves the plot to a file. Defaults to False.
            vis (bool, optional): If True, displays the plot. Defaults to False.
            path_to_plot (list or np.array, optional): The path to be plotted. If None, uses the default path. Defaults to None.

        Returns:
            None
        """
        if path_to_plot is None:
            path = self.path_to_optimize
        else: 
            path = path_to_plot

        # Create a new figure with a specific size
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot the heightmap
        im = ax.imshow(self.heightmap, cmap='gist_earth', interpolation='bilinear', vmax=1000)
        ax.plot(path[:,1], path[:,0], linestyle='-', color='white', linewidth=2,
            markersize=6, markeredgecolor='darkblue', markerfacecolor='lightblue',
            label=f'{self.basepath_approach} control points path')

        plt.plot(path[:, 1], path[:, 0], linestyle='-', color='red', label=f"{self.algo} clothoids path", linewidth=3)

        # Set title and labels
        ax.set_title('Terrain map with resulting paths', fontsize=16, fontweight='bold')
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)

        # Remove ticks but keep labels
        ax.tick_params(axis='both', which='both', bottom=False, top=False,
                        labelbottom=True, left=False, right=False, labelleft=True)

        # Add a grid for better readability
        ax.grid(color='white', linestyle='-', linewidth=0.5, alpha=0.3)

        # Add a horizontal colorbar below the plot
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', aspect=30, pad=0.08)
        cbar.set_label('X Coordinate', fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        # Add legend
        ax.legend(loc='upper right', fontsize=10)

        if save:
            if PATH:
                plt.savefig(PATH)

        if vis:
            plt.show()

    def compare_paths(self):
        """
        Compares the original path with the optimized path based on various criteria.

        Returns:
            None
        """
        path_base_comp = np.array(self.path_to_optimize)
        path_base_comp = clothoids_from_control_points(path_base_comp, tau=self.tau)
        diff = self.problem.check_clothoid_radius_constraint(path_base_comp, 1/self.problem.curvature_radius)
        path_length = self.problem._compute_path_length(path_base_comp)

        print("\nCompare origin path and optimized one:", flush=True)
        print(f"Fitness of origin path {self.basepath_approach} : {path_length}", flush=True)
        print(f"Diff for {self.basepath_approach} : {diff}", flush=True)
        print(f"Diff*1000 for {self.basepath_approach} : {diff*1000}", flush=True)
        print(f"Fitness with curve diff calculated by problem for {self.basepath_approach} : {path_length + diff * 1000}", flush=True)
        print(f"Fitness calculated by problem for {self.basepath_approach} : {path_length}", flush=True)
        print(f"Fitness of optimized path {self.algo}: {self.res.F[0]}", flush=True)

        return path_length, self.res.F[0]
