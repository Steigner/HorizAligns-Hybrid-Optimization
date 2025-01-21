from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from clothoids.create_clothoids import clothoids_from_control_points

import numpy as np

class GetHistory(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.data["best"] = []
        self.data["avg"] = []

    def notify(self, algorithm):
        F = algorithm.pop.get("F")
        CV = algorithm.pop.get("CV")

        self.data["best"].append(algorithm.pop.get("F").min())
        self.data["avg"].append(np.mean(F[CV <= 0]))


class PathProblem(Problem):
    def __init__(self, arr_len, arr_xl, arr_xu, path, heightmap, tau):
        super().__init__(n_var=arr_len, xl=arr_xl, xu=arr_xu, n_obj=1, n_constr=1)
        self.tunnel_factor = 5
        self.gradient_factor = 2
        self.curvature_radius = 100
        self.gradient_change_limit = 0.08
        self.height_limit = 800
        self.path = path
        self.heightmap = heightmap
        self.width = heightmap.shape[1]
        self.height = heightmap.shape[0]
        self.tau = tau

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
        return np.round(new_point)

    def get_height(self, point):
        """
        Retrieves the height value at a given point on the heightmap.

        Args:
            point: The coordinates of the point.

        Returns:
            The height value at the specified point.
        """
        return self.heightmap[int(point[0]), int(point[1])]

    def compute_gradient_diff(self, A, B):
        """
        Computes the gradient (slope) between two points on a heightmap.

        Args:
            A: The coordinates of the first point.
            B: The coordinates of the second point.

        Returns:
            The gradient between the two points.
        """
        horizontal_distance = np.linalg.norm(B - A)
        h_A = self.get_height(A)
        h_B = self.get_height(B)
        height_difference = h_B - h_A
        return height_difference / horizontal_distance

    def differential_curvature(self, points, min_radius):
        """
        Calculates the differential curvature of a curve defined by a set of points.

        Args:
            points: A NumPy array of points representing the curve.
            min_radius: The minimum radius of curvature allowed.

        Returns:
            A tuple containing:
                - The valid points that satisfy the minimum radius constraint.
                - The corresponding valid radii.
                - The radii for all points.
        """
        # derivation computation first and second
        dx = np.gradient(points[:, 0])
        dy = np.gradient(points[:, 1])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)

        # curvature computation
        curvature = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2) ** 1.5

        # curves radius
        radii = np.where(curvature != 0, 1 / curvature, float("inf"))

        # validation of radius
        valid_mask = radii > min_radius
        valid_points = points[valid_mask]
        valid_radii = radii[valid_mask]

        return valid_points, valid_radii, radii

    def clip_point(self, new_point):
        """
        Clips a point to ensure it stays within the bounds of a given heightmap.

        Args:
            new_point: The coordinates of the point to clip.

        Returns:
            The clipped coordinates.
        """
        return [
            int(np.clip(new_point[0], 0, self.height - 1)),
            int(np.clip(new_point[1], 0, self.width - 1)),
        ]

    def check_clothoid_radius_constraint(self, clothoid_points, min_radius):
        valid_points, valid_radii, all_radii = self.differential_curvature(
            clothoid_points, min_radius
        )
        return len(clothoid_points) - len(valid_points)

    def _compute_path_length(self, path_points):
        """
        Computes the path length based on various constraints in the PathProblem instance.

        Args:
            path_points (list or np.array): List or array of points along the path.

        Returns:
            path_length (float): The computed length of the path.
        """
        path_length = 0
        A = path_points[0]

        for i in range(1, len(path_points)):
            B = path_points[i]
            h_cP = self.get_height(B)
            gradient = self.compute_gradient_diff(A, B)

            if h_cP > self.height_limit:
                if (
                    gradient < -self.gradient_change_limit
                    or gradient > self.gradient_change_limit
                ):
                    path_length += (
                        np.linalg.norm(B - A)
                        * self.tunnel_factor
                        * self.gradient_factor
                    )
                else:
                    path_length += np.linalg.norm(B - A) * self.tunnel_factor
            elif (
                gradient < -self.gradient_change_limit
                or gradient > self.gradient_change_limit
            ):
                path_length += np.linalg.norm(B - A) * self.gradient_factor
            else:
                path_length += np.linalg.norm(B - A)
            A = B

        return path_length

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluates the path length and penalty based on various constraints in the PathProblem instance.

        Args:
            x (list or np.array): List or array of data points to be evaluated.
            out (dict): Dictionary to store the results of the evaluation.

        Returns:
            None: The results are stored in the 'out' dictionary with keys 'F' for path lengths and 'G' for penalties.
        """
            
        res = []
        pen = []
        
        curvature_radius = self.curvature_radius
        
        # unpack kwargs into the variable path
        if "path" in kwargs:
            path = kwargs["path"]
        else:
            path = self.path

        for data in x:
            # add start point
            new_path = [path[0]]
            # compute the rest of the points
            for i in range(len(path) - 2):
                A = np.array(path[i])
                B = np.array(path[i + 1])
                t_point = data[i]
                new_point = self.find_perpendicular_point(A, B, t_point)
                new_point = self.clip_point(new_point)
                new_path.append(new_point)
            # add end point
            new_path.append(path[-1])

            new_path = np.array(new_path)
            clothoids = clothoids_from_control_points(new_path, tau=self.tau)
            diff = self.check_clothoid_radius_constraint(
                clothoids, 1 / curvature_radius
            )
            path_length = self._compute_path_length(clothoids)

            res.append(path_length + diff * 1000)
            pen.append(diff)

        out["F"] = np.array(res)
        out["G"] = np.array(pen)