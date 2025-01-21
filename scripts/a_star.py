import heapq
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Callable, Optional


class A_star(object):
    def get_neighbors(
        self, current: Tuple[int, int], shape: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        x, y = current
        neighbors = []
        for dx, dy in (
            (0, 1),
            (1, 0),
            (0, -1),
            (-1, 0),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ):
            nx, ny = x + dx, y + dy
            if 0 <= nx < shape[0] and 0 <= ny < shape[1]:
                neighbors.append((nx, ny))
        return neighbors

    def astar_heightmap(
        self,
        heightmap: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        heuristic: Callable[[Tuple[int, int], Tuple[int, int]], float],
        cost_function: Callable[[np.ndarray, Tuple[int, int], Tuple[int, int]], float],
    ) -> List[Tuple[int, int]]:
        heap = [(0, start)]
        came_from = {}
        cost_so_far = {start: 0}
        shape = heightmap.shape

        while heap:
            _, current = heapq.heappop(heap)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for next in self.get_neighbors(current, shape):
                new_cost = cost_so_far[current] + cost_function(
                    heightmap, current, next
                )
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + heuristic(next, goal)
                    heapq.heappush(heap, (priority, next))
                    came_from[next] = current

        return []  # No path found

    def downsample_heightmap(self, heightmap, target_height, target_width):
        current_height, current_width = heightmap.shape

        # Calculate the downsampling factors
        height_factor = current_height // target_height
        width_factor = current_width // target_width

        # Ensure the factors are at least 1
        height_factor = max(1, height_factor)
        width_factor = max(1, width_factor)

        # Calculate the new dimensions
        new_height = current_height // height_factor
        new_width = current_width // width_factor

        # Reshape and mean
        reshaped = heightmap[: new_height * height_factor, : new_width * width_factor]
        reshaped = reshaped.reshape(new_height, height_factor, new_width, width_factor)
        downsampled = reshaped.mean(axis=(1, 3))

        return downsampled

    # Heuristic
    def euclidean_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    # Cost
    def height_difference_cost(
        self, heightmap: np.ndarray, current: Tuple[int, int], next: Tuple[int, int]
    ) -> float:
        height_diff = abs(heightmap[next] - heightmap[current])
        return height_diff + 1  # Add base cost to prefer shorter paths

    def complex_cost1(
        self, heightmap: np.ndarray, current: Tuple[int, int], next: Tuple[int, int]
    ) -> float:
        # Calculate base distance (assuming grid-based movement)
        base_distance = (
            1 if (abs(current[0] - next[0]) + abs(current[1] - next[1])) == 1 else 1.414
        )  # Diagonal movement

        # Calculate height difference
        current_height = heightmap[current]
        next_height = heightmap[next]
        height_diff = next_height - current_height

        # Calculate gradient (as a percentage)
        gradient = (height_diff / base_distance) * 100

        # Factor 1: Change of gradients
        if abs(gradient) > 8:
            base_distance *= 2

        # Factor 2: Tunnels (assuming height values are in meters)
        tunnel_threshold = 800  # meters
        if current_height > tunnel_threshold and next_height > tunnel_threshold:
            base_distance *= 5

        return base_distance

    def complex_cost2(
        self, heightmap: np.ndarray, current: Tuple[int, int], next: Tuple[int, int]
    ) -> float:
        # Calculate base distance (assuming grid-based movement)
        base_distance = (
            1 if (abs(current[0] - next[0]) + abs(current[1] - next[1])) == 1 else 1.414
        )  # Diagonal movement

        # Calculate height difference
        current_height = heightmap[current]
        next_height = heightmap[next]
        height_diff = next_height - current_height

        # Calculate gradient (as a percentage)
        gradient = (height_diff / base_distance) * 100

        cost = abs(gradient)

        # Factor 1: Change of gradients
        if abs(gradient) > 8:
            cost *= 2

        # Factor 2: Tunnels (assuming height values are in meters)
        tunnel_threshold = 800  # meters
        if current_height > tunnel_threshold and next_height > tunnel_threshold:
            cost *= 5

        return cost

    def plot_heightmap_and_path(
        self, heightmap_data, path=None, start=None, finish=None
    ):
        plt.figure(figsize=(12, 8))

        # Plot the heightmap
        im = plt.imshow(self, heightmap_data, cmap="gist_earth")
        plt.colorbar(label="Elevation", orientation="horizontal")

        # Plot start point
        if start:
            plt.plot(start[1], start[0], "go", markersize=10, label="Start")

        # Plot finish point
        if finish:
            plt.plot(finish[1], finish[0], "ro", markersize=10, label="Finish")

        # Plot path
        if path is not None:
            try:
                if isinstance(path, (list, tuple)) and len(path) > 0:
                    if isinstance(path[0], (list, tuple)) and len(path[0]) == 2:
                        # Path is a list of coordinate tuples
                        path_y, path_x = zip(*path)
                        plt.plot(
                            path_x,
                            path_y,
                            linestyle="-",
                            color="white",
                            linewidth=2,
                            label="Path",
                        )
                    elif isinstance(path[0], int):
                        # Path is a flat list of alternating y and x coordinates
                        path_y = path[::2]
                        path_x = path[1::2]
                        plt.plot(
                            path_x,
                            path_y,
                            linestyle="-",
                            color="white",
                            linewidth=2,
                            label="Path",
                        )
                    else:
                        print(
                            "Warning: Path format not recognized. Unable to plot path."
                        )
                else:
                    print("Warning: Path is empty. Unable to plot path.")
            except Exception as e:
                print(f"Error plotting path: {e}")

        plt.title("Heightmap with Path")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()

    def plot_heightmap_and_paths(
        self,
        heightmap_data: np.ndarray,
        paths: List[List[Tuple[int, int]]],
        start: Optional[Tuple[int, int]] = None,
        finish: Optional[Tuple[int, int]] = None,
        path_labels: Optional[List[str]] = None,
        path_colors: Optional[List[str]] = None,
    ):
        plt.figure(figsize=(12, 8))

        # Plot the heightmap
        im = plt.imshow(heightmap_data, cmap="gist_earth")
        plt.colorbar(label="Elevation", orientation="horizontal")

        # Plot start point
        if start:
            plt.plot(start[1], start[0], "go", markersize=10, label="Start")

        # Plot finish point
        if finish:
            plt.plot(finish[1], finish[0], "ro", markersize=10, label="Finish")

        # Default colors if not provided
        if path_colors is None:
            path_colors = plt.cm.rainbow(np.linspace(0, 1, len(paths)))

        # Default labels if not provided
        if path_labels is None:
            path_labels = [f"Path {i+1}" for i in range(len(paths))]

        # Plot paths
        for i, path in enumerate(paths):
            if path:
                path_y, path_x = zip(*path)
                plt.plot(
                    path_x,
                    path_y,
                    linestyle="-",
                    color=path_colors[i],
                    linewidth=2,
                    label=path_labels[i],
                )

        plt.title("Heightmap with Multiple Paths")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()
