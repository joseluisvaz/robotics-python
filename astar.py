import numpy as np
import matplotlib.pyplot as plt
import heapq


class AStar(object):
    def __init__(self, xlim, ylim, walls):
        self.xlim = xlim
        self.ylim = ylim
        self.walls = walls

    def heuristic(self, a, b):
        return 1.0 * np.hypot(a[0] - b[0], a[1] - b[1])

    def backtrack_path(self, parents, start, goal):
        path = [
            goal,
        ]
        current = goal
        while current != start:
            path.append(parents[current])
            current = parents[current]
        path.reverse()
        return path

    def verify_node(self, next):
        return (
            next not in self.walls
            and (self.xlim > next[0] >= 0)
            and (self.ylim > next[1] >= 0)
        )

    def a_star_search(self, start, goal, callback_fn):
        frontier = []
        heapq.heappush(frontier, (0.0, start))

        parents = {start: None}
        cost_so_far = {start: 0}

        motions = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]

        counter = 0
        while len(frontier) != 0:
            counter += 1
            _, current = heapq.heappop(frontier)

            N_STEPS = 200
            if len(frontier) != 0 and counter % N_STEPS == 0:
                callback_fn(frontier, cost_so_far)

            if current == goal:
                print("Goal found!")
                break

            for i, _ in enumerate(motions):
                next = (current[0] + motions[i][0], current[1] + motions[i][1])
                new_cost = cost_so_far[current] + np.hypot(
                    current[0] - next[0], current[1] - next[1]
                )

                if not self.verify_node(next):
                    continue

                if (next not in cost_so_far) or (new_cost < cost_so_far[next]):
                    priority = new_cost + self.heuristic(goal, next)
                    heapq.heappush(frontier, (priority, next))
                    cost_so_far[next] = new_cost
                    parents[next] = current

        return parents, frontier

    def search(self, start, goal, callback_fn):
        parents, _ = self.a_star_search(start, goal, callback_fn)
        return self.backtrack_path(parents, start, goal)


if __name__ == "__main__":

    start = (20, 20)
    goal = (70, 70)
    xlim = 80
    ylim = 80

    walls = list(
        map(
            tuple,
            np.stack(
                (np.arange(5, 31), 30 * np.ones((31 - 5,), dtype=np.int64))
            ).T.tolist(),
        )
    )
    walls += list(
        map(
            tuple,
            np.stack((30 * np.ones((30 - 5,), dtype=np.int64), np.arange(5, 30))).T,
        )
    )

    walls_array = np.array(walls)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.scatter(*walls_array.T, c="k", marker="s")
    ax.axis("equal")

    def callback_fn(frontier, cost_so_far):
        ax.clear()
        ax.scatter(start[0], start[1], c="g", s=100)
        ax.scatter(goal[0], goal[1], c="r", s=100)
        ax.scatter(*walls_array.T, c="k", marker="s")
        points = np.stack([elem[1] for elem in frontier])
        ax.scatter(points[:, 0], points[:, 1], c="gray", s=50, alpha=0.2)
        # plot cost
        points = np.stack([elem for elem in cost_so_far.keys()])
        cost = np.array([elem for elem in cost_so_far.values()])
        ax.scatter(points[:, 0], points[:, 1], c=cost, cmap="coolwarm", s=50, alpha=0.2)
        ax.axis("equal")
        plt.pause(0.1)

    astar = AStar(xlim, ylim, walls)
    path = astar.search(start, goal, callback_fn)
    ax.scatter(*np.array(path).T, c="b", s=8)
    plt.show()
