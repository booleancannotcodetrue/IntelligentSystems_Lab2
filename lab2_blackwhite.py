import matplotlib.pyplot as plt
import heapq
import numpy as np
import time

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(array, start, goal, animate=False, ax=None):
    neighbors = [(0,1),(0,-1),(1,0),(-1,0)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
    steps = []
    step_count = 0
    while oheap:
        current = heapq.heappop(oheap)[1]
        print(f"[A*] Step {step_count}: Exploring node {current}, Open set size: {len(oheap)}, Closed set size: {len(close_set)}")
        step_count += 1
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            if animate and ax:
                animate_search(array, steps, data[::-1], start, goal, ax, "A* Search")
            return data[::-1], close_set
        close_set.add(current)
        steps.append((set(close_set), list(oheap)))
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + 1
            if 0 <= neighbor[0] < array.shape[0] and 0 <= neighbor[1] < array.shape[1]:
                if array[neighbor[0]][neighbor[1]] == 1:
                    continue
            else:
                continue
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                continue
            if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    if animate and ax:
        animate_search(array, steps, [], start, goal, ax, "A* Search")
    return [], close_set

def dijkstra(array, start, goal, animate=False, ax=None):
    neighbors = [(0,1),(0,-1),(1,0),(-1,0)]
    visited = set()
    came_from = {}
    gscore = {start:0}
    oheap = [(0, start)]
    steps = []
    step_count = 0
    while oheap:
        current_cost, current = heapq.heappop(oheap)
        print(f"[Dijkstra] Step {step_count}: Exploring node {current}, Open set size: {len(oheap)}, Visited set size: {len(visited)}")
        step_count += 1
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            if animate and ax:
                animate_search(array, steps, data[::-1], start, goal, ax, "Dijkstra Search")
            return data[::-1], visited
        visited.add(current)
        steps.append((set(visited), list(oheap)))
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + 1
            if 0 <= neighbor[0] < array.shape[0] and 0 <= neighbor[1] < array.shape[1]:
                if array[neighbor[0]][neighbor[1]] == 1:
                    continue
            else:
                continue
            if neighbor in visited and tentative_g_score >= gscore.get(neighbor, float('inf')):
                continue
            if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                heapq.heappush(oheap, (gscore[neighbor], neighbor))
    if animate and ax:
        animate_search(array, steps, [], start, goal, ax, "Dijkstra Search")
    return [], visited

def greedy_best_first(array, start, goal, animate=False, ax=None):
    neighbors = [(0,1),(0,-1),(1,0),(-1,0)]
    visited = set()
    came_from = {}
    oheap = [(heuristic(start, goal), start)]
    steps = []
    step_count = 0
    while oheap:
        _, current = heapq.heappop(oheap)
        print(f"[GBFS] Step {step_count}: Exploring node {current}, Open set size: {len(oheap)}, Visited set size: {len(visited)}")
        step_count += 1
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            if animate and ax:
                animate_search(array, steps, data[::-1], start, goal, ax, "GBFS Search")
            return data[::-1], visited
        visited.add(current)
        steps.append((set(visited), list(oheap)))
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            if 0 <= neighbor[0] < array.shape[0] and 0 <= neighbor[1] < array.shape[1]:
                if array[neighbor[0]][neighbor[1]] == 1:
                    continue
            else:
                continue
            if neighbor not in visited:
                came_from[neighbor] = current
                heapq.heappush(oheap, (heuristic(neighbor, goal), neighbor))
    if animate and ax:
        animate_search(array, steps, [], start, goal, ax, "GBFS Search")
    return [], visited

def animate_search(maze, steps, path, start, goal, ax, title):
    for visited, open_set in steps:
        ax.clear()
        ax.imshow(maze, cmap=plt.cm.binary)
        ax.set_title(title)
        for (x, y) in visited:
            ax.scatter(y, x, marker=".", color="lightblue", s=20)
        for _, (x, y) in open_set:
            ax.scatter(y, x, marker=".", color="orange", s=20)
        if path:
            x_coords = [p[1] for p in path]
            y_coords = [p[0] for p in path]
            ax.plot(x_coords, y_coords, color="yellow", linewidth=2)
        ax.scatter(start[1], start[0], marker="o", color="green", s=100)
        ax.scatter(goal[1], goal[0], marker="x", color="red", s=100)
        plt.pause(0.05)
    if path:
        x_coords = [p[1] for p in path]
        y_coords = [p[0] for p in path]
        ax.plot(x_coords, y_coords, color="yellow", linewidth=2)
    ax.scatter(start[1], start[0], marker="o", color="green", s=100)
    ax.scatter(goal[1], goal[0], marker="x", color="red", s=100)
    plt.pause(1)

maze = np.array([
        [0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
        [1,1,0,1,1,0,1,1,1,0,1,1,0,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
    [0,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,0],
    [0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0],
    [0,1,0,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0],
    [0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0],
    [0,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,0,1,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
    [1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,1,1,1,0],
    [0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    [0,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
    [0,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,0],
    [0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0],
    [0,1,0,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0],
    [0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0],
    [0,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,0,1,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
    [1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,1,1,1,0]
])

start = (0,0)
goal = (19,19)

plt.close('all')

fig, axs = plt.subplots(1, 3, figsize=(18,6))
plt.ion()

path_astar, visited_astar = astar(maze, start, goal, animate=True, ax=axs[0])
path_dijkstra, visited_dijkstra = dijkstra(maze, start, goal, animate=True, ax=axs[1])
path_gbfs, visited_gbfs = greedy_best_first(maze, start, goal, animate=True, ax=axs[2])

plt.ioff()
plt.show()