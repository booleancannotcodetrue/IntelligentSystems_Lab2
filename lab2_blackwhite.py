import matplotlib.pyplot as plt
import heapq
import numpy as np

def heuristic(a, b):
    # Manhattan distance heuristic
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(array, start, goal):
    # A* Search: uses both g-score (cost) and heuristic 
    neighbors = [(0,1),(0,-1),(1,0),(-1,0)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
    step_count = 0

    while oheap:
        current = heapq.heappop(oheap)[1]   
        print(f"[A*] Step {step_count}: Exploring node {current}, Open set size: {len(oheap)}, Closed set size: {len(close_set)}")
        step_count += 1
        if current == goal:
            # Reconstruct path
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data[::-1], close_set
        
        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + 1
            # Check bounds and obstacles
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
    return [], close_set


def dijkstra(array, start, goal):
    # Dijkstra's Algorithm: uses only g-score (cost)
    neighbors = [(0,1),(0,-1),(1,0),(-1,0)]
    visited = set()
    came_from = {}
    gscore = {start:0}
    oheap = [(0, start)]
    step_count = 0
    while oheap:
        current_cost, current = heapq.heappop(oheap)
        print(f"[Dijkstra] Step {step_count}: Exploring node {current}, Open set size: {len(oheap)}, Visited set size: {len(visited)}")
        step_count += 1

        if current == goal:
            # Reconstruct path
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data[::-1], visited
        
        visited.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + 1
            # Check bounds and obstacles
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
    return [], visited

def greedy_best_first(array, start, goal):
    # Greedy Best-First Search: uses only heuristic
    neighbors = [(0,1),(0,-1),(1,0),(-1,0)]
    visited = set()
    came_from = {}
    oheap = [(heuristic(start, goal), start)]
    step_count = 0
    while oheap:
        _, current = heapq.heappop(oheap)
        print(f"[GBFS] Step {step_count}: Exploring node {current}, Open set size: {len(oheap)}, Visited set size: {len(visited)}")
        step_count += 1

        if current == goal:
            # Reconstruct path
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data[::-1], visited
        
        visited.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            # Check bounds and obstacles
            if 0 <= neighbor[0] < array.shape[0] and 0 <= neighbor[1] < array.shape[1]:
                if array[neighbor[0]][neighbor[1]] == 1:
                    continue
            else:
                continue

            if neighbor not in visited:
                came_from[neighbor] = current
                heapq.heappush(oheap, (heuristic(neighbor, goal), neighbor))
    return [], visited

def draw_result(maze, visited, path, start, goal, ax, title):
    # Visualize maze, visited nodes, and final path
    ax.imshow(maze, cmap=plt.cm.binary)
    ax.set_title(title)
    # Plot visited nodes
    for (x, y) in visited:
        ax.scatter(y, x, marker='.', color='lightblue', s=20)
    # Plot final path
    if path:
        x_coords = [p[1] for p in path]
        y_coords = [p[0] for p in path]
        ax.plot(x_coords, y_coords, color="yellow", linewidth=2)
    # Plot start and goal
    ax.scatter(start[1], start[0], marker="o", color="green", s=100)
    ax.scatter(goal[1], goal[0], marker="x", color="red", s=100)

maze = np.array([
    [0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    [1,1,0,1,1,0,1,1,1,0,1,1,0,1,0,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
    [0,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,0],
    [0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0],
    [0,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0],
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
goal = (17,0)

plt.close('all')

fig, axs = plt.subplots(1, 3, figsize=(17,6))

# Run all three search algorithms
path_astar, visited_astar = astar(maze, start, goal)
path_dijkstra, visited_dijkstra = dijkstra(maze, start, goal)
path_gbfs, visited_gbfs = greedy_best_first(maze, start, goal)

# Draw results
draw_result(maze, visited_astar, path_astar, start, goal, axs[0], "A* Search")
draw_result(maze, visited_dijkstra, path_dijkstra, start, goal, axs[1], "Dijkstra Search")
draw_result(maze, visited_gbfs, path_gbfs, start, goal, axs[2], "Greedy Best-First Search")

plt.show()
