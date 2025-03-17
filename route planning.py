import networkx as nx


G = nx.Graph()


G.add_node("A", pos=(0, 0))
G.add_node("B", pos=(1, 2))
G.add_node("C", pos=(2, 2))
G.add_node("D", pos=(3, 1))


G.add_edge("A", "B", weight=2.2)
G.add_edge("A", "C", weight=3.1)
G.add_edge("B", "C", weight=1.5)
G.add_edge("B", "D", weight=2.7)
G.add_edge("C", "D", weight=1.8)


pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue")
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

import osmnx as ox


place_name = "Berkeley, California, USA"
G = ox.graph_from_place(place_name, network_type='drive')


ox.plot_graph(G)


def dijkstra(graph, start, end):
    shortest_paths = {start: (None, 0)}
    current_node = start
    visited = set()

    while current_node != end:
        visited.add(current_node)
        destinations = graph[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node, weight in destinations.items():
            weight = weight['weight'] if 'weight' in weight else 1.0
            total_weight = weight_to_current_node + weight
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, total_weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > total_weight:
                    shortest_paths[next_node] = (current_node, total_weight)

        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Route Not Possible"
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node
    path = path[::-1]
    return path



start_node = "A"
end_node = "D"
path = dijkstra(G, start_node, end_node)
print("Shortest path:", path)

import heapq


def a_star(graph, start, end, heuristic):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph}
    f_score[start] = heuristic(start, end)

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for neighbor, weight in graph[current].items():
            tentative_g_score = g_score[current] + weight['weight']
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return "Route Not Possible"



def heuristic(node1, node2):
    pos1 = G.nodes[node1]['pos']
    pos2 = G.nodes[node2]['pos']
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5



start_node = "A"
end_node = "D"
path = a_star(G, start_node, end_node, heuristic)
print("A* path:", path)

from scipy.optimize import linprog


c = [2.2, 3.1, 1.5, 2.7, 1.8]


A_eq = [
    [1, 1, 0, 0, 0],
    [-1, 0, 1, 1, 0],
    [0, -1, -1, 0, 1],
    [0, 0, 0, -1, -1]
]
b_eq = [1, 0, 0, -1]


result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1))

print("Optimal path:", result.x)

from scipy.optimize import milp
import numpy as np


c = np.array([2.2, 3.1, 1.5, 2.7, 1.8])


A_eq = np.array([
    [1, 1, 0, 0, 0],
    [-1, 0, 1, 1, 0],
    [0, -1, -1, 0, 1],
    [0, 0, 0, -1, -1]
])
b_eq = np.array([1, 0, 0, -1])


bounds = [(0, 1)] * 5
integrality = [1] * 5


result = milp(c=c, constraints=(A_eq, b_eq), bounds=bounds, integrality=integrality)


print("Optimal path:", result.x)

import matplotlib.pyplot as plt


pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue")
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)


path_edges = list(zip(path[:-1], path[1:]))
nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="r", width=2)

plt.show()

import folium


path_coords = [G.nodes[node]['pos'] for node in path]


m = folium.Map(location=path_coords[0], zoom_start=15)


folium.PolyLine(path_coords, color="blue", weight=2.5, opacity=1).add_to(m)


m.save("path.html")

