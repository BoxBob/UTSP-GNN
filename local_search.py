import numpy as np

def two_opt(route, dist_matrix):
    n = len(route)
    best = route.copy()
    improved = True
    while improved:
        improved = False
        for i in range(1, n-2):
            for j in range(i+1, n):
                if j-i == 1: continue
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                if tour_length(new_route, dist_matrix) < tour_length(best, dist_matrix):
                    best = new_route
                    improved = True
        route = best
    return best

def tour_length(route, dist_matrix):
    return sum(dist_matrix[route[i], route[(i+1)%len(route)]] for i in range(len(route)))

# Heatmap-guided greedy tour

def greedy_tour_from_heatmap(H):
    n = H.shape[0]
    visited = [False]*n
    tour = [0]
    visited[0] = True
    for _ in range(n-1):
        last = tour[-1]
        probs = H[last].copy()
        for i in range(n):
            if visited[i]:
                probs[i] = -np.inf
        next_city = np.argmax(probs)
        tour.append(next_city)
        visited[next_city] = True
    return tour
