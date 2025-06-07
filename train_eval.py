import time
import numpy as np
import torch
import torch.nn.functional as F
from utsp_gnn import SimpleGNN, t_to_h, surrogate_loss
from utsp_gnn_mod import GATGNN
from utsp_gnn_gin import GINGNN
from utsp_gnn_sage import SAGEGNN
from utsp_gnn_sag import SAGGNN  # Import the SAG GNN model
from surrogate_losses import surrogate_loss_entropy, surrogate_loss_laplacian
from utils import generate_tsp_instance, compute_distance_matrix, optimal_tour_length, gap_percent
from local_search import two_opt, greedy_tour_from_heatmap, tour_length


def run_experiment(model_class, loss_fn, n_cities, n_epochs=50, hidden_dim=64, lr=1e-3, tau=0.1, seed=42):
    coords = generate_tsp_instance(n_cities, seed)
    D = compute_distance_matrix(coords)
    n = n_cities
    edge_index = torch.tensor(np.array(np.meshgrid(np.arange(n), np.arange(n))).reshape(2, -1), dtype=torch.long)
    x = torch.from_numpy(coords).float()
    model = model_class(in_dim=2, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start_time = time.time()
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        edge_logits = model(x, edge_index)
        T = edge_logits.view(n, n)
        T = F.softmax(T, dim=0)
        H = t_to_h(T)
        loss = loss_fn(T, H, D)
        loss.backward()
        optimizer.step()
    # Evaluation
    model.eval()
    with torch.no_grad():
        edge_logits = model(x, edge_index)
        T = edge_logits.view(n, n)
        T = F.softmax(T, dim=0)
        H = t_to_h(T)
        H_np = H.cpu().numpy()
    # Greedy tour from heatmap
    pred_tour = greedy_tour_from_heatmap(H_np)
    pred_len = tour_length(pred_tour, D)
    # 2-opt improvement
    improved_tour = two_opt(pred_tour, D)
    improved_len = tour_length(improved_tour, D)
    # Optimal (approx for n>20)
    opt_len, _ = optimal_tour_length(D)
    gap = gap_percent(improved_len, opt_len)
    elapsed = time.time() - start_time
    return {
        'pred_len': pred_len,
        'improved_len': improved_len,
        'opt_len': opt_len,
        'gap_percent': gap,
        'coords': coords,
        'H': H_np,
        'tour': improved_tour,
        'elapsed': elapsed
    }

if __name__ == "__main__":
    sizes = [20, 50, 100, 200, 500]
    models = [
        ("SAG (Baseline)", SAGGNN),
        ("GCN", SimpleGNN),
        ("GAT", GATGNN),
        ("GIN", GINGNN),
        ("GraphSAGE", SAGEGNN)
    ]
    losses = [
        ("Original", surrogate_loss),
        ("Entropy", surrogate_loss_entropy),
        ("Laplacian", surrogate_loss_laplacian)
    ]
    for n in sizes:
        print(f"\n===== TSP size n={n} =====")
        for model_name, model_class in models:
            for loss_name, loss_fn in losses:
                print(f"--- {model_name} | Loss: {loss_name} ---")
                res = run_experiment(model_class, loss_fn, n_cities=n)
                print(f"Tour length: {res['improved_len']:.3f} | Opt: {res['opt_len']:.3f} | Gap: {res['gap_percent']:.2f}% | Time: {res['elapsed']:.2f}s")
    print("\nNote: The baseline model is now SAG as in the original UTSP paper.")
