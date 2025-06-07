import torch
import torch.nn.functional as F

def surrogate_loss_entropy(T, H, D, lambda1=1.0, lambda2=1.0, lambda3=0.1):
    n = T.size(0)
    # Row sum constraint (each row sums to 1)
    row_sum = torch.sum(T, dim=1)
    loss_row = ((row_sum - 1) ** 2).sum()
    # No self-loops (main diagonal of H)
    loss_diag = torch.diagonal(H).sum()
    # Minimize expected TSP length
    loss_dist = (torch.from_numpy(D).to(T.device) * H).sum()
    # Encourage sharpness (low entropy in T)
    entropy = - (T * (T+1e-8).log()).sum()
    return lambda1 * loss_row + lambda2 * loss_diag + loss_dist + lambda3 * entropy


def surrogate_loss_laplacian(T, H, D, lambda1=1.0, lambda2=1.0, lambda3=0.1):
    n = T.size(0)
    row_sum = torch.sum(T, dim=1)
    loss_row = ((row_sum - 1) ** 2).sum()
    loss_diag = torch.diagonal(H).sum()
    loss_dist = (torch.from_numpy(D).to(T.device) * H).sum()
    # Laplacian smoothness penalty (encourage similar values for adjacent edges)
    lap = ((T - T.mean(dim=1, keepdim=True))**2).sum()
    return lambda1 * loss_row + lambda2 * loss_diag + loss_dist + lambda3 * lap
