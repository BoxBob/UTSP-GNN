import matplotlib.pyplot as plt
import numpy as np
import torch
from utsp_gnn import train_utsp, t_to_h

# Run the UTSP training and get model, coordinates, and distance matrix
model, coords, D = train_utsp(n_cities=20, n_epochs=50)

# Get the edge probabilities (heatmap) from the trained model
model.eval()
with torch.no_grad():
    n = coords.shape[0]
    edge_index = torch.tensor(np.array(np.meshgrid(np.arange(n), np.arange(n))).reshape(2, -1), dtype=torch.long)
    x = torch.from_numpy(coords).float()
    edge_logits = model(x, edge_index)
    T = edge_logits.view(n, n)
    T = torch.softmax(T, dim=0)
    H = t_to_h(T)
    H_np = H.cpu().numpy()

# Plot the TSP instance
plt.figure(figsize=(6, 6))
plt.scatter(coords[:, 0], coords[:, 1], c='blue', label='Cities')
for i, (x, y) in enumerate(coords):
    plt.text(x, y, str(i), fontsize=8, ha='right')
plt.title('TSP Instance (Cities)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Plot the heatmap H
plt.figure(figsize=(6, 5))
plt.imshow(H_np, cmap='hot', interpolation='nearest')
plt.colorbar(label='Edge Probability')
plt.title('Predicted Heatmap H (Edge Probabilities)')
plt.xlabel('City i')
plt.ylabel('City j')
plt.show()
