# UTSP: Unsupervised Learning for TSP with GNNs

This repository implements and extends the UTSP framework (arXiv:2303.10538v2) for solving the Travelling Salesman Problem (TSP) using unsupervised learning and Graph Neural Networks (GNNs).

## Features
- Baseline UTSP model (GCN-based)
- Modified UTSP model (GAT-based)
- Surrogate loss for unsupervised training
- Heatmap generation and visualization
- Local search heuristics (greedy, 2-opt)
- Experimental comparison on TSP sizes n=20,50,100,200,500
- Utilities for data, metrics, and visualization

## Requirements
See `req.txt` for dependencies. Install with:

    pip install -r req.txt

## Usage
- **Train and evaluate models:**

      python train_eval.py

- **Visualize heatmaps and tours:**

      python visualize_utsp.py

## File Structure
- `utsp_gnn.py`: Baseline GCN UTSP model
- `utsp_gnn_mod.py`: Modified GAT UTSP model
- `local_search.py`: Local search heuristics
- `utils.py`: Data generation, metrics
- `train_eval.py`: Training and evaluation scripts
- `visualize_utsp.py`: Visualization
- `req.txt`: Requirements
- `report.md`: Experimental report

## Results
See `report.md` for experimental results, implementation details, and performance comparisons.

## References
- [arXiv:2303.10538v2](https://arxiv.org/abs/2303.10538)
