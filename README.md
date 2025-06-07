# UTSP: Unsupervised Learning for TSP with GNNs

This repository implements and extends the UTSP framework (arXiv:2303.10538v2) for solving the Travelling Salesman Problem (TSP) using unsupervised learning and Graph Neural Networks (GNNs).

## Setup Instructions

### 1. Clone the repository

```
git clone <https://github.com/BoxBob/UTSP-GNN.git>
cd UTSP
```

### 2. Create and activate a Python virtual environment (recommended)

On **Windows** (PowerShell):

```
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On **Linux/Mac**:

```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install required modules

```
pip install -r req.txt
```

## Running Experiments

### 1. Train and evaluate all models and losses

This will run all GNN architectures (SAG, GCN, GAT, GIN, GraphSAGE) with all surrogate losses (Original, Entropy, Laplacian) on TSP sizes n=20, 50, 100, 200, 500. Results will be printed to the console.

```
python train_eval.py
```

### 2. Visualize a TSP instance, heatmap, and predicted tour

```
python visualize_utsp.py
```

## File Structure
- `utsp_gnn.py`: Baseline GCN UTSP model
- `utsp_gnn_sag.py`: SAG (Scattering Attention GNN, baseline in paper)
- `utsp_gnn_mod.py`: Modified GAT UTSP model
- `utsp_gnn_gin.py`: GIN-based UTSP model
- `utsp_gnn_sage.py`: GraphSAGE-based UTSP model
- `local_search.py`: Local search heuristics
- `surrogate_losses.py`: Alternative surrogate loss functions
- `utils.py`: Data generation, metrics
- `train_eval.py`: Training and evaluation scripts
- `visualize_utsp.py`: Visualization
- `req.txt`: Requirements
- `report.md`: Experimental report
- `.gitignore`: Git ignore rules

## Results and Report
- See `report.md` for experimental results, implementation details, and performance comparisons.


## Notes
- The baseline model is SAG (Scattering Attention GNN) as in the original UTSP paper.
- All code is compatible with Python 3.8+ and tested on Windows and Linux.
- For best results, use a machine with a GPU and sufficient RAM for large TSP instances.

---

For any issues or questions, please open an issue or contact the repository maintainer.

## References
- [arXiv:2303.10538v2](https://arxiv.org/abs/2303.10538)
