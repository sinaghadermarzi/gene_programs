# gene_programs

Sanity check of a "gene program" projection method, using MNIST as a stand-in for single-cell data.

The idea: given a data matrix `X` (cells × genes) and a fixed set of program definitions `P`
(programs × genes), train an encoder `f` that maps each cell to a vector of program scores
such that `f(X) @ P` reconstructs `X` (MSE loss). On MNIST, pixels play the role of genes,
images play the role of cells, and the 10 programs are the mean image of each digit, so a
correct encoder should give each image its highest score on the program of its own digit.

## What is in the repo

| File | Purpose |
| --- | --- |
| `models.py` | The two encoders: `geneprog_encoder_linear` (one linear layer) and `geneprog_encoder_MLP` (three fully connected layers with ReLU). |
| `gene_programs.py` | `train_and_encode_for_mnist(...)`: trains an encoder with Adam and cosine-annealing warm restarts, logs per-epoch reconstruction metrics (MSE, R², Pearson) and label metrics on the softmaxed program scores (cross-entropy, one-vs-rest AUC, top-program accuracy) to Weights & Biases, saves `model.pt`, and returns the program scores and reconstruction for all of `X`. Run as a script it trains a linear encoder for 4 epochs and an MLP encoder for 10. |
| `mnist.ipynb` | Explores MNIST (PCA / t-SNE / UMAP), builds the digit programs, trains the MLP encoder for 250 epochs via `train_and_encode_for_mnist`, then visualizes reconstructions, the program-score embedding, a per-label heatmap of program scores, and top program vs. label. |
| `mnist_gnn.ipynb` | Variant with a graph encoder: builds a 5-nearest-neighbour graph of the images with `scanpy`, trains a `torch_geometric` `GCNConv` encoder with `GraphSAINTRandomWalkSampler` and the same reconstruction loss and metrics. |
| `conda_env.yml` | Exported conda environment (`hugface`: Python 3.9, PyTorch 2.1.2 + CUDA 12.1, `datasets`, `wandb`, `scikit-learn`, `umap-learn`, Jupyter, ...). |

## Setup

```bash
conda env create -f conda_env.yml
conda activate hugface
```

`mnist_gnn.ipynb` additionally needs `torch_geometric`, `scanpy` and `anndata`, which are not in
`conda_env.yml`. That notebook also hard-codes `torch.device('mps:0')` (Apple Silicon); change it
to `cuda:0` or `cpu` elsewhere.

## Running

```bash
# script: linear encoder (4 epochs) then MLP encoder (10 epochs)
python gene_programs.py

# notebooks (run from the repo root so `from models import ...` resolves)
jupyter lab
```

MNIST is downloaded on first use through `datasets.load_dataset("mnist")`.

## Outputs and caveats

- Each run writes `./gene_program_runs/<RUN_NAME>/model.pt`; `RUN_NAME` is the encoder name plus a timestamp.
- `mnist.ipynb` saves its figures (`pca.png`, `pca_ratio.png`, `tsne.png`, `umap.png`) into the working directory.
- W&B logging is on by default and `wandb.init` uses `project="gene_programs", entity="sinag"`.
  Pass `WANDB_LOGGING=False` (script / `mnist.ipynb`) or edit the entity before running under another account.
