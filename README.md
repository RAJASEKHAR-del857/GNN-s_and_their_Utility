# Movie Recommendation System: GCMC vs. Funk SVD vs. LLORMA

A comparative study of three structurally different collaborative filtering architectures — **Graph Convolutional Matrix Completion (GCMC)**, **Funk SVD (matrix factorization)**, and **LLORMA (Local Low-Rank Approximation)** — trained and evaluated on the **MovieLens 20M** dataset, with all models engineered to run within the memory budget of a single Tesla T4 GPU.

---

## 1. Project Overview

Recommendation systems can be built on very different mathematical foundations: some treat user-item interactions as a **graph** and learn via message passing, others treat them as a **low-rank matrix** to be factorized, and others learn a **mixture of local low-rank models** anchored around similar users/items. This project implements all three approaches on the same dataset and the same train/test split to get an apples-to-apples comparison of:

- Predictive accuracy (RMSE)
- Training/convergence behavior
- GPU vs. CPU execution efficiency
- Practical trade-offs (memory footprint, training time, implementation complexity)

**Dataset:** [MovieLens 20M](https://grouplens.org/datasets/movielens/20m/) — 20,000,263 ratings, ~138K users, ~27K movies, ratings on a 0.5–5.0 scale (10 classes).

**Hardware constraint:** Tesla T4 (15 GB VRAM). This constraint directly shaped several engineering decisions described below.

---

## 2. Models Implemented

### 2.1 Graph Convolutional Matrix Completion (GCMC)
A graph-based encoder-decoder architecture that treats users and movies as nodes in a bipartite graph, with a separate edge type per rating value (0.5, 1.0, ..., 5.0).

- **Encoder:** For each rating class *r*, a dedicated weight matrix `W_r` transforms neighbor embeddings, which are then aggregated (sum + ReLU) into a node's hidden representation:
  
  $$h_u = \text{ReLU}\Big(\sum_{r \in R}\sum_{i \in N_r(u)} W_r \cdot e_i\Big)$$

- **Decoder:** A bilinear decoder scores each rating class using a learned matrix `Q_r` per class, producing a full probability distribution over the 10 rating classes rather than a single scalar:
  
  $$P(r \mid u, i) = \text{softmax}(u^T Q_r\, m)$$

  The final predicted rating is the expectation over this distribution.
- **Loss:** Negative log-likelihood (NLL) over the predicted rating-class distribution.
- **Side-feature extension (Enhanced GCMC):** Extended the base encoder to fuse in:
  - **Movie side features:** one-hot encoded genres (19 genres → `[NUM_MOVIES, 19]` tensor).
  - **User side features:** per-user `mean_rating`, `rating_count`, and `std_rating`, normalized to `[0, 1]` (`[NUM_USERS, 3]` tensor).
  
  These are projected and added to the base node embeddings before message passing, letting the graph convolution condition on genre/behavioral signal in addition to raw interaction structure.

### 2.2 Funk SVD (Matrix Factorization)
The latent-factor approach popularized by the Netflix Prize. Each user and movie is represented as a learned embedding vector, and the predicted rating is a bias-adjusted dot product:

$$\hat{r}_{u,i} = \mu + b_u + b_i + q_i^T p_u$$

Unlike GCMC, this approach ignores graph/relational structure entirely and relies purely on the latent overlap between user and movie vectors, trained with MSE loss via Adam.

### 2.3 LLORMA (Local Low-Rank Approximation)
Implemented from scratch to test whether a **mixture of local low-rank models** — each specialized to a neighborhood of similar users/items and combined via kernel-weighted averaging — could outperform a single global low-rank model like Funk SVD, without the graph-structure assumptions of GCMC. Unlike GCMC's message passing (which parallelizes well on GPU), LLORMA's sequential anchor-based factorization is far less GPU-friendly, which was profiled directly (see Section 6).

---

## 3. Handling the Memory Budget

Running GNN-scale models on 20M ratings inside 15 GB of VRAM required several deliberate trade-offs:

| Constraint | Decision |
|---|---|
| Full graph would need gigabytes for edge indices + embeddings | Subsampled training set to **100,000** interactions, test set to **20,000** interactions |
| Standard latent dims (64/128) were too large for available memory | Reduced hidden dimension to **16** for GCMC, **10** for Funk SVD (>75% reduction in weight matrix size) |
| Repeated OOM risk when switching between models | Implemented a `clear_memory()` routine: `del` references → `gc.collect()` → `torch.cuda.empty_cache()` |
| Non-sequential raw user/movie IDs unsuitable for embedding lookup | Built `user_map` / `movie_map` dictionaries to remap IDs to dense 0-indexed ranges |
| Bipartite graph needs a single unified node index space | Indexed users as `[0, NUM_USERS)` and movies as `[NUM_USERS, NUM_USERS + NUM_MOVIES)` so one adjacency structure covers both node types |

---

## 4. Key Engineering Fixes

- **Gradient detachment bug:** Converting a tensor still attached to the autograd graph directly to NumPy raised a `RuntimeError`. Fixed by calling `.detach().cpu().numpy()` before handing predictions to scikit-learn's `mean_squared_error`.
- **Evaluation batch mismatch:** An early RMSE computation accidentally reused a 10-sample slice instead of the full 20,000-sample test set — corrected to evaluate on the complete held-out test tensor.
- **Rating-to-class mapping:** Since MovieLens ratings are discrete values (0.5–5.0), ratings were mapped to class indices (`rating_to_class` / `class_to_rating`) so GCMC could treat rating prediction as a 10-way classification problem instead of regression.

---

## 5. Results

| Metric | GCMC (Graph-Based) | Funk SVD (Matrix Factorization) |
|---|---|---|
| **RMSE** | **1.20** | **3.62** |
| Strengths | Captures relational topology; robust with limited training data | Simple, fast to compute per step |
| Weaknesses | Higher memory overhead | Needs far more epochs to converge |

| Metric | Baseline GCMC | Enhanced GCMC (+ side features) |
|---|---|---|
| **RMSE** | 1.10 | **1.08** |
| Epochs to reach baseline's final loss | 20 | 16 (4 epochs earlier) |

**Why GCMC outperformed Funk SVD:**
1. **Topology awareness** — even with only 100K of 20M interactions sampled, the graph structure gives GCMC additional context through neighbor connectivity that Funk SVD cannot see.
2. **Distributional prediction** — GCMC predicts a full probability distribution over rating classes, which suits the discrete nature of MovieLens ratings better than Funk SVD's continuous regression target.
3. **Convergence efficiency** — Funk SVD typically needs hundreds of epochs to fully converge; under the time/memory budget used here, GCMC extracted more signal in fewer passes.

**Why side features helped:** Adding genre and user-behavior statistics gave the encoder richer initial node representations, so the message-passing layers started from more informative embeddings — leading to both faster convergence and a lower final RMSE.

---

## 6. GPU Profiling

CUDA event timers and `nvidia-smi` utilization sampling were used to compare GPU vs. CPU execution across all three models. Key finding: GCMC's message-passing computation parallelizes efficiently on GPU (dense batched matrix multiplications), while LLORMA's anchor-based local factorization is inherently sequential (each anchor point requires its own local model fit), making it far less GPU-efficient despite operating on similarly sized data.

---

## 7. Tech Stack

- **Language/Framework:** Python, PyTorch
- **Data handling:** pandas, NumPy
- **Evaluation:** scikit-learn (`train_test_split`, `mean_squared_error`)
- **Visualization:** Matplotlib (training loss convergence curves)
- **Hardware:** NVIDIA Tesla T4 GPU (CUDA)

---

## 8. Repository Structure

```
├── data/
│   ├── ratings.csv          # MovieLens 20M ratings
│   └── movies.csv           # Movie metadata (titles, genres)
├── notebooks/
│   └── GNN_AND_SVD_GPU.ipynb   # Full pipeline: preprocessing, GCMC, Funk SVD, LLORMA, evaluation
├── README.md
└── requirements.txt
```

---

## 9. How to Run

```bash
# Clone the repo
git clone https://github.com/RAJASEKHAR-del857/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install -r requirements.txt

# Download the MovieLens 20M dataset
mkdir -p data
wget https://files.grouplens.org/datasets/movielens/ml-20m.zip -O data/ml-20m.zip
unzip data/ml-20m.zip -d data/
mv data/ml-20m/ratings.csv data/ml-20m/movies.csv data/
rm -rf data/ml-20m data/ml-20m.zip

# (Alternative: download manually from https://grouplens.org/datasets/movielens/20m/
# and place ratings.csv / movies.csv in data/)

# Run the notebook
jupyter notebook notebooks/GNN_AND_SVD_GPU.ipynb
```

---

## 10. Future Improvements

- **Graph sampling at scale:** Use `NeighborLoader` (PyTorch Geometric) to train on the full 20M interactions via mini-batched sub-graphs instead of a 100K subsample.
- **Richer side features:** Incorporate director, release year, and budget for movies; age/occupation/location for users.
- **Bias terms in GCMC:** Add explicit user/item bias terms (as in Funk SVD) to the GCMC prediction to capture systematic rating tendencies.
- **Cold-start handling:** Use side features / popularity-based fallbacks for new users and movies with no interaction history.
- **Hybrid initialization:** Initialize GCMC node embeddings from pretrained Funk SVD latent factors.

---

## Author

**Bommaka Rajasekhar Reddy**
B.Tech Computer Science, IIT Dharwad
[GitHub](https://github.com/RAJASEKHAR-del857) · [LinkedIn](https://linkedin.com/in/bommaka-rajasekhar-reddy)
