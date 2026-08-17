# Movie Recommendation System: GCMC vs. Funk SVD vs. LLORMA

A comparative study of three structurally different collaborative filtering approaches — **Graph Convolutional Matrix Completion (GCMC)**, **Funk SVD**, and **LLORMA** — evaluated on the **MovieLens 20M** dataset.

The project focuses not only on recommendation accuracy, but also on **convergence behavior, CPU/GPU performance, memory consumption, and energy usage** under a constrained **NVIDIA Tesla T4 (15 GB VRAM)** environment.

---

## 1. Project Overview

Recommendation systems can model user-item interactions in fundamentally different ways:

* **GCMC** models users and movies as nodes in a bipartite graph and learns through graph message passing.
* **Funk SVD** represents users and movies as latent vectors and predicts ratings through matrix factorization.
* **LLORMA** combines multiple local low-rank models instead of learning a single global factorization.

This project implements all three approaches and evaluates them under a common experimental setup.

### Objectives

The main objectives were to compare:

* Recommendation accuracy using **RMSE**
* Training and convergence behavior
* CPU vs. GPU execution
* GPU utilization and memory consumption
* CPU and RAM utilization
* Temperature and energy consumption
* Practical implementation and scalability trade-offs

---

## 2. Dataset

The experiments use the **MovieLens 20M** dataset released by GroupLens.

**Dataset statistics:**

| Property       |         Value |
| -------------- | ------------: |
| Ratings        |    20,000,263 |
| Users          |         ~138K |
| Movies         |          ~27K |
| Rating range   |     0.5 – 5.0 |
| Rating classes |            10 |
| Dataset        | MovieLens 20M |

Dataset source:

https://grouplens.org/datasets/movielens/20m/

### Experimental Subsampling

Although the original dataset contains more than **20 million ratings**, training the complete graph-based model directly exceeded the available GPU memory budget.

Therefore, the experiment used:

* **100,000 interactions for training**
* **20,000 interactions for testing**

This allowed all three models to be evaluated under the same constrained experimental environment.

> **Important:** The project uses MovieLens 20M as the source dataset, but the reported experiments are performed on the selected 100K/20K interaction subsets.

---

# 3. Models

## 3.1 Graph Convolutional Matrix Completion (GCMC)

GCMC treats the recommendation problem as a **bipartite graph**.

* Users are represented as one type of node.
* Movies are represented as another type of node.
* Ratings are represented as different edge types.

The MovieLens rating values:

```text
0.5, 1.0, 1.5, ..., 5.0
```

are mapped to **10 rating classes**.

### Encoder

For each rating class, GCMC learns a separate transformation matrix.

Neighbor representations are transformed and aggregated to produce node embeddings:

```text
h_u = ReLU(
        Σ_r Σ_i∈N_r(u) W_r e_i
      )
```

where:

* `W_r` is the learned transformation for rating class `r`
* `N_r(u)` represents neighbors connected through rating type `r`
* `e_i` is the neighboring node representation

### Decoder

The decoder uses a learned bilinear matrix for every rating class:

```text
P(r | u, i) = softmax(uᵀ Q_r i)
```

Instead of directly predicting a single rating, GCMC predicts a **probability distribution over the 10 possible rating classes**.

The final predicted rating is obtained from the expected value of this distribution.

### Loss

The model is trained using **Negative Log-Likelihood (NLL)** over the predicted rating-class distribution.

---

# 4. Enhanced GCMC

In addition to the original graph structure, the GCMC model was extended with **side information**.

### Movie Features

Movie genres were one-hot encoded:

```text
19 movie genres → [NUM_MOVIES, 19]
```

The genre representation is projected into the model's embedding space.

### User Features

Three behavioral statistics were calculated for every user:

```text
mean_rating
rating_count
rating_std
```

These features were normalized and represented as:

```text
[NUM_USERS, 3]
```

They were then projected into the embedding space and incorporated into the node representation before graph message passing.

### Motivation

The original GCMC representation depends primarily on the user-item interaction graph.

The enhanced model provides additional information about:

* What genres a movie belongs to
* A user's average rating tendency
* How frequently the user rates movies
* How consistent the user's ratings are

This gives the graph encoder additional information beyond the interaction topology.

---

# 5. Funk SVD

Funk SVD is a traditional **matrix factorization** approach.

Each user and movie receives a latent embedding:

```text
p_u ∈ R^k
q_i ∈ R^k
```

The predicted rating is:

```text
r̂_ui = μ + b_u + b_i + q_iᵀp_u
```

where:

* `μ` = global average rating
* `b_u` = user bias
* `b_i` = movie bias
* `p_u` = user latent vector
* `q_i` = movie latent vector

The model is trained using **Mean Squared Error (MSE)** with Adam optimization.

Unlike GCMC, Funk SVD does not explicitly model the graph structure or rating-edge types.

---

# 6. LLORMA

**LLORMA (Local Low-Rank Matrix Approximation)** models the rating matrix using multiple local low-rank approximations.

Instead of learning one global factorization, LLORMA:

1. Selects anchor points.
2. Builds local user/item neighborhoods around each anchor.
3. Trains a low-rank model for each local region.
4. Combines predictions using kernel-based weighting.

This allows different regions of the user-item matrix to learn different latent structures.

### GPU Considerations

LLORMA has a different computational profile from GCMC and Funk SVD.

Its anchor-based local factorization involves repeated local model construction and optimization, making the workload substantially less parallel than the dense tensor operations used by GCMC.

This difference was investigated during the CPU/GPU profiling stage.

---

# 7. Memory-Constrained Training

A major part of the project was engineering the models to operate within the memory budget of a **single NVIDIA Tesla T4 GPU with 15 GB VRAM**.

Several optimizations were required.

### 7.1 Interaction Subsampling

The full MovieLens 20M graph was too large for the available memory budget.

Therefore:

```text
Training interactions → 100,000
Testing interactions  → 20,000
```

were selected for the experimental pipeline.

### 7.2 Reduced Embedding Dimensions

Large latent dimensions significantly increase memory consumption.

The experiment therefore used smaller dimensions:

```text
GCMC hidden dimension → 16
Funk SVD latent dimension → 10
```

This reduced the size of learnable parameter matrices and intermediate tensors.

### 7.3 Dense ID Mapping

MovieLens user and movie IDs are not guaranteed to be convenient contiguous indices for embedding lookup.

Two mappings were created:

```python
user_map
movie_map
```

which convert raw IDs into dense zero-based indices.

### 7.4 Unified Bipartite Graph Indexing

Users and movies were represented in a single node-index space:

```text
Users:
[0, NUM_USERS)

Movies:
[NUM_USERS, NUM_USERS + NUM_MOVIES)
```

This allows both node types to be represented within a common adjacency structure.

### 7.5 GPU Memory Management

Between model experiments, GPU memory was explicitly released using:

```python
del model
gc.collect()
torch.cuda.empty_cache()
```

This reduced the risk of out-of-memory errors when switching between different models.

---

# 8. Engineering Fixes

Several implementation issues were identified and fixed during development.

## 8.1 Autograd / NumPy Conversion

Predictions produced by PyTorch were still attached to the autograd computation graph.

Direct conversion to NumPy resulted in an error.

The solution was:

```python
predictions.detach().cpu().numpy()
```

before passing predictions to scikit-learn evaluation functions.

---

## 8.2 Evaluation Batch Mismatch

An early RMSE calculation accidentally evaluated only a small slice of the test data instead of the complete test set.

The evaluation pipeline was corrected to use the full:

```text
20,000-sample test set
```

for the reported RMSE.

---

## 8.3 Rating-Class Conversion

GCMC treats rating prediction as a classification problem.

MovieLens ratings were therefore converted into class indices:

```text
0.5 → class 0
1.0 → class 1
1.5 → class 2
...
5.0 → class 9
```

Mappings were maintained using:

```text
rating_to_class
class_to_rating
```

The predicted class distribution was then converted back into the corresponding rating expectation.

---

# 9. Results

## 9.1 Model Comparison

The models produced substantially different results under the constrained experimental setup.

| Metric           |                   GCMC |                                  Funk SVD |
| ---------------- | ---------------------: | ----------------------------------------: |
| RMSE             |               **1.20** |                                      3.62 |
| Primary approach |            Graph-based |                      Matrix factorization |
| Prediction       |    Rating distribution |                         Continuous rating |
| Main strength    | Relational information |                                Simplicity |
| Main limitation  | Higher memory overhead | Slower convergence under the tested setup |

> LLORMA was also implemented and profiled as part of the comparison, with particular focus on its CPU/GPU execution characteristics.

---

# 10. Baseline vs. Enhanced GCMC

The side-feature extension improved the GCMC model.

| Metric                              | Baseline GCMC | Enhanced GCMC |
| ----------------------------------- | ------------: | ------------: |
| RMSE                                |        1.0977 |    **1.0847** |
| Improvement                         |            -- |     **~1.2%** |
| Epochs to reach baseline final loss |            20 |        **16** |

### Interpretation

The enhanced model combines:

```text
Graph structure
      +
Movie genres
      +
User behavioral statistics
      ↓
Enhanced node representation
      ↓
Graph message passing
      ↓
Rating prediction
```

The additional side information resulted in both:

* Lower final RMSE
* Faster convergence

---

# 11. Why GCMC Performed Better in This Experiment

Under the tested configuration, GCMC achieved substantially lower RMSE than Funk SVD.

Several factors contributed to this result.

### 11.1 Graph Structure

GCMC explicitly uses user-item connectivity.

A user's representation can therefore incorporate information from connected movie nodes and their rating relationships.

Funk SVD instead relies on learned latent vectors without explicitly modeling this graph structure.

### 11.2 Rating Distribution

GCMC predicts a probability distribution over the **10 discrete rating classes**.

This is different from Funk SVD, which directly predicts a continuous rating.

The distributional formulation is naturally aligned with the discrete MovieLens rating values.

### 11.3 Convergence

Under the experimental training budget, GCMC extracted useful information in fewer passes.

Funk SVD required substantially more optimization to approach convergence.

---

# 12. GPU and CPU Profiling

The project also investigated the computational behavior of the three models.

### Tools

```text
CUDA event timers
nvidia-smi
PyTorch CUDA APIs
```

were used to collect performance information.

The profiling considered:

* Execution time
* GPU utilization
* GPU memory usage
* CPU utilization
* RAM usage
* GPU temperature
* Energy consumption

### GCMC

GCMC contains matrix/tensor operations and message-passing computations that can be parallelized efficiently on a GPU.

This makes it well suited to GPU execution when sufficient memory is available.

### Funk SVD

Funk SVD has a relatively simple computational structure based on latent-factor operations.

It generally has a smaller model footprint than GCMC.

### LLORMA

LLORMA's local anchor-based optimization involves repeated local factorization.

These operations are less naturally parallel than GCMC's dense tensor computations.

Consequently, LLORMA exhibited a different CPU/GPU performance profile and was less GPU-friendly in the tested implementation.

---

# 13. Key Engineering Challenges

The project required solving several practical problems beyond implementing the algorithms themselves.

### Challenge 1 — GPU Memory

The full 20M-interaction graph could not be directly trained within the 15 GB T4 memory budget.

**Solution:**

* Interaction subsampling
* Reduced embedding dimensions
* Dense ID remapping
* Explicit CUDA memory cleanup

### Challenge 2 — Different Model Architectures

GCMC, Funk SVD, and LLORMA use fundamentally different computational structures.

**Solution:**

A common experimental pipeline was used for:

```text
Data preprocessing
       ↓
Train/test split
       ↓
Model training
       ↓
Prediction
       ↓
RMSE evaluation
       ↓
Performance profiling
```

### Challenge 3 — Fair Evaluation

Different prediction formats had to be converted into a common rating representation.

**Solution:**

All model outputs were converted to predicted ratings and evaluated using the same **RMSE metric**.

---

# 14. Experimental Pipeline

```text
                 MovieLens 20M
                       |
                       v
              Data preprocessing
                       |
                       v
              Train/Test split
                100K / 20K
                       |
          +------------+------------+
          |            |            |
          v            v            v
        GCMC       Funk SVD      LLORMA
          |            |            |
          v            v            v
       Training     Training     Training
          |            |            |
          +------------+------------+
                       |
                       v
                Rating Prediction
                       |
                       v
                  RMSE Evaluation
                       |
                       v
             Performance Profiling
                       |
          +------------+-------------+
          |            |             |
          v            v             v
        CPU          GPU          Memory
      Runtime     Utilization    Consumption
                       |
                       v
              Model Comparison
```

---

# 15. Technology Stack

### Programming

* Python

### Deep Learning

* PyTorch
* CUDA

### Data Processing

* NumPy
* Pandas

### Evaluation

* Scikit-learn

  * `train_test_split`
  * `mean_squared_error`

### Visualization

* Matplotlib

### Hardware Monitoring

* NVIDIA `nvidia-smi`
* CUDA event timing

### Hardware

* NVIDIA Tesla T4
* 15 GB VRAM

---

# 16. Repository Structure

```text
.
├── data/
│   ├── ratings.csv
│   └── movies.csv
│
├── notebooks/
│   └── GNN_AND_SVD_GPU.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

The notebook contains the complete experimental pipeline:

```text
Preprocessing
     ↓
MovieLens data preparation
     ↓
GCMC
     ↓
Enhanced GCMC
     ↓
Funk SVD
     ↓
LLORMA
     ↓
Evaluation
     ↓
CPU/GPU profiling
     ↓
Visualization
```

---

# 17. Installation

Clone the repository:

```bash
git clone https://github.com/RAJASEKHAR-del857/GNN-s_and_their_Utility.git

cd GNN-s_and_their_Utility
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 18. Dataset Setup

Download the MovieLens 20M dataset from:

https://grouplens.org/datasets/movielens/20m/

Place the required files in:

```text
data/
├── ratings.csv
└── movies.csv
```

The experiment expects the standard MovieLens files:

```text
ratings.csv
movies.csv
```

---

# 19. Running the Experiment

Launch Jupyter:

```bash
jupyter notebook
```

Then open:

```text
notebooks/GNN_AND_SVD_GPU.ipynb
```

Run the notebook from top to bottom.

The notebook performs:

1. Dataset loading
2. User/movie ID mapping
3. Train/test preparation
4. Rating-class conversion
5. GCMC training
6. Enhanced GCMC training
7. Funk SVD training
8. LLORMA training
9. RMSE evaluation
10. CPU/GPU profiling
11. Resource analysis
12. Training-loss visualization

---

# 20. Future Improvements

Several extensions could make the system more scalable and accurate.

### Full Dataset Graph Training

Use graph sampling techniques such as:

```text
PyTorch Geometric NeighborLoader
```

to train GCMC on the complete MovieLens 20M dataset without constructing the entire graph in GPU memory.

### Richer Side Features

Additional movie and user metadata could be incorporated:

```text
Movie:
- Genres
- Release year
- Metadata

User:
- Demographics
- Occupation
- Location
```

### GCMC Bias Terms

Add explicit user and movie bias terms similar to Funk SVD to capture systematic rating tendencies.

### Cold-Start Recommendation

Use side features and popularity-based fallbacks for users or movies with limited interaction history.

### Hybrid Models

Initialize GCMC embeddings using latent representations learned by Funk SVD.

### Distributed Training

Move beyond a single T4 GPU using:

```text
DistributedDataParallel
Multi-GPU training
Graph partitioning
```

for experiments on the complete MovieLens dataset.

---

# 21. Key Takeaways

This project demonstrates the differences between three fundamentally different recommendation approaches:

| Model        | Core Idea                     | Main Advantage                          |
| ------------ | ----------------------------- | --------------------------------------- |
| **GCMC**     | Graph message passing         | Captures user-item relational structure |
| **Funk SVD** | Global low-rank factorization | Simple and computationally efficient    |
| **LLORMA**   | Local low-rank models         | Captures local matrix structure         |

The project also demonstrates that recommendation-model performance is not determined only by prediction accuracy.

**Model architecture, convergence, GPU utilization, memory footprint, execution time, and energy consumption all affect the practical choice of a recommendation algorithm.**

---

## Author

**Bommaka Rajasekhar Reddy**
B.Tech Computer Science and Engineering, IIT Dharwad

* GitHub: https://github.com/RAJASEKHAR-del857
* LinkedIn: https://linkedin.com/in/bommaka-rajasekhar-reddy

```
```
