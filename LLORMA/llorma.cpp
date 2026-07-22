/*
 * LLORMA — Local Low-Rank Matrix Approximation
 * Implementation in C++ for MovieLens ratings.csv
 *
 * Algorithm:
 *   1. Load ratings.csv  →  build user / movie index maps
 *   2. Train / test split (80 / 20)
 *   3. Sample K anchor points from training set
 *   4. For each anchor compute Gaussian kernel weights over ALL training rows
 *   5. For each anchor run weighted SGD to get local user/movie factor matrices
 *   6. Predict = weighted sum of local dot-products across all K anchors
 *   7. Evaluate RMSE on test set
 *
 * Build:
 *   g++ -O2 -std=c++17 -o llorma llorma.cpp
 *
 * Run:
 *   ./llorma ratings.csv
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <numeric>

// ─────────────────────────── hyper-parameters ────────────────────────────────

static const int    K          = 10;     // number of anchor points
static const int    LATENT_DIM = 5;      // dimension of local factor vectors
static const int    EPOCHS     = 20;     // SGD epochs per anchor
static const double LR         = 0.01;  // learning rate
static const double REG        = 0.01;  // L2 regularisation
static const double BANDWIDTH  = 1.0;   // Gaussian kernel bandwidth
static const double MIN_WEIGHT = 1e-6;  // ignore near-zero kernel weights
static const double TEST_RATIO = 0.2;
static const int    SEED       = 42;

// ─────────────────────────── data structures ─────────────────────────────────

struct Rating {
    int    user_idx;
    int    movie_idx;
    double rating;
};

// ─────────────────────────── CSV loader ──────────────────────────────────────

/*
 * Reads ratings.csv (header: userId,movieId,rating,timestamp).
 * Builds sequential 0-based index maps for users and movies.
 * Returns all ratings as a vector of Rating structs.
 */
std::vector<Rating> load_csv(
    const std::string& path,
    std::unordered_map<int,int>& user_map,
    std::unordered_map<int,int>& movie_map)
{
    std::vector<Rating> data;
    std::ifstream file(path);

    if (!file.is_open()) {
        std::cerr << "ERROR: cannot open " << path << "\n";
        std::exit(1);
    }

    std::string line;
    std::getline(file, line); // skip header

    int user_counter  = 0;
    int movie_counter = 0;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string tok;

        std::getline(ss, tok, ','); int raw_user  = std::stoi(tok);
        std::getline(ss, tok, ','); int raw_movie = std::stoi(tok);
        std::getline(ss, tok, ','); double rating = std::stod(tok);
        // timestamp column ignored

        if (user_map.find(raw_user) == user_map.end())
            user_map[raw_user] = user_counter++;

        if (movie_map.find(raw_movie) == movie_map.end())
            movie_map[raw_movie] = movie_counter++;

        data.push_back({ user_map[raw_user], movie_map[raw_movie], rating });
    }

    std::cout << "Loaded   : " << data.size()   << " ratings\n";
    std::cout << "Users    : " << user_map.size()  << "\n";
    std::cout << "Movies   : " << movie_map.size() << "\n";

    return data;
}

// ─────────────────────────── train / test split ───────────────────────────────

void split(
    const std::vector<Rating>& data,
    std::vector<Rating>& train,
    std::vector<Rating>& test,
    double test_ratio,
    int seed)
{
    std::vector<int> idx(data.size());
    std::iota(idx.begin(), idx.end(), 0);

    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);

    int test_size = static_cast<int>(data.size() * test_ratio);

    for (int i = 0; i < test_size; ++i)
        test.push_back(data[idx[i]]);

    for (int i = test_size; i < (int)data.size(); ++i)
        train.push_back(data[idx[i]]);

    std::cout << "Train    : " << train.size() << "\n";
    std::cout << "Test     : " << test.size()  << "\n";
}

// ─────────────────────────── factor matrix helpers ────────────────────────────

/*
 * A factor matrix is stored as a flat vector of size (num_entities × LATENT_DIM).
 * Row i starts at index i * LATENT_DIM.
 */
using FactorMatrix = std::vector<double>;

FactorMatrix make_factors(int n, int dim, std::mt19937& rng)
{
    std::normal_distribution<double> dist(0.0, 0.01);
    FactorMatrix F(n * dim);
    for (auto& v : F) v = dist(rng);
    return F;
}

inline double dot(const FactorMatrix& U, int u,
                  const FactorMatrix& V, int v,
                  int dim)
{
    double s = 0.0;
    const double* pu = U.data() + u * dim;
    const double* pv = V.data() + v * dim;
    for (int d = 0; d < dim; ++d) s += pu[d] * pv[d];
    return s;
}

// ─────────────────────────── Gaussian kernel ─────────────────────────────────

/*
 * Kernel between anchor rating r_a and sample rating r_i.
 * Uses the scalar rating as the "feature vector" (1-D Gaussian).
 */
inline double kernel(double r_anchor, double r_sample, double bw)
{
    double diff = r_anchor - r_sample;
    return std::exp(-(diff * diff) / (2.0 * bw * bw));
}

// ─────────────────────────── local SGD for one anchor ────────────────────────

/*
 * Trains local user / movie factors for anchor k.
 * Only training rows with kernel weight > MIN_WEIGHT are used.
 *
 * Update rule (weighted SGD + L2 reg):
 *   error  = rating - U[u]·V[i]
 *   U[u]  += lr * (w * error * V[i]  - reg * U[u])
 *   V[i]  += lr * (w * error * U[u]  - reg * V[i])
 */
void train_local(
    const std::vector<Rating>& train,
    const std::vector<double>& weights, // kernel weight per training row
    int num_users,
    int num_movies,
    FactorMatrix& U,   // output user factors  [num_users  × LATENT_DIM]
    FactorMatrix& V,   // output movie factors [num_movies × LATENT_DIM]
    int epochs,
    double lr,
    double reg,
    int seed_offset,
    int anchor_id)
{
    std::mt19937 rng(SEED + seed_offset);
    U = make_factors(num_users,  LATENT_DIM, rng);
    V = make_factors(num_movies, LATENT_DIM, rng);

    // collect indices of rows with meaningful weight
    std::vector<int> active;
    active.reserve(train.size());
    for (int i = 0; i < (int)train.size(); ++i)
        if (weights[i] > MIN_WEIGHT) active.push_back(i);

    if (active.empty()) {
        std::cout << "  Anchor " << anchor_id << " : 0 active rows — skipping\n";
        return;
    }

    std::cout << "  Anchor " << std::setw(2) << anchor_id
              << " : " << active.size() << " active rows  →  training...\n";

    for (int ep = 0; ep < epochs; ++ep) {
        std::shuffle(active.begin(), active.end(), rng);

        double total_loss = 0.0;

        for (int idx : active) {
            int    u = train[idx].user_idx;
            int    i = train[idx].movie_idx;
            double r = train[idx].rating;
            double w = weights[idx];

            double pred  = dot(U, u, V, i, LATENT_DIM);
            double error = r - pred;
            total_loss  += w * error * error;

            double* pu = U.data() + u * LATENT_DIM;
            double* pv = V.data() + i * LATENT_DIM;

            for (int d = 0; d < LATENT_DIM; ++d) {
                double pu_d = pu[d];
                double pv_d = pv[d];
                pu[d] += lr * (w * error * pv_d - reg * pu_d);
                pv[d] += lr * (w * error * pu_d - reg * pv_d);
            }
        }

        // print loss every 5 epochs
        if ((ep + 1) % 5 == 0 || ep == 0)
            std::cout << "    Epoch " << std::setw(2) << ep+1
                      << "  weighted-loss = "
                      << std::fixed << std::setprecision(4)
                      << total_loss / active.size() << "\n";
    }
}

// ─────────────────────────── LLORMA model ────────────────────────────────────

struct LLORMAModel {
    int K;
    int num_users;
    int num_movies;

    // anchor info
    std::vector<double> anchor_ratings;  // scalar rating of each anchor

    // local factor matrices per anchor
    std::vector<FactorMatrix> U_list;    // K × [num_users  × LATENT_DIM]
    std::vector<FactorMatrix> V_list;    // K × [num_movies × LATENT_DIM]
};

/*
 * Predict rating for (user_idx, movie_idx) using the trained LLORMA model.
 * Prediction = ( Σ_k  w_k * U_k[u]·V_k[i] ) / ( Σ_k w_k )
 * where w_k = kernel(anchor_k_rating, query_rating_proxy)
 *
 * Since at prediction time we don't know the true rating, we use the
 * global mean rating as the query proxy — a standard approximation.
 */
double predict(
    const LLORMAModel& model,
    int u,
    int i,
    double query_rating_proxy,
    double bw)
{
    double num = 0.0, denom = 0.0;

    for (int k = 0; k < model.K; ++k) {
        double w = kernel(model.anchor_ratings[k], query_rating_proxy, bw);
        if (w < MIN_WEIGHT) continue;

        // safety: user or movie may not appear in local training data
        // factors are zero-initialised so dot product will be 0
        double d = dot(model.U_list[k], u, model.V_list[k], i, LATENT_DIM);
        num   += w * d;
        denom += w;
    }

    if (denom < MIN_WEIGHT) return query_rating_proxy; // fallback to mean
    return num / denom;
}

// ─────────────────────────── RMSE ────────────────────────────────────────────

double compute_rmse(
    const LLORMAModel& model,
    const std::vector<Rating>& test,
    double global_mean,
    double bw)
{
    double sse = 0.0;
    for (const auto& r : test) {
        double pred = predict(model, r.user_idx, r.movie_idx, global_mean, bw);
        double diff = r.rating - pred;
        sse += diff * diff;
    }
    return std::sqrt(sse / test.size());
}

// ─────────────────────────── main ────────────────────────────────────────────

int main(int argc, char* argv[])
{
    std::string csv_path = "ratings.csv";
    if (argc > 1) csv_path = argv[1];

    std::cout << "===========================================\n";
    std::cout << "  LLORMA — Local Low-Rank Matrix Approx.\n";
    std::cout << "===========================================\n\n";

    // ── 1. Load data ──────────────────────────────────────────────────────────
    std::cout << "[1] Loading data...\n";
    std::unordered_map<int,int> user_map, movie_map;
    auto data = load_csv(csv_path, user_map, movie_map);

    int NUM_USERS  = (int)user_map.size();
    int NUM_MOVIES = (int)movie_map.size();

    // ── 2. Train / test split ─────────────────────────────────────────────────
    std::cout << "\n[2] Splitting train/test...\n";
    std::vector<Rating> train, test;
    split(data, train, test, TEST_RATIO, SEED);

    // ── 3. Global mean (used as rating proxy at prediction time) ──────────────
    double global_mean = 0.0;
    for (const auto& r : train) global_mean += r.rating;
    global_mean /= train.size();
    std::cout << "\nGlobal mean rating: "
              << std::fixed << std::setprecision(4) << global_mean << "\n";

    // ── 4. Sample K anchor points ─────────────────────────────────────────────
    std::cout << "\n[3] Sampling " << K << " anchor points...\n";
    std::mt19937 rng(SEED);
    std::uniform_int_distribution<int> dist(0, (int)train.size() - 1);

    std::vector<int>    anchor_indices;
    std::vector<double> anchor_ratings;

    // ensure unique anchors
    std::unordered_set<int> chosen;
    while ((int)anchor_indices.size() < K) {
        int idx = dist(rng);
        if (chosen.insert(idx).second) {
            anchor_indices.push_back(idx);
            anchor_ratings.push_back(train[idx].rating);
            std::cout << "  Anchor " << anchor_indices.size()
                      << " → train[" << idx << "]"
                      << "  user=" << train[idx].user_idx
                      << "  movie=" << train[idx].movie_idx
                      << "  rating=" << train[idx].rating << "\n";
        }
    }

    // ── 5. Compute kernel weights for every anchor ────────────────────────────
    std::cout << "\n[4] Computing kernel weights...\n";
    // weights[k][i] = kernel weight of training row i for anchor k
    std::vector<std::vector<double>> all_weights(K, std::vector<double>(train.size()));

    for (int k = 0; k < K; ++k) {
        double r_a = anchor_ratings[k];
        int active_count = 0;
        for (int i = 0; i < (int)train.size(); ++i) {
            all_weights[k][i] = kernel(r_a, train[i].rating, BANDWIDTH);
            if (all_weights[k][i] > MIN_WEIGHT) ++active_count;
        }
        std::cout << "  Anchor " << std::setw(2) << k+1
                  << " (r=" << r_a << ")  active rows: " << active_count << "\n";
    }

    // ── 6. Train local models ─────────────────────────────────────────────────
    std::cout << "\n[5] Training local models (" << EPOCHS << " epochs each)...\n";
    auto t_start = std::chrono::high_resolution_clock::now();

    LLORMAModel model;
    model.K          = K;
    model.num_users  = NUM_USERS;
    model.num_movies = NUM_MOVIES;
    model.anchor_ratings = anchor_ratings;
    model.U_list.resize(K);
    model.V_list.resize(K);

    for (int k = 0; k < K; ++k) {
        std::cout << "\n--- Local model for Anchor " << k+1 << " ---\n";
        train_local(
            train,
            all_weights[k],
            NUM_USERS,
            NUM_MOVIES,
            model.U_list[k],
            model.V_list[k],
            EPOCHS, LR, REG,
            k,      // seed offset so each anchor starts differently
            k + 1
        );
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\nTotal training time: "
              << std::fixed << std::setprecision(2) << elapsed << " seconds\n";

    // ── 7. Evaluate RMSE on test set ──────────────────────────────────────────
    std::cout << "\n[6] Evaluating on test set (" << test.size() << " ratings)...\n";

    auto t_eval_start = std::chrono::high_resolution_clock::now();
    double rmse = compute_rmse(model, test, global_mean, BANDWIDTH);
    auto t_eval_end = std::chrono::high_resolution_clock::now();
    double eval_time = std::chrono::duration<double>(t_eval_end - t_eval_start).count();

    std::cout << "\n===========================================\n";
    std::cout << "  LLORMA RESULTS\n";
    std::cout << "===========================================\n";
    std::cout << "  K (anchors)    : " << K           << "\n";
    std::cout << "  Latent dim     : " << LATENT_DIM  << "\n";
    std::cout << "  Epochs/anchor  : " << EPOCHS      << "\n";
    std::cout << "  Bandwidth      : " << BANDWIDTH   << "\n";
    std::cout << "  Train size     : " << train.size()<< "\n";
    std::cout << "  Test  size     : " << test.size() << "\n";
    std::cout << "  Training time  : " << std::fixed << std::setprecision(2)
              << elapsed   << " s\n";
    std::cout << "  Eval time      : " << std::fixed << std::setprecision(2)
              << eval_time << " s\n";
    std::cout << "-------------------------------------------\n";
    std::cout << "  RMSE           : " << std::fixed << std::setprecision(4)
              << rmse << "\n";
    std::cout << "===========================================\n";

    return 0;
}
