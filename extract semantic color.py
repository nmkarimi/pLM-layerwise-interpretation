import os
import pandas as pd
import json
import concurrent.futures
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, kendalltau, pearsonr
import numpy as np
from skimage.color import deltaE_ciede2000, lab2rgb
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
import skfuzzy as fuzz
from multiprocessing import Pool
import multiprocessing
import joblib

# Base path for files on the server
BASE_PATH = "/home/mehdi/Documents/revise_hypo_2/t5/24"
CHECKPOINT_DIR = os.path.join(BASE_PATH, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def match_score_func(dif_c, thre):
    if dif_c <= thre:
        score_match = abs(thre - dif_c)
        gap_score = abs(thre - dif_c) / 5
    else:
        score_match = -1 * abs(dif_c - thre)
        gap_score = abs(dif_c - thre) / 5
    return score_match, gap_score

def smith_waterman_score(seq1, seq2, color1, color2, threshold):
    rows, cols = len(seq1), len(seq2)
    H = np.zeros((rows + 1, cols + 1))
    max_score = 0.0
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            match_score, gap_penalty = match_score_func(deltaE_ciede2000(color1[i-1], color2[j-1]), threshold)
            H[i, j] = max(
                0,
                H[i - 1, j - 1] + match_score,
                H[i - 1, j] - gap_penalty,
                H[i, j - 1] - gap_penalty
            )
            if H[i, j] > max_score:
                max_score = H[i, j]
    return max_score / (min(rows, cols) * threshold)

def process_row(bb):
    row = bb[0]
    threshold = bb[1]
    return smith_waterman_score(row['sequence1'], row['sequence2'], row['color_0'], row['color_1'], threshold)

def delta_e_matrix(colors):
    num_colors = colors.shape[0]
    de_matrix = np.zeros((num_colors, num_colors))
    for i in range(num_colors):
        for j in range(num_colors):
            if i != j:
                de_matrix[i, j] = deltaE_ciede2000(colors[i], colors[j])
    return de_matrix

class ColorOptimizationProblem(ElementwiseProblem):
    def __init__(self, num_clusters, distance_matrix, initial_data, data, label, elementwise_runner=None):
        n_var = num_clusters * 3
        xl = np.array([0, -128, -128] * num_clusters)
        xu = np.array([100, 127, 127] * num_clusters)
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=xl, xu=xu, elementwise_runner=elementwise_runner)
        self.num_clusters = num_clusters
        self.distance_matrix = distance_matrix

    def _evaluate(self, x, out, *args, **kwargs):
        colors_lab = x.reshape((self.num_clusters, 3))
        color_diff_matrix = delta_e_matrix(colors_lab)
        color_diff_sorted_indices = np.argsort(color_diff_matrix, axis=1)
        distance_sorted_indices = np.argsort(self.distance_matrix, axis=1)

        score1 = np.sum(color_diff_sorted_indices.flatten() != distance_sorted_indices.flatten())
        score2 = np.sum((color_diff_matrix > 30) & (color_diff_matrix < 45))

        out["F"] = [score1, -1 * score2]

def optimize_colors_nsga2(distance_matrix, num_clusters, initial_data, data, label, k):
    n_threads = max(2, min(16, multiprocessing.cpu_count() - 2))
    pool = Pool(n_threads)
    runner = StarmapParallelization(pool.starmap)
    problem = ColorOptimizationProblem(num_clusters, distance_matrix, initial_data, data, label, elementwise_runner=runner)

    n_obj = 2
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
    algorithm = NSGA3(
        pop_size=200, 
        ref_dirs=ref_dirs, 
        mutation=PolynomialMutation(prob=0.15), 
        crossover=SimulatedBinaryCrossover(prob=0.9)
    )

    termination = DefaultMultiObjectiveTermination(xtol=1e-5, cvtol=1e-6, ftol=0.0005, period=20, n_max_gen=500)
    res = minimize(problem, algorithm, termination, seed=1, verbose=False)

    pool.close()
    pool.join()
    optimal_colors_lab = res.X[np.argmin(res.F[:, 0])].reshape((num_clusters, 3))
    return optimal_colors_lab

def get_combined_color(membership, colors):
    return np.sum(membership[:, np.newaxis] * colors, axis=0)

def get_color_for_embedding(input_embedding, k):
    data = np.array(input_embedding)
    best_num_clusters = 7
    best_m = 1.6
    centers, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data.T, best_num_clusters, best_m, error=0.005, maxiter=30000, metric='cosine'
    )

    joblib.dump({'centers': centers, 'u': u}, os.path.join(CHECKPOINT_DIR, f'cmeans_model_{k}.joblib'))

    U = u.T
    distance_matrix_centers = 1 - cosine_similarity(centers)
    label = np.argmax(u, axis=0)

    optimal_colors_lab = optimize_colors_nsga2(distance_matrix_centers, best_num_clusters, centers, data, label, k)

    rgb_colors = []
    lab_per_point = []
    for i in range(len(data)):
        combined_lab = get_combined_color(u[:, i], optimal_colors_lab)
        lab_per_point.append(combined_lab)
        rgb_colors.append(list(lab2rgb(combined_lab.reshape(1, 1, 3)).flatten() * 255))

    return [np.array(lab_per_point), None, U, len(centers), data, best_m, centers, label, None, rgb_colors]

CACHED_TRAIN_DF = None

def get_prepared_train_data():
    global CACHED_TRAIN_DF
    if CACHED_TRAIN_DF is not None:
        return CACHED_TRAIN_DF

    train_combined_path = os.path.join(CHECKPOINT_DIR, "train_all_colored.parquet")
    if os.path.exists(train_combined_path):
        print("Found fully preprocessed Train data. Loading from checkpoint...")
        CACHED_TRAIN_DF = pd.read_parquet(train_combined_path)
        return CACHED_TRAIN_DF

    print("Checking and preparing Training batches...")
    all_dfs = []
    for kk in range(int(900 / 50)):
        batch_save_path = os.path.join(CHECKPOINT_DIR, f"colored_train_batch_{kk}.parquet")
        
        if os.path.exists(batch_save_path):
            print(f"Loading already processed Train batch {kk} from disk...")
            df_batch = pd.read_parquet(batch_save_path)
            all_dfs.append(df_batch)
            continue

        input_file = os.path.join(BASE_PATH, f"processed_embedding_T5_{kk}.csv")
        if not os.path.exists(input_file):
            continue

        print(f"Coloring Train batch {kk} (Fresh computation)...")
        df_results = pd.read_csv(input_file)
        df_results['embedding_0'] = df_results['embedding_0'].apply(json.loads)
        df_results['embedding_1'] = df_results['embedding_1'].apply(json.loads)

        X = []
        for _, row in df_results.iterrows():
            X.extend(row['embedding_0'])
            X.extend(row['embedding_1'])
        X = np.array(X)

        res_list = get_color_for_embedding(X, kk)
        lab_colors = res_list[0]
        rgb_colors = res_list[9]

        a = 0
        df_results['color_0'] = None
        df_results['color_1'] = None
        df_results['color_plot_0'] = None
        df_results['color_plot_1'] = None

        for index, row in df_results.iterrows():
            l1, l2 = len(row['sequence1']), len(row['sequence2'])
            df_results.at[index, 'color_0'] = lab_colors[a: a + l1].tolist()
            df_results.at[index, 'color_1'] = lab_colors[a + l1: a + l1 + l2].tolist()
            df_results.at[index, 'color_plot_0'] = rgb_colors[a: a + l1]
            df_results.at[index, 'color_plot_1'] = rgb_colors[a + l1: a + l1 + l2]
            a += l1 + l2

        df_results = df_results.drop(columns=['embedding_0', 'embedding_1'])
        df_results.to_parquet(batch_save_path, index=False)
        all_dfs.append(df_results)

    CACHED_TRAIN_DF = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    if not CACHED_TRAIN_DF.empty:
        CACHED_TRAIN_DF.to_parquet(train_combined_path, index=False)
    print(f"Training dataset ready: {len(CACHED_TRAIN_DF)} pairs.")
    return CACHED_TRAIN_DF

def print_correlations(split_name, score_smith, tm_scores, seq_sim):
    p_tm, _ = pearsonr(score_smith, tm_scores)
    s_tm, _ = spearmanr(score_smith, tm_scores)
    k_tm, _ = kendalltau(score_smith, tm_scores)

    print(f"\n--- {split_name} Structure Correlation (TM_score) ---")
    print(f"Pearson (R):    {p_tm:.4f}")
    print(f"Spearman (R):   {s_tm:.4f}")
    print(f"Kendall Tau (R): {k_tm:.4f}")

    if seq_sim is not None:
        p_seq, _ = pearsonr(score_smith, seq_sim)
        s_seq, _ = spearmanr(score_smith, seq_sim)
        k_seq, _ = kendalltau(score_smith, seq_sim)
        print(f"--- {split_name} Sequence Correlation (Seq_sim) ---")
        print(f"Pearson (S):    {p_seq:.4f}")
        print(f"Spearman (S):   {s_seq:.4f}")
        print(f"Kendall Tau (S): {k_seq:.4f}")
    
    return p_tm

def objective(threshold):
    th_key = str(threshold)
    print(f"\n==========================================")
    print(f"Evaluating Threshold: {th_key}")
    print(f"==========================================")
    df_results = get_prepared_train_data()
    if df_results.empty:
        return 0.0

    file_path = os.path.join(BASE_PATH, "t5_24_color_train.xlsx")
    existing_df = None
    if os.path.exists(file_path):
        try:
            existing_df = pd.read_excel(file_path)
        except Exception:
            existing_df = None

    # If this threshold has already been evaluated, load from cached excel file
    if existing_df is not None and th_key in existing_df.columns:
        print(f"Threshold {th_key} already evaluated. Reading from cached excel...")
        score_smith = existing_df[th_key].tolist()
    else:
        num_workers = max(1, multiprocessing.cpu_count() - 2)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            score_smith = list(executor.map(process_row, [[row, threshold] for _, row in df_results.iterrows()]))

        if existing_df is not None:
            existing_df[th_key] = score_smith
            existing_df.to_excel(file_path, index=False)
        else:
            df_to_save = df_results.copy()
            df_to_save[th_key] = score_smith
            df_to_save.to_excel(file_path, index=False)

    tm_scores = df_results['TM_score'].tolist()
    sim_col = 'sequence_similarity' if 'sequence_similarity' in df_results.columns else ('similarity_score' if 'similarity_score' in df_results.columns else None)
    seq_sim = df_results[sim_col].tolist() if sim_col else None

    # Compute and display all three correlations for structure and sequence during training
    p_tm = print_correlations("TRAIN", score_smith, tm_scores, seq_sim)
    return p_tm

def Test(best_threshold):
    best_th_key = str(best_threshold)
    test_file_path = os.path.join(BASE_PATH, "T5_24_color_test.xlsx")
    
    print(f"\n==========================================")
    print(f"Running Test Evaluation with Threshold: {best_th_key}")
    print(f"==========================================")
    
    all_test_dfs = []
    for kk in range(int(900 / 50), int(1133 / 50) + 1):
        batch_save_path = os.path.join(CHECKPOINT_DIR, f"colored_test_batch_{kk}.parquet")

        if os.path.exists(batch_save_path):
            print(f"Loading already processed Test batch {kk} from disk...")
            df_results = pd.read_parquet(batch_save_path)
        else:
            input_file = os.path.join(BASE_PATH, f"processed_embedding_T5_{kk}.csv")
            if not os.path.exists(input_file):
                continue

            print(f"Coloring Test batch {kk}...")
            df_results = pd.read_csv(input_file)
            df_results['embedding_0'] = df_results['embedding_0'].apply(json.loads)
            df_results['embedding_1'] = df_results['embedding_1'].apply(json.loads)

            X = []
            for _, row in df_results.iterrows():
                X.extend(row['embedding_0'])
                X.extend(row['embedding_1'])
            X = np.array(X)

            res_list = get_color_for_embedding(X, kk)
            lab_colors = res_list[0]
            rgb_colors = res_list[9]

            a = 0
            df_results['color_0'] = None
            df_results['color_1'] = None
            df_results['color_plot_0'] = None
            df_results['color_plot_1'] = None

            for index, row in df_results.iterrows():
                l1, l2 = len(row['sequence1']), len(row['sequence2'])
                df_results.at[index, 'color_0'] = lab_colors[a: a + l1].tolist()
                df_results.at[index, 'color_1'] = lab_colors[a + l1: a + l1 + l2].tolist()
                df_results.at[index, 'color_plot_0'] = rgb_colors[a: a + l1]
                df_results.at[index, 'color_plot_1'] = rgb_colors[a + l1: a + l1 + l2]
                a += l1 + l2

            df_results = df_results.drop(columns=['embedding_0', 'embedding_1'])
            df_results.to_parquet(batch_save_path, index=False)

        num_workers = max(1, multiprocessing.cpu_count() - 2)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_row, [[row, best_threshold] for _, row in df_results.iterrows()]))

        df_results[best_th_key] = results
        all_test_dfs.append(df_results)

    if not all_test_dfs:
        print("No test data found!")
        return 0.0

    df_test_final = pd.concat(all_test_dfs, ignore_index=True)
    df_test_final.to_excel(test_file_path, index=False)

    score_smith = df_test_final[best_th_key].tolist()
    tm_scores = df_test_final['TM_score'].tolist()
    sim_col = 'sequence_similarity' if 'sequence_similarity' in df_test_final.columns else ('similarity_score' if 'similarity_score' in df_test_final.columns else None)
    seq_sim = df_test_final[sim_col].tolist() if sim_col else None

    p_corr = print_correlations("FINAL TEST", score_smith, tm_scores, seq_sim)
    return p_corr

if __name__ == '__main__':
    log_file = os.path.join(CHECKPOINT_DIR, "bayes_logs.json")
    pbounds = {'threshold': (5.0, 100.0)}
    optimizer = BayesianOptimization(
        f=objective, 
        pbounds=pbounds, 
        random_state=42, 
        allow_duplicate_points=True
    )

    if os.path.exists(log_file):
        print(f"Resuming Bayesian Optimization from log: {log_file}")
        load_logs(optimizer, logs=[log_file])
        print(f"Loaded {len(optimizer.space)} prior evaluations.")

    logger = JSONLogger(path=log_file)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    total_target_evals = 22  # 10 initial points + 12 optimization iterations
    current_evals = len(optimizer.space)
    remaining_evals = max(0, total_target_evals - current_evals)

    if remaining_evals > 0:
        init_pts = max(0, 10 - current_evals)
        n_iters = remaining_evals - init_pts
        print(f"Continuing with init_points={init_pts} and n_iter={n_iters}...")
        optimizer.maximize(init_points=init_pts, n_iter=n_iters)
    else:
        print("Optimization already completed! Proceeding with best parameters.")

    best_threshold = optimizer.max['params']['threshold']
    print(f"\n==========================================")
    print(f"Best Threshold Identified: {best_threshold}")
    print(f"==========================================\n")
    
    Test(best_threshold)
