# --- START OF FILE graph.py ---

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, pairwise_distances # Added pairwise_distances
from sklearn.cluster import KMeans
import networkx as nx
import math
import random
from collections import Counter, defaultdict
from scipy.stats import skew, kurtosis
import traceback
import pickle # Added for saving test

print("\n--- Graph Construction Setup ---")

# --- Parameters ---
FEATURE_THRESHOLD = 10.0  # Threshold for 'active' points in feature extraction
MAX_K_TO_TEST = 15       # Maximum number of clusters to evaluate for analysis (if run)
RANDOM_SEED = 42
# <<< IMPORTANT: This should ideally be determined after analyzing the training data features >>>
# <<< Set a default, but it can be overridden by analysis in construct_graph_nodes_and_adjacency >>>
DEFAULT_N_CLUSTERS = 9
N_SAMPLES_PLOT_PER_CLUSTER = 3 # Number of example windows to plot for each cluster assignment vis

# Set random seeds for reproducibility in clustering
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# --- Feature Extraction Function (Copied from hybrid graph v1.py, assumed verified) ---
def extract_features(window, threshold=FEATURE_THRESHOLD):
    """
    Extracts a 15-dimensional feature vector from a single time-series window.

    Args:
        window (np.ndarray): 1D array representing the time-series window.
        threshold (float): Threshold to determine 'active' points.

    Returns:
        np.ndarray: 15-dimensional feature vector. Returns None if input is invalid.
    """
    if not isinstance(window, np.ndarray) or window.ndim != 1 or window.size == 0:
        print("Warning: Invalid input window for feature extraction. Returning None.")
        return None # Return None for invalid input

    features = []
    epsilon = 1e-9 # To avoid division by zero or issues with zero std dev

    # Basic Stats
    mean_val = np.mean(window)
    std_val = np.std(window)
    features.append(mean_val)
    features.append(std_val)
    features.append(np.max(window))
    features.append(np.min(window))
    features.append(np.sqrt(np.mean(window**2))) # RMS

    # Shape Stats (handle low std dev)
    features.append(skew(window) if std_val > epsilon else 0)
    features.append(kurtosis(window) if std_val > epsilon else 0)

    # Active Power Stats
    active_indices = np.where(window > threshold)[0]
    n_active = len(active_indices)
    features.append(n_active)

    if n_active > 0:
        active_signal = window[active_indices]
        features.append(np.mean(active_signal))
        features.append(np.sum(active_signal)) # Energy approximation
        features.append(np.max(active_signal))
        # Peak count in active signal (simple difference method)
        if len(active_signal) > 2:
            diffs = np.diff(active_signal)
            # Count points where slope changes from positive to negative
            peaks = np.sum((diffs[:-1] > 0) & (diffs[1:] < 0))
            features.append(peaks)
        else:
            features.append(0) # Not enough points for peaks
    else:
        # Append zeros if no active power
        features.extend([0, 0, 0, 0])

    # FFT Features (handle potential errors)
    try:
        n_fft = len(window)
        if n_fft > 1: # Need at least 2 points for FFT
             fft_vals = np.fft.rfft(window) # Real FFT for real signal
             fft_mag_sq = np.abs(fft_vals)**2 # Power spectrum
             n_rfft = len(fft_mag_sq)

             # Define frequency bins (simple split into 3)
             bin1_end = max(1, n_rfft // 3) # Ensure at least one element
             bin2_end = max(bin1_end + 1, 2 * n_rfft // 3) # Ensure progression

             energy_bin1 = np.sum(fft_mag_sq[0:bin1_end])
             energy_bin2 = np.sum(fft_mag_sq[bin1_end:bin2_end])
             energy_bin3 = np.sum(fft_mag_sq[bin2_end:])
             total_energy = energy_bin1 + energy_bin2 + energy_bin3 + epsilon # Avoid division by zero

             features.extend([energy_bin1 / total_energy,
                              energy_bin2 / total_energy,
                              energy_bin3 / total_energy])
        else:
             features.extend([0, 0, 0]) # Not enough points for FFT
    except Exception as fft_e:
        print(f"Warning: FFT error - {fft_e}. Appending zeros for FFT features.")
        features.extend([0, 0, 0])

    # Ensure features are finite and numeric, handle NaNs/Infs
    final_features = np.nan_to_num(np.array(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # Double check after nan_to_num
    if np.any(np.isinf(final_features)) or np.any(np.isnan(final_features)):
        print("Warning: NaNs or Infs still detected after nan_to_num. Applying again.")
        final_features = np.nan_to_num(final_features, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure correct dimension (should be 15)
    if len(final_features) != 15:
         print(f"Warning: Feature vector length is {len(final_features)}, expected 15. Padding/truncating if necessary.")
         if len(final_features) < 15:
              final_features = np.pad(final_features, (0, 15 - len(final_features)), 'constant', constant_values=0.0)
         elif len(final_features) > 15:
              final_features = final_features[:15]

    return final_features

# --- Cluster Analysis Function (from graph_2.py, slightly modified) ---
def analyze_clusters_features(scaled_train_features, max_k=MAX_K_TO_TEST, plot=True):
    """
    Analyzes the optimal number of clusters (K) for feature vectors using
    the Elbow method (inertia) and Silhouette score. Operates only on training data features.

    Args:
        scaled_train_features (np.ndarray): Scaled feature vectors for the training set [N_train, n_features=15].
        max_k (int): Maximum K value to test.
        plot (bool): Whether to display the analysis plots.

    Returns:
        int: Suggested K based on analysis (e.g., highest silhouette score), or None if analysis fails.
    """
    print(f"\n--- Analyzing Optimal K for Features (Train Data, up to K={max_k}) ---")
    n_train_samples = scaled_train_features.shape[0]

    if scaled_train_features.ndim != 2 or n_train_samples < 2 or scaled_train_features.shape[1] == 0:
        print("  Warning: Insufficient/Invalid training features for cluster analysis. Cannot determine optimal K.")
        return None

    # K must be less than the number of samples
    actual_max_k = min(max_k, n_train_samples - 1)

    if actual_max_k < 2:
        print(f"  Warning: Only {n_train_samples} training samples available. Cannot perform clustering analysis (K must be >= 2).")
        return None

    k_range = range(2, actual_max_k + 1)
    print(f"  Testing K values on {n_train_samples} training features: {list(k_range)}")

    inertia = []
    silhouette_avg = []
    best_k_silhouette = -1
    max_silhouette = -1.1 # Silhouette score is between -1 and 1

    for k in k_range:
        current_inertia = np.nan
        current_silhouette = np.nan
        try:
            kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10, init='k-means++')
            cluster_labels = kmeans.fit_predict(scaled_train_features)
            current_inertia = kmeans.inertia_

            # Silhouette score requires at least 2 unique clusters formed
            n_unique_labels = len(np.unique(cluster_labels))
            if n_unique_labels > 1:
                current_silhouette = silhouette_score(scaled_train_features, cluster_labels)
                if current_silhouette > max_silhouette:
                    max_silhouette = current_silhouette
                    best_k_silhouette = k
            # else: print(f"    Warning: Only {n_unique_labels} cluster found for K={k}. Cannot compute Silhouette score.")

        except Exception as e:
            print(f"    Error during clustering analysis for K={k}: {e}")
            # Don't stop analysis, just record NaN for this K
        inertia.append(current_inertia)
        silhouette_avg.append(current_silhouette)

    if plot:
        plt.figure(figsize=(14, 6))
        valid_inertia_idx = ~np.isnan(inertia)
        valid_silhouette_idx = ~np.isnan(silhouette_avg)
        k_range_array = np.array(list(k_range))

        plt.subplot(1, 2, 1)
        if np.any(valid_inertia_idx):
            plt.plot(k_range_array[valid_inertia_idx], np.array(inertia)[valid_inertia_idx], marker='o')
            plt.xticks(k_range_array[valid_inertia_idx])
            plt.xlabel('Number of Clusters (K)')
            plt.ylabel('Inertia (WCSS)')
            plt.title('Elbow Method (Training Features)')
            plt.grid(True, linestyle=':')
        else:
            plt.text(0.5, 0.5, "Inertia values N/A", ha='center', va='center')

        plt.subplot(1, 2, 2)
        if np.any(valid_silhouette_idx):
            plt.plot(k_range_array[valid_silhouette_idx], np.array(silhouette_avg)[valid_silhouette_idx], marker='o')
            plt.xticks(k_range_array[valid_silhouette_idx])
            plt.xlabel('Number of Clusters (K)')
            plt.ylabel('Average Silhouette Score')
            plt.title('Silhouette Analysis (Training Features)')
            plt.grid(True, linestyle=':')
            if best_k_silhouette > 0:
                 plt.axvline(x=best_k_silhouette, color='red', linestyle='--', label=f'Best K (Silhouette) = {best_k_silhouette}')
                 plt.legend()
        else:
             plt.text(0.5, 0.5, "Silhouette scores N/A", ha='center', va='center')

        plt.tight_layout()
        plt.show()

    print("--- Feature Cluster Analysis (Train Data) Done ---")

    if best_k_silhouette > 0:
         print(f"Suggested K based on highest Silhouette score: {best_k_silhouette} (Score: {max_silhouette:.4f})")
         return best_k_silhouette
    else:
         print(f"Could not determine a suggested K from Silhouette scores.")
         # Could add logic here to look at elbow plot 'knee' if silhouette fails
         return None


# --- Clustering and Node Feature Calculation ---
def cluster_and_calculate_nodes(scaled_all_features, all_signals_padded, train_indices, n_clusters):
    """
    Performs KMeans clustering on training features, assigns labels to all,
    calculates feature centroids, and calculates average signal embeddings (node features).

    Args:
        scaled_all_features (np.ndarray): Scaled feature vectors for *all* windows [N_total, n_features=15].
        all_signals_padded (np.ndarray): Padded signal windows for *all* samples [N_total, seq_len].
                                         Should be the version (scaled/unscaled) the model will use.
        train_indices (list): Indices corresponding to the training windows.
        n_clusters (int): The chosen number of clusters (K).

    Returns:
        tuple: (kmeans_model, all_labels, feature_centroids, average_signal_node_features, actual_n_clusters)
               - KMeans: The fitted KMeans model object.
               - np.ndarray: Cluster labels assigned to *all* windows [N_total].
               - np.ndarray: Centroids of the scaled *features* [n_clusters, n_features=15].
               - np.ndarray: Average signal embeddings (node features) [n_clusters, seq_len].
               - int: The actual number of clusters used (may differ from requested if adjusted).
        Returns None if clustering fails.
    """
    print(f"\n--- Clustering Features into {n_clusters} Clusters & Calculating Node Features ---")
    scaled_train_features = scaled_all_features[train_indices]
    n_train_samples = scaled_train_features.shape[0]
    seq_len = all_signals_padded.shape[1]

    if n_train_samples < n_clusters:
        print(f"Warning: Number of training samples ({n_train_samples}) is less than n_clusters ({n_clusters}). Reducing n_clusters to {n_train_samples}.")
        n_clusters = n_train_samples
        if n_clusters <= 1:
             print("Error: Cannot perform clustering with <= 1 training sample/cluster.")
             return None # Indicate failure

    print(f"  Fitting KMeans on {n_train_samples} training features (K={n_clusters})...")
    try:
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10, init='k-means++')
        # Fit only on training features
        train_labels = kmeans_model.fit_predict(scaled_train_features)
        feature_centroids = kmeans_model.cluster_centers_ # Shape: [n_clusters, 15]
        print("  KMeans fitting complete.")

        # Verify training clustering results
        train_cluster_counts = dict(Counter(train_labels))
        n_found_train = len(train_cluster_counts)
        print(f"  Training cluster label counts: {train_cluster_counts}")
        if n_found_train < n_clusters:
            print(f"  Warning: Found only {n_found_train} non-empty clusters in training data (K={n_clusters}). Centroids exist for all K, but some might represent empty training clusters.")

        # Predict labels for ALL windows using the fitted model
        print(f"  Predicting cluster labels for all {scaled_all_features.shape[0]} windows...")
        all_labels = kmeans_model.predict(scaled_all_features) # Shape: [N_total]
        print(f"  Overall cluster label counts: {dict(Counter(all_labels))}")
        print(f"  Feature Centroids shape: {feature_centroids.shape}")

        # Calculate Average Signal Embeddings (Node Features for GCN)
        print(f"  Calculating average signal embeddings for {n_clusters} nodes (using training windows)...")
        average_signal_node_features = np.zeros((n_clusters, seq_len), dtype=np.float32)
        counts_per_cluster_train = np.zeros(n_clusters, dtype=int)

        # Map train_indices to their position within the training set for indexing train_labels
        train_indices_map = {original_idx: i for i, original_idx in enumerate(train_indices)}

        for original_idx in train_indices:
            # Find the index within the train_labels array
            train_array_idx = train_indices_map[original_idx]
            cluster_label = train_labels[train_array_idx]

            if 0 <= cluster_label < n_clusters:
                average_signal_node_features[cluster_label] += all_signals_padded[original_idx]
                counts_per_cluster_train[cluster_label] += 1
            else:
                 print(f"Warning: Invalid cluster label {cluster_label} encountered for training index {original_idx}.")


        # Compute the average
        for k in range(n_clusters):
            if counts_per_cluster_train[k] > 0:
                average_signal_node_features[k] /= counts_per_cluster_train[k]
            else:
                # Handle clusters with no training samples assigned
                print(f"  Warning: Cluster {k} has no training samples. Node feature will be zeros.")
                # average_signal_node_features[k] remains zeros

        print(f"  Average Signal Node Features shape: {average_signal_node_features.shape}") # [n_clusters, seq_len]
        print("--- Clustering & Node Feature Calculation Done ---")

        return kmeans_model, all_labels, feature_centroids, average_signal_node_features, n_clusters

    except Exception as e:
        print(f"Error during clustering or node feature calculation: {e}")
        traceback.print_exc()
        return None # Indicate failure


# --- Transition Matrix Function (from graph_2.py, verified OK) ---
def calculate_transition_matrix(all_labels, train_indices, n_clusters):
    """
    Calculates the transition probability matrix based *only* on the sequence of
    cluster labels observed in the training data window sequence.

    Args:
        all_labels (np.ndarray): Cluster labels assigned to *all* windows [N_total].
        train_indices (list): Indices corresponding to the training windows, assumed to be in temporal order.
        n_clusters (int): The total number of clusters (K).

    Returns:
        np.ndarray: Transition probability matrix [n_clusters, n_clusters],
                    where T[i, j] is the probability of transitioning from cluster i to j.
                    Returns None if calculation is not possible.
    """
    print("\n--- Calculating Transition Matrix (Based on Training Window Label Sequence) ---")
    # Extract the sequence of labels corresponding to the training data order
    # IMPORTANT: Assumes train_indices roughly preserves the temporal order of generated windows.
    train_labels_sequence = all_labels[train_indices]
    num_windows = len(train_labels_sequence)

    if num_windows < 2:
        print("  Warning: Need at least 2 training labels to calculate transitions. Returning identity matrix.")
        return np.eye(n_clusters, dtype=float) if n_clusters > 0 else None

    transition_counts = np.zeros((n_clusters, n_clusters), dtype=int)
    valid_transitions = 0
    for i in range(num_windows - 1):
        # Assume train_indices are sorted, so consecutive indices represent consecutive windows
        current_state = train_labels_sequence[i]
        next_state = train_labels_sequence[i+1]

        if 0 <= current_state < n_clusters and 0 <= next_state < n_clusters:
            transition_counts[current_state, next_state] += 1
            valid_transitions += 1
        # else: # Don't warn excessively if labels are just outside range briefly
        #     print(f"  Warning: Invalid state label during transition calc: {current_state} -> {next_state}")

    print(f"  Counted {valid_transitions} transitions.")
    if valid_transitions == 0 and num_windows >= 2:
         print("  Warning: No valid transitions found between states in the training sequence.")
         # Return identity matrix as fallback
         return np.eye(n_clusters, dtype=float)

    # Apply Laplace Smoothing (add-1 smoothing)
    transition_counts += 1

    # Normalize counts to get probabilities
    row_sums = transition_counts.sum(axis=1, keepdims=True)

    # Handle cases where a state might have no outgoing transitions (shouldn't happen with smoothing)
    # Use np.divide for safe division
    transition_probabilities = np.divide(transition_counts, row_sums,
                                         out=np.zeros_like(transition_counts, dtype=float),
                                         where=row_sums!=0)

    # Check for rows that had zero sum *before* smoothing (though smoothing handles division)
    zero_sum_rows = np.where(transition_counts.sum(axis=1) == n_clusters)[0] # rows that are all 1s after smoothing
    if len(zero_sum_rows) > 0:
        # These rows will have prob 1/n_clusters for each transition after smoothing.
        # We might prefer a self-loop probability of 1 if a state truly had no outgoing transitions initially.
        # For simplicity with smoothing, we accept the smoothed probabilities.
        pass # print(f"  Note: Rows {zero_sum_rows} had no outgoing transitions before smoothing.")


    print(f"  Transition Probability Matrix shape: {transition_probabilities.shape}")
    print("--- Transition Matrix Calculation Done ---")
    return transition_probabilities


# --- Visualization Function (Window vs Average Signal Node Feature) ---
def plot_window_assignments(all_signals_padded, all_metadata_with_labels, average_signal_node_features, n_clusters, n_samples_per_cluster=3):
    """
    Plots a few sample time-series windows vs. the average signal embedding
    (node feature) of their assigned cluster.

    Args:
        all_signals_padded (np.ndarray): Padded signal windows [N_total, seq_len].
        all_metadata_with_labels (list): Metadata list updated with 'cluster_label'.
        average_signal_node_features (np.ndarray): Node features (average signals) [n_clusters, seq_len].
        n_clusters (int): The total number of clusters (nodes).
        n_samples_per_cluster (int): Number of sample windows to plot per cluster.
    """
    print(f"\n--- Visualizing Window Assignments vs. Average Signal Node Features ---")
    if n_clusters <= 0 or average_signal_node_features.shape[0] != n_clusters:
        print("  Cannot plot assignments: Invalid number of clusters or node features.")
        return

    seq_len = all_signals_padded.shape[1]
    time_steps = np.arange(seq_len)

    # Determine grid layout
    ncols = math.ceil(math.sqrt(n_clusters))
    nrows = math.ceil(n_clusters / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5), sharex=True, sharey=True)
    axes = np.array(axes).flatten() # Flatten for easy indexing, ensure it's numpy array

    # Determine global y-limits for consistent scaling across plots
    global_min = min(np.min(all_signals_padded), np.min(average_signal_node_features))
    global_max = max(np.max(all_signals_padded), np.max(average_signal_node_features))
    y_padding = (global_max - global_min) * 0.1
    y_min = global_min - y_padding if not np.isnan(global_min) else -1
    y_max = global_max + y_padding if not np.isnan(global_max) else 1

    plotted_clusters_count = 0
    for cluster_label in range(n_clusters):
        ax = axes[cluster_label]
        indices_in_cluster = [i for i, meta in enumerate(all_metadata_with_labels) if meta.get('cluster_label') == cluster_label]
        node_feature_signal = average_signal_node_features[cluster_label] # Average signal for this cluster

        if not indices_in_cluster:
            ax.plot(time_steps, node_feature_signal, color='red', linestyle=':', linewidth=1.5, label=f'Avg Signal Node {cluster_label} (No Samples)')
            ax.set_title(f'Node {cluster_label} (No Samples Assigned)', fontsize=9)
            ax.legend(fontsize=7)
        else:
            plotted_clusters_count += 1
            # Plot the average signal (node feature) first
            ax.plot(time_steps, node_feature_signal, color='red', linestyle='--', linewidth=2.0, label=f'Avg Signal Node {cluster_label}')

            # Sample indices from this cluster to plot individual windows
            n_to_sample = min(n_samples_per_cluster, len(indices_in_cluster))
            sampled_global_indices = random.sample(indices_in_cluster, n_to_sample)

            # Plot sampled windows
            for global_idx in sampled_global_indices:
                signal = all_signals_padded[global_idx, :]
                ax.plot(time_steps, signal, alpha=0.6, linewidth=1.0, label=f'Win {global_idx}') # Blueish default

            ax.set_title(f"Node {cluster_label} ({len(indices_in_cluster)} windows)", fontsize=9)
            ax.legend(fontsize=7)

        ax.grid(True, linestyle=':')
        ax.set_ylim(y_min, y_max)
        # Add labels only to edge plots
        if cluster_label >= n_clusters - ncols : ax.set_xlabel("Time Step")
        if cluster_label % ncols == 0: ax.set_ylabel("Signal Value")
        ax.tick_params(axis='both', which='major', labelsize=8)

    # Hide unused subplots
    for j in range(n_clusters, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Sample Windows vs. Average Signal Node Feature for {plotted_clusters_count}/{n_clusters} Populated Nodes", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    print("--- Window Assignment Visualization Done ---")


# --- Graph Creation/Visualization Function (Adapted from graph_2.py) ---
def create_and_visualize_graph(transition_matrix, average_signal_node_features, n_clusters, visualize=True):
    """
    Creates a NetworkX directed graph from the transition matrix and uses the
    average signal embeddings as node features. Optionally visualizes the graph.

    Args:
        transition_matrix (np.ndarray): Transition probability matrix [n_clusters, n_clusters].
        average_signal_node_features (np.ndarray): Node features (average signals) [n_clusters, seq_len].
        n_clusters (int): The number of nodes (clusters).
        visualize (bool): Whether to plot the graph.

    Returns:
        networkx.DiGraph: The constructed graph with average signals as node features and edge weights.
                          Returns None if graph creation fails.
    """
    print("\n--- Creating Graph Structure ---")
    if n_clusters <= 0: print("  Error: Cannot create graph with 0 nodes."); return None
    if transition_matrix is None or transition_matrix.shape != (n_clusters, n_clusters):
        print(f"  Error: Transition matrix shape incompatible ({transition_matrix.shape if transition_matrix is not None else 'None'}) vs n_clusters ({n_clusters})."); return None
    if average_signal_node_features.shape != (n_clusters, average_signal_node_features.shape[1]): # Check rows match n_clusters
        print(f"  Error: Node features shape {average_signal_node_features.shape} incompatible with n_clusters {n_clusters}."); return None

    G = nx.DiGraph()
    seq_len = average_signal_node_features.shape[1]

    # Add nodes with features (average signal embeddings)
    for node_idx in range(n_clusters):
        features = np.array(average_signal_node_features[node_idx, :], dtype=float)
        # Ensure features are serializable if needed later (e.g., list) - not critical for NetworkX attr
        G.add_node(node_idx, features=features, label=f'Node {node_idx}')

    print(f"  Added {G.number_of_nodes()} nodes. Node labels are indices 0 to {n_clusters-1}.")
    print(f"  Node feature vector length (seq_len): {seq_len}")

    # Add edges with weights (transition probabilities)
    edges_added = 0
    min_weight_threshold = 1e-6 # Avoid adding negligible edges

    for i in range(n_clusters):
        for j in range(n_clusters):
            weight = transition_matrix[i, j]
            if weight > min_weight_threshold:
                # Add edge with weight attribute
                G.add_edge(i, j, weight=float(weight))
                edges_added += 1

    print(f"  Added {edges_added} weighted edges based on training transitions (weight > {min_weight_threshold}).")

    if visualize and G.number_of_nodes() > 0:
        print("  Visualizing graph structure (edges represent transitions)...")
        plt.figure(figsize=(max(8, n_clusters * 1.0), max(7, n_clusters * 0.9))) # Adjusted size
        try:
            # Layout algorithm suitable for directed graphs, adjust params as needed
            pos = nx.spring_layout(G, k=1.8/math.sqrt(n_clusters) if n_clusters > 1 else 1, iterations=70, seed=RANDOM_SEED) if G.number_of_edges() > 0 else nx.circular_layout(G)

            node_draw_labels = {idx: data['label'] for idx, data in G.nodes(data=True)}
            nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue', alpha=0.9)
            nx.draw_networkx_labels(G, pos, labels=node_draw_labels, font_size=9, font_weight='bold')

            if G.number_of_edges() > 0:
                edge_weights_list = [G[u][v]['weight'] for u, v in G.edges()]
                max_w = max(edge_weights_list) if edge_weights_list else 1.0
                max_w = max(max_w, min_weight_threshold) # Avoid division by zero

                # Scale edge width and alpha based on transition probability
                edge_widths = [(w / max_w * 3.0) + 0.5 for w in edge_weights_list] # Adjusted scaling
                edge_alphas = [(w / max_w * 0.6) + 0.25 for w in edge_weights_list] # Adjusted scaling

                nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=edge_alphas, edge_color='gray',
                                    arrowstyle='-|>', arrowsize=15, node_size=700) # Adjusted size

                # Optionally draw edge labels (can get cluttered)
                # edge_weights_dict = {(i, j): f"{G[i][j]['weight']:.2f}" for i, j in G.edges()}
                # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights_dict, label_pos=0.4, font_size=7)

            else:
                print("  No edges to draw.")

            plt.title(f"Graph Structure ({n_clusters} Nodes from Feature Clusters, Edges=Transitions)", fontsize=12)
            plt.axis('off')
            plt.show()
        except Exception as plot_e:
            print(f"Error during graph visualization: {plot_e}") # Catch plotting errors

        print("--- Graph Visualization Done ---")
    elif visualize:
         print("  Cannot visualize graph: 0 nodes.")

    print("--- Graph Creation Done ---")
    return G


# --- Main Orchestration Function ---
def construct_graph_nodes_and_adjacency(
    all_signals_padded, # Should be the potentially scaled signals from dataset.py
    all_metadata,
    train_indices,
    n_clusters_input=DEFAULT_N_CLUSTERS,
    perform_k_analysis=False,
    visualize_graph=True,
    visualize_assignments=True
    ):
    """
    Constructs the graph: extracts features, clusters, calculates node features
    (average signals) and transitions, builds the NetworkX graph object.

    Args:
        all_signals_padded (np.ndarray): Padded signal windows [N_total, seq_len].
                                         Assumed to be the version (scaled/unscaled) the model will use.
        all_metadata (list): Metadata list for all windows.
        train_indices (list): List of global window indices for the training set.
        n_clusters_input (int): The desired number of clusters (K). Can be overridden by analysis.
        perform_k_analysis (bool): If True, run silhouette/elbow analysis and plot.
        visualize_graph (bool): If True, plot the final graph structure.
        visualize_assignments (bool): If True, plot sample windows vs. average signal node feature.

    Returns:
       tuple: (G, feature_centroids, average_signal_node_features, all_metadata_with_labels, feature_scaler, actual_n_clusters)
              - networkx.DiGraph: The constructed graph. Node 'features' are average signals.
              - np.ndarray: Feature centroids array [n_clusters, 15]. For inference assignment.
              - np.ndarray: Average signal node features [n_clusters, seq_len]. For GCN input.
              - list: Metadata list updated with 'cluster_label'.
              - sklearn.preprocessing.StandardScaler: The scaler fitted on training *features*.
              - int: Actual number of clusters used.
              Returns (None, None, None, all_metadata, None, 0) if construction fails.
    """
    print("\n=== Starting Graph Construction Process ===")
    n_total_windows = all_signals_padded.shape[0]
    n_features_expected = 15 # Based on extract_features function

    # 1. Extract Features for ALL windows
    print(f"\n--- Extracting {n_features_expected} Features for {n_total_windows} Windows ---")
    all_feature_vectors_list = [extract_features(window) for window in all_signals_padded]
    # Filter out None results if any window was invalid
    valid_features_mask = [fv is not None for fv in all_feature_vectors_list]
    if not all(valid_features_mask):
        print(f"Warning: {len(valid_features_mask) - sum(valid_features_mask)} windows failed feature extraction.")
        # Decide how to handle - Option 1: error out, Option 2: proceed with valid ones (needs index mapping)
        # Let's error out for simplicity now, as it indicates upstream issues.
        raise ValueError("Feature extraction failed for some windows.")
    all_feature_vectors = np.array(all_feature_vectors_list)

    if all_feature_vectors.shape[1] != n_features_expected:
         raise ValueError(f"Feature extraction resulted in {all_feature_vectors.shape[1]} features, expected {n_features_expected}.")
    print(f"  Shape of extracted features (all): {all_feature_vectors.shape}")

    # 2. Scale Features (Fit ONLY on training features)
    print("\n--- Scaling Features ---")
    if not train_indices:
         print("Error: No training indices provided for scaling features.")
         return None, None, None, all_metadata, None, 0
    train_features = all_feature_vectors[train_indices]
    if train_features.shape[0] == 0:
         print("Error: No training features extracted for scaling.")
         return None, None, None, all_metadata, None, 0

    feature_scaler = StandardScaler()
    try:
        feature_scaler.fit(train_features)
        print(f"  Feature scaler fitted on {len(train_features)} training samples.")
    except Exception as e:
        print(f"Error fitting feature scaler: {e}")
        return None, None, None, all_metadata, None, 0

    scaled_all_features = feature_scaler.transform(all_feature_vectors)
    scaled_all_features = np.nan_to_num(scaled_all_features, nan=0.0, posinf=0.0, neginf=0.0) # Handle potential issues
    print(f"  Shape of scaled features (all): {scaled_all_features.shape}")
    scaled_train_features = scaled_all_features[train_indices]

    # 3. Analyze K for Feature Clustering (Optional)
    chosen_k = n_clusters_input
    if perform_k_analysis:
        suggested_k = analyze_clusters_features(scaled_train_features, max_k=MAX_K_TO_TEST, plot=True)
        if suggested_k is not None and suggested_k > 1:
             print(f"  K analysis suggests K={suggested_k}. Using this value.")
             chosen_k = suggested_k
        else:
             print(f"  K analysis failed or suggested K<=1. Using the provided/default n_clusters_input: {n_clusters_input}")
             chosen_k = n_clusters_input
    else:
         print(f"\n--- Skipping K Analysis for Feature Clustering ---")
         print(f"  Using predefined/default n_clusters: {chosen_k}")

    if chosen_k <= 1:
         print(f"Error: Invalid number of clusters selected: {chosen_k}. Must be > 1.")
         return None, None, None, all_metadata, feature_scaler, 0

    # 4. Cluster Features, Assign Labels, Calculate Centroids and Avg Signal Node Features
    cluster_results = cluster_and_calculate_nodes(
        scaled_all_features, all_signals_padded, train_indices, chosen_k
    )
    if cluster_results is None:
        print("Error: Clustering and node feature calculation failed.")
        return None, None, None, all_metadata, feature_scaler, 0
    kmeans_model, all_labels, feature_centroids, average_signal_node_features, actual_n_clusters = cluster_results
    # Use actual_n_clusters from now on
    chosen_k = actual_n_clusters

    # 5. Add cluster labels to metadata (create a copy to avoid modifying original)
    all_metadata_with_labels = [m.copy() for m in all_metadata]
    if len(all_labels) != len(all_metadata_with_labels):
        print(f"Error: Mismatch between number of labels ({len(all_labels)}) and metadata entries ({len(all_metadata_with_labels)}).")
        return None, None, None, all_metadata, feature_scaler, 0
    for i in range(len(all_metadata_with_labels)):
        all_metadata_with_labels[i]['cluster_label'] = all_labels[i]

    # 6. Visualize window assignments vs average signal node features (Optional)
    if visualize_assignments:
        plot_window_assignments(all_signals_padded, all_metadata_with_labels, average_signal_node_features, chosen_k, N_SAMPLES_PLOT_PER_CLUSTER)

    # 7. Calculate Transition Matrix (Based on Training Labels sequence)
    transition_matrix = calculate_transition_matrix(all_labels, train_indices, chosen_k)
    if transition_matrix is None:
         print(f"Error: Transition matrix calculation failed.")
         return None, None, None, all_metadata_with_labels, feature_scaler, chosen_k # Return metadata with labels assigned so far

    # 8. Create Graph with Average Signals as Node Features
    graph_obj = create_and_visualize_graph(transition_matrix, average_signal_node_features, chosen_k, visualize=visualize_graph)
    if graph_obj is None:
        print("Error: Graph creation failed.")
        return None, feature_centroids, average_signal_node_features, all_metadata_with_labels, feature_scaler, chosen_k

    print("\n=== Graph Construction Process Finished Successfully ===")
    return graph_obj, feature_centroids, average_signal_node_features, all_metadata_with_labels, feature_scaler, chosen_k


# --- Example Usage (Illustrative - requires data from dataset.py run) ---
if __name__ == "__main__":
    print("\n--- Running graph.py directly for testing ---")
    print("  NOTE: This requires output from dataset.py. Using simulated data for structure check.")

    # Simulate inputs from dataset.py
    N_SAMPLES = 500
    SEQ_LEN = 480 # Match dataset.py window size
    N_TRAIN = 360
    simulated_signals = np.random.rand(N_SAMPLES, SEQ_LEN).astype(np.float32) * 100
    simulated_metadata = [{'global_index': i, 'label': 'Healthy'} for i in range(N_SAMPLES)]
    simulated_train_indices = list(range(N_TRAIN))
    # Simulate a dummy signal scaler (not used in graph.py directly, but expected)
    simulated_signal_scaler = StandardScaler()

    try:
        G, feature_centroids, avg_signal_nodes, meta_with_labels, feat_scaler, n_clust = construct_graph_nodes_and_adjacency(
            simulated_signals,
            simulated_metadata,
            simulated_train_indices,
            n_clusters_input=DEFAULT_N_CLUSTERS,
            perform_k_analysis=False,      # Keep analysis off for simple test
            visualize_graph=True,          # Show graph plot for visual check
            visualize_assignments=True,    # Show assignment plot for visual check
        )

        if G is not None:
            print("\n--- Graph Construction Test Summary ---")
            print(f"Graph nodes: {G.number_of_nodes()} (Actual Clusters: {n_clust})")
            print(f"Graph edges: {G.number_of_edges()}")
            if G.number_of_nodes() > 0:
                print(f"Feature Centroids shape: {feature_centroids.shape}")
                print(f"Average Signal Node Features shape: {avg_signal_nodes.shape}")
                # Check feature assignment in graph structure
                print(f"Features shape for node 0 in graph: {G.nodes[0]['features'].shape}") # Should be [SEQ_LEN]
            print(f"Metadata updated with 'cluster_label': {'cluster_label' in meta_with_labels[0]}")
            print(f"Feature Scaler returned: {feat_scaler is not None}")
            if feat_scaler: print(f"  Feature Scaler type: {type(feat_scaler)}")

            # Test saving the outputs (similar to what train.py would do)
            print("\n--- Testing Saving Artifacts (Simulation) ---")
            save_dir = './models_test_graph'
            os.makedirs(save_dir, exist_ok=True)
            graph_save_path = os.path.join(save_dir, 'graph_structure_test.pkl')
            data_to_save = {
                'graph': G,
                'feature_centroids': feature_centroids,
                'average_signal_node_features': avg_signal_nodes,
                'signal_scaler': simulated_signal_scaler, # Include the one from dataset.py
                'feature_scaler': feat_scaler,
                'all_metadata_with_labels': meta_with_labels,
                'n_clusters': n_clust,
                'seq_len': SEQ_LEN
            }
            with open(graph_save_path, 'wb') as f:
                pickle.dump(data_to_save, f)
            print(f"  => Simulated saving graph data to {graph_save_path}")
            # Clean up test directory
            # import shutil
            # shutil.rmtree(save_dir)

            print("\n--- graph.py Test Finished Successfully ---")
        else:
             print("\n--- graph.py Test Failed: Graph construction returned None ---")

    except Exception as e:
        print(f"\nFATAL ERROR during graph.py test: {e}")
        traceback.print_exc()

# --- END OF FILE graph.py ---