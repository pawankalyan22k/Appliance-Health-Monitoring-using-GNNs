# --- START OF FILE anomaly_detection.py ---

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset # Added Subset, ConcatDataset
import numpy as np
import os
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import networkx as nx
from collections import defaultdict
import ast # Added to parse paths string
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler # Needed for scalers
import traceback
import random
from collections import Counter, defaultdict # Make sure Counter is included

# Import custom modules
# dataset module might be needed for PADDING_VALUE or utility functions if refactored
# graph module needed for extract_features
import dataset as dataset_utils
import graph as graph_utils
import model as model_utils # Use a different alias to avoid conflict

print("\n--- Anomaly Detection Script ---")

# --- Configuration ---
class InferConfig:
    # Paths
    MODEL_SAVE_DIR = './models'
    BEST_MODEL_NAME = 'gcn_transformer_best.pt'
    # FINAL_GRAPH_STRUCTURE_NAME corresponds to the output from train.py
    FINAL_GRAPH_STRUCTURE_NAME = 'graph_structure_final.pkl'
    CONFIG_SAVE_NAME = 'train_config.json' # Load the config saved by train.py

    RESULTS_DIR = './anomaly_detection_results'
    REPORT_NAME = 'anomaly_report.txt'

    # Data Parameters (Should align with training, some loaded from config)
    SIGNAL_COLUMN_INDEX = dataset_utils.SIGNAL_COLUMN_INDEX
    WINDOW_SIZE = dataset_utils.WINDOW_SIZE # Use same window size
    STRIDE = dataset_utils.STRIDE         # Use same stride
    PADDING_VALUE = dataset_utils.PADDING_VALUE
    RANDOM_SEED = dataset_utils.RANDOM_SEED # For reproducible sampling if needed

    # --- Data Files for Inference ---
    # <<< EXCLUDES UH_repeated_cycle as requested >>>
    paths_str_inference = """{
        './dishwasher-dataset/11_REFIT_B2_DW_healthy_activations.npz': 'Healthy',
        './dishwasher-dataset/11_REFIT_B2_DW_unhealthy_high_energy_activations.npz': 'UH_high_energy',
        './dishwasher-dataset/11_REFIT_B2_DW_unhealthy_low_extended_energy_activations.npz': 'UH_low_energy',
        './dishwasher-dataset/11_REFIT_B2_DW_unhealthy_noisy_activations.npz': 'UH_noisy'
    }"""
    try:
        NPZ_FILE_PATHS_INFERENCE = ast.literal_eval(paths_str_inference)
        if not isinstance(NPZ_FILE_PATHS_INFERENCE, dict): raise TypeError("Paths not dict.")
    except Exception as e:
        print(f"FATAL: Could not parse NPZ_FILE_PATHS_INFERENCE: {e}")
        exit(1)

    # --- Split Counts Used During HEALTHY-ONLY Training ---
    # Used ONLY to identify the 40 healthy validation windows for thresholding
    SPLIT_COUNTS_VALIDATION_SETUP = {
        'train': {'Healthy': 360}, # Matches dataset.py's split logic
        'val':   {'Healthy': 40},
        'test':  {'Healthy': 0}
    }

    # Thresholding and Evaluation
    THRESHOLD_METHOD = 'QUANTILE' # Options: 'QUANTILE', 'MAX_HEALTHY' ('F1_OPTIMAL' not suitable for healthy-only val set)
    QUANTILE = 0.99 # Quantile for QUANTILE method (e.g., 99th percentile of healthy errors)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 64 # Can be larger for inference

    # Plotting
    SAVE_PLOTS = True
    PLOT_FORMAT = 'png' # e.g., 'png', 'pdf', 'svg'
    N_PLOT_SAMPLES = 6 # Number of examples per plot category (normal, anomaly types)


# --- Helper Functions ---

def load_inference_artifacts(config):
    """Loads the trained model, graph artifacts, training config, and scalers."""
    print("\n--- Loading Inference Artifacts ---")
    model_path = os.path.join(config.MODEL_SAVE_DIR, config.BEST_MODEL_NAME)
    graph_artifact_path = os.path.join(config.MODEL_SAVE_DIR, config.FINAL_GRAPH_STRUCTURE_NAME)
    config_path = os.path.join(config.MODEL_SAVE_DIR, config.CONFIG_SAVE_NAME)

    # Check existence
    if not os.path.exists(model_path): raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not os.path.exists(graph_artifact_path): raise FileNotFoundError(f"Graph artifacts not found: {graph_artifact_path}")
    if not os.path.exists(config_path): raise FileNotFoundError(f"Training config not found: {config_path}")

    # Load training config
    try:
        with open(config_path, 'r') as f:
            train_config_dict = json.load(f)
        print("  Loaded training configuration.")
        # Extract necessary parameters for model instantiation
        required_keys = ['INPUT_DIM', 'SEQ_LEN', 'N_CLUSTERS', 'D_MODEL', 'NHEAD',
                         'NUM_ENCODER_LAYERS', 'NUM_DECODER_LAYERS', 'DIM_FEEDFORWARD',
                         'GCN_OUT_DIM', 'DROPOUT', 'TRANSFORMER_ACTIVATION']
        if not all(k in train_config_dict for k in required_keys):
             raise KeyError(f"Essential keys missing in loaded train_config.json: {required_keys}")
        seq_len_config = train_config_dict['SEQ_LEN'] # Get sequence length from config

    except Exception as e:
        print(f"FATAL ERROR loading/parsing train_config.json: {e}")
        raise e

    # Load graph artifacts (saved by train.py)
    try:
        with open(graph_artifact_path, 'rb') as f:
            inference_artifacts = pickle.load(f)
        # Verify essential keys
        required_artifact_keys = ['graph', 'feature_centroids', 'all_state_embeddings',
                                  'signal_scaler', 'feature_scaler', 'n_clusters', 'seq_len'] # Removed metadata key check, less critical
        if not all(key in inference_artifacts for key in required_artifact_keys):
             raise KeyError(f"Essential keys missing in loaded graph artifacts file: {required_artifact_keys}")

        # Compare sequence lengths consistency
        if inference_artifacts['seq_len'] != seq_len_config:
             print(f"Warning: Seq_len mismatch! Config: {seq_len_config}, Artifacts: {inference_artifacts['seq_len']}. Using config value.")
        config.SEQ_LEN = seq_len_config # Prioritize config

        print("  Loaded inference artifacts (graph, centroids, embeddings, scalers).")
    except Exception as e:
        print(f"FATAL ERROR loading {config.FINAL_GRAPH_STRUCTURE_NAME}: {e}")
        raise e

    # Initialize Model Architecture
    print("  Initializing model architecture...")
    try:
        model_instance = model_utils.GCNTransformerAutoencoder(
            input_dim=train_config_dict['INPUT_DIM'],
            seq_len=config.SEQ_LEN, # Use length from config
            n_clusters=train_config_dict['N_CLUSTERS'],
            d_model=train_config_dict['D_MODEL'],
            nhead=train_config_dict['NHEAD'],
            num_encoder_layers=train_config_dict['NUM_ENCODER_LAYERS'],
            num_decoder_layers=train_config_dict['NUM_DECODER_LAYERS'],
            dim_feedforward=train_config_dict['DIM_FEEDFORWARD'],
            gcn_out_dim=train_config_dict['GCN_OUT_DIM'],
            dropout=train_config_dict['DROPOUT'], # Dropout has no effect in eval mode
            activation=train_config_dict.get('TRANSFORMER_ACTIVATION', 'relu') # Default to relu if missing
        )
        print("  Model architecture initialized.")
    except KeyError as ke:
        print(f"FATAL ERROR: Missing key in loaded train_config.json needed for model init: {ke}")
        raise ke
    except Exception as e:
         print(f"FATAL ERROR during model instantiation: {e}")
         raise e

    # Load Model Weights
    try:
        checkpoint = torch.load(model_path, map_location=torch.device(config.DEVICE))
        # Compatibility check (optional but helpful)
        # E.g., check if checkpoint['config']['N_CLUSTERS'] matches loaded artifacts['n_clusters']
        model_instance.load_state_dict(checkpoint['model_state_dict'])
        model_instance.to(config.DEVICE)
        model_instance.eval() # Set model to evaluation mode
        print(f"  Loaded model weights from epoch {checkpoint.get('epoch','N/A')} "
              f"with best val loss {checkpoint.get('best_val_loss',0.0):.6f}")
    except Exception as e:
         print(f"FATAL ERROR loading model state_dict from {model_path}: {e}")
         raise e

    # Prepare static tensors
    try:
        # GCN state embeddings should already be tensors if saved correctly, else convert
        state_embeddings_np = inference_artifacts['all_state_embeddings']
        all_state_embeddings_tensor = torch.tensor(state_embeddings_np, dtype=torch.float).to(config.DEVICE)
        print(f"  Prepared static state embeddings tensor: {all_state_embeddings_tensor.shape}")
    except Exception as e:
         print(f"FATAL ERROR preparing state embeddings tensor: {e}")
         raise e

    print("--- Loading Complete ---")
    # Return model, config, artifacts dictionary, and state embeddings tensor
    return model_instance, train_config_dict, inference_artifacts, all_state_embeddings_tensor


def create_inference_windows(npz_file_paths, column_index, window_size, stride, padding_value):
    """Loads data from multiple NPZ files, applies sliding window to each series."""
    print("\n--- Creating Inference Windows ---")
    all_windows_list = []
    all_metadata_list = []
    global_window_idx = 0

    for filepath, origin_label in npz_file_paths.items():
        print(f"  Processing file: {filepath} (Label: {origin_label})")
        if not os.path.exists(filepath):
            print(f"    Warning: File not found, skipping: {filepath}")
            continue
        try:
            with np.load(filepath, allow_pickle=True) as data:
                keys = list(data.keys())
                if not keys: print(f"    Info: File empty: {filepath}"); continue

                for key in keys:
                    array = data[key]
                    signal = None
                    if isinstance(array, np.ndarray) and array.ndim == 2 and array.shape[0] > 0 and array.shape[1] > column_index:
                        signal = array[:, column_index].astype(np.float32).copy()
                    elif isinstance(array, np.ndarray) and array.ndim == 1 and array.shape[0] > 0 and column_index == 0:
                        signal = array.astype(np.float32).copy()

                    if signal is not None and signal.size > 0:
                        # Apply sliding window to this individual signal
                        total_length = len(signal)
                        if total_length < window_size:
                             print(f"    Warn: Signal '{key}' length ({total_length}) < window size ({window_size}). Padding.")
                             padding_len = window_size - total_length
                             padded_window = np.pad(signal, (0, padding_len), 'constant', constant_values=padding_value)
                             all_windows_list.append(padded_window)
                             meta = {'source_file': os.path.basename(filepath), 'source_key': key,
                                     'label': origin_label, 'global_index': global_window_idx,
                                     'original_length': total_length}
                             all_metadata_list.append(meta)
                             global_window_idx += 1
                             continue # Move to next key

                        start_indices = range(0, total_length - window_size + 1, stride) # Ensure full windows
                        for i, start_idx in enumerate(start_indices):
                            end_idx = start_idx + window_size
                            window = signal[start_idx:end_idx]
                            all_windows_list.append(window)
                            meta = {'source_file': os.path.basename(filepath), 'source_key': key,
                                    'label': origin_label, 'global_index': global_window_idx,
                                    'original_length': window_size} # Length is now fixed
                            all_metadata_list.append(meta)
                            global_window_idx += 1

                        # Handle potential partial window at the end if stride doesn't align perfectly
                        last_start = (total_length - window_size) // stride * stride # Start of last full window
                        if total_length - last_start > window_size:
                             # Check if there's remaining data after the last full window
                             start_partial = last_start + stride
                             if start_partial < total_length:
                                 # Extract the remainder from the end
                                 # Take the last window_size elements
                                 window = signal[-window_size:]
                                 # Only add if it wasn't already added by the main loop (edge case)
                                 if start_partial+window_size > total_length:
                                     all_windows_list.append(window)
                                     meta = {'source_file': os.path.basename(filepath), 'source_key': key,
                                             'label': origin_label, 'global_index': global_window_idx,
                                             'original_length': window_size}
                                     all_metadata_list.append(meta)
                                     global_window_idx += 1
                    # else: print(f"    Warn: Skipping key '{key}'. Invalid signal data.")

        except Exception as e:
            print(f"  Error processing file {filepath}: {e}")
            traceback.print_exc()
            continue # Skip problematic files

    if not all_windows_list:
        raise ValueError("No valid windows generated from inference files.")

    all_windows_array = np.stack(all_windows_list).astype(np.float32)
    print(f"--- Generated {all_windows_array.shape[0]} inference windows ---")
    return all_windows_array, all_metadata_list


def assign_cluster_labels(windows_array, feature_scaler, feature_centroids):
    """Assigns cluster labels based on nearest feature centroid."""
    print("\n--- Assigning Cluster Labels to Inference Windows ---")
    print("  Extracting features...")
    features_list = [graph_utils.extract_features(window) for window in windows_array]
    valid_mask = [f is not None for f in features_list]
    if not all(valid_mask): raise ValueError("Feature extraction failed for some inference windows.")
    features_array = np.array(features_list)

    print("  Scaling features...")
    if feature_scaler is None: raise ValueError("Feature scaler is required but was not loaded.")
    try:
        scaled_features = feature_scaler.transform(features_array)
        scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
         print(f"Error scaling inference features: {e}")
         raise e

    print("  Finding nearest centroids...")
    if feature_centroids is None: raise ValueError("Feature centroids are required but were not loaded.")
    assigned_labels = np.zeros(len(scaled_features), dtype=int)
    try:
        for i in range(len(scaled_features)):
            distances = np.linalg.norm(feature_centroids - scaled_features[i].reshape(1, -1), axis=1)
            assigned_labels[i] = np.argmin(distances)
    except Exception as e:
         print(f"Error finding nearest centroids: {e}")
         raise e

    print(f"  Assigned labels summary: {dict(Counter(assigned_labels))}")
    print("--- Cluster Assignment Done ---")
    return assigned_labels


# Re-use ApplianceWindowDataset from dataset.py
class InferenceDataset(dataset_utils.ApplianceWindowDataset):
    pass # Inherits functionality, might customize later if needed


# --- Functions for Thresholding and Evaluation (similar to previous attempt) ---
def find_quantile_threshold(errors, quantile):
    """Finds threshold based on quantile of errors (typically from healthy validation)."""
    if len(errors) == 0: raise ValueError("Cannot determine quantile threshold: No errors provided.")
    threshold = np.quantile(errors, quantile)
    print(f"  Threshold ({quantile*100:.1f}th Quantile): {threshold:.6f}")
    return threshold

def find_max_healthy_threshold(errors):
    """Finds threshold based on max error (typically from healthy validation)."""
    if len(errors) == 0: raise ValueError("Cannot determine max threshold: No errors provided.")
    threshold = np.max(errors)
    print(f"  Threshold (Max Healthy Error): {threshold:.6f}")
    return threshold

def evaluate_performance(true_labels, predicted_labels, report_lines):
    """Calculates and prints standard classification metrics."""
    print("\n--- Evaluating Performance ---")
    accuracy = accuracy_score(true_labels, predicted_labels)
    # Use label '1' as positive class (anomaly)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='binary', pos_label=1, zero_division=0
    )
    cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1]) # Ensure order [Normal, Anomaly]

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision (Anomaly): {precision:.4f}")
    print(f"  Recall (Anomaly):    {recall:.4f}")
    print(f"  F1 Score (Anomaly):  {f1:.4f}")
    print(f"  Confusion Matrix (Rows: True, Cols: Pred):\n    [TN, FP]\n    [FN, TP]\n{cm}")

    report_lines.append("\n--- Performance Metrics ---\n")
    report_lines.append(f"Accuracy:  {accuracy:.4f}\n")
    report_lines.append(f"Precision (Anomaly): {precision:.4f}\n")
    report_lines.append(f"Recall (Anomaly):    {recall:.4f}\n")
    report_lines.append(f"F1 Score (Anomaly):  {f1:.4f}\n")
    report_lines.append(f"Confusion Matrix (TN, FP, FN, TP):\n{cm}\n")
    print("--- Evaluation Done ---")
    return accuracy, precision, recall, f1, cm

def calculate_reconstruction_errors(model_ae, dataloader, criterion, device,
                                    all_state_embeddings_tensor, all_metadata_with_labels):
    """Calculates reconstruction error for each sample in the dataloader."""
    model_ae.eval()
    errors = []
    true_labels = []
    global_indices_list = []
    num_batches = len(dataloader)

    print(f"\n--- Calculating Reconstruction Errors ({len(dataloader.dataset)} samples)---")
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (signals, batch_labels, batch_global_indices) in enumerate(dataloader):
            signals = signals.to(device)
            # batch_labels stay on CPU
            # batch_global_indices stay on CPU

            # --- START FIX ---
            # Get the ORIGINAL string labels from metadata using global indices
            try:
                batch_original_labels_str = [all_metadata_with_labels[g_idx]['label']
                                             for g_idx in batch_global_indices.tolist()]
            except Exception as e:
                print(f"\nError looking up original labels in metadata: {e}")
                print(f"  Skipping batch {batch_idx+1}/{num_batches}.")
                continue

            # Define the correct binary mapping
            label_map = {label: (0 if label == 'Healthy' else 1)
                         for label in config.NPZ_FILE_PATHS_INFERENCE.values()} # Use config defined during inference

            # Convert string labels to CORRECT binary labels (0 or 1)
            correct_binary_labels = [label_map.get(s_label, -1) for s_label in batch_original_labels_str]
            if -1 in correct_binary_labels:
                print(f"Warning: Unknown string label encountered in batch {batch_idx+1}.")
                # Handle unknown labels if necessary, e.g., skip or assign default
                # For now, we'll add them to the list and they might cause issues later if not handled

            # --- END FIX ---
            # Get state indices for the batch using lookup
            try:
                state_indices_list = [all_metadata_with_labels[g_idx]['cluster_label'] for g_idx in batch_global_indices.tolist()]
                state_indices = torch.tensor(state_indices_list, dtype=torch.long).to(device)
            except IndexError as e:
                max_meta_idx = len(all_metadata_with_labels) - 1
                problem_indices = [g_idx for g_idx in batch_global_indices.tolist() if g_idx > max_meta_idx]
                print(f"\nError: Global index out of bounds during error calculation.")
                print(f"  Metadata length: {max_meta_idx+1}, Problematic Indices: {problem_indices}")
                print(f"  Skipping batch {batch_idx+1}/{num_batches}.")
                continue
            except KeyError as e:
                print(f"\nError: 'cluster_label' missing during error calculation.")
                print(f"  Skipping batch {batch_idx+1}/{num_batches}.")
                continue
            except Exception as e:
                 print(f"\nUnexpected error getting state indices during error calculation: {e}")
                 traceback.print_exc()
                 print(f"  Skipping batch {batch_idx+1}/{num_batches}.")
                 continue

            # Run model inference
            output = model_ae(signals, state_indices, all_state_embeddings_tensor, tgt=signals)

            # Calculate loss per sample in the batch
            # Assuming criterion is MAE (L1Loss) or MSELoss with reduction='none' or equivalent manual calc
            if isinstance(criterion, (nn.L1Loss, nn.MSELoss)) and criterion.reduction == 'none':
                 sample_errors_tensor = torch.mean(criterion(output, signals), dim=(1, 2)) # Mean over seq_len and feature_dim
            elif isinstance(criterion, nn.L1Loss): # Handle case where reduction wasn't 'none'
                 sample_errors_tensor = torch.mean(torch.abs(output - signals), dim=(1, 2))
            elif isinstance(criterion, nn.MSELoss):
                 sample_errors_tensor = torch.mean((output - signals)**2, dim=(1, 2))
            else:
                 print("Warning: Unsupported criterion reduction. Calculating approximate error.")
                 batch_loss = criterion(output, signals) # Assumes criterion supports reduction='mean' or similar
                 sample_errors_tensor = torch.full((signals.size(0),), batch_loss.item() / signals.size(0), device=device)

            errors.extend(sample_errors_tensor.cpu().numpy())
            true_labels.extend(correct_binary_labels)
            global_indices_list.extend(batch_global_indices.cpu().numpy())

            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed batch {batch_idx+1}/{num_batches}")

    end_time = time.time()
    print(f"--- Error Calculation Finished (Time: {end_time - start_time:.2f}s) ---")
    if len(errors) != len(dataloader.dataset):
         print(f"Warning: Number of calculated errors ({len(errors)}) does not match dataset size ({len(dataloader.dataset)}).")

    return np.array(errors), np.array(true_labels), np.array(global_indices_list)


# --- Plotting Functions (Adapted from previous attempt) ---
def plot_error_distribution(errors, labels, threshold, config, plot_suffix=""):
    plt.figure(figsize=(10, 6))
    sns.histplot(errors[labels == 0], color="blue", label="Healthy (Normal)", kde=True, stat="density", linewidth=0, bins=50)
    sns.histplot(errors[labels == 1], color="red", label="Unhealthy (Anomaly)", kde=True, stat="density", linewidth=0, bins=50)
    plt.axvline(threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
    plt.title(f'Reconstruction Error Distribution {plot_suffix}')
    plt.xlabel('Reconstruction Error (MAE/MSE)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle=':')
    if config.SAVE_PLOTS:
        fname = f'error_distribution{plot_suffix}.{config.PLOT_FORMAT}'
        plt.savefig(os.path.join(config.RESULTS_DIR, fname))
        print(f"Saved plot: {fname}")
    plt.show()

def plot_errors_over_time(errors, labels, threshold, global_indices, config):
    plt.figure(figsize=(18, 7)) # Wider plot
    # Plot all errors
    plt.plot(global_indices, errors, label='Reconstruction Error', color='gray', marker='.', linestyle='-', markersize=2, linewidth=0.5, alpha=0.6)
    plt.axhline(threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold ({threshold:.4f})')

    # Highlight true anomalies
    true_anomaly_indices = global_indices[labels == 1]
    true_anomaly_errors = errors[labels == 1]
    if len(true_anomaly_indices) > 0:
        plt.scatter(true_anomaly_indices, true_anomaly_errors, color='darkorange', marker='o', s=15, label='True Anomaly', alpha=0.8)

    # Highlight detected anomalies (errors > threshold)
    detected_anomaly_mask = errors > threshold
    detected_anomaly_indices = global_indices[detected_anomaly_mask]
    detected_anomaly_errors = errors[detected_anomaly_mask]
    if len(detected_anomaly_indices) > 0:
        plt.scatter(detected_anomaly_indices, detected_anomaly_errors, color='red', marker='x', s=40, label='Detected Anomaly', alpha=0.9)

    plt.title('Reconstruction Error Over Inference Samples')
    plt.xlabel('Global Sample Index')
    plt.ylabel('Reconstruction Error')
    plt.legend(markerscale=1.5)
    plt.grid(True, linestyle=':')
    plt.ylim(bottom=0) # Ensure y-axis starts at 0
    if config.SAVE_PLOTS:
        fname = f'errors_over_time.{config.PLOT_FORMAT}'
        plt.savefig(os.path.join(config.RESULTS_DIR, fname))
        print(f"Saved plot: {fname}")
    plt.show()

def plot_signal_reconstruction(signals_original_all, # Unscaled, all windows
                               signals_reconstructed_all, # Unscaled, test windows only
                               indices_to_plot_global, # Global indices to plot
                               test_global_indices_map, # Map: local test idx -> global idx
                               all_metadata, # Global metadata
                               errors_test, # Errors for test set (local indexing)
                               threshold, config, plot_suffix):
    """Plots original vs. reconstructed signals for selected GLOBAL indices."""
    n_samples = len(indices_to_plot_global)
    if n_samples == 0: print(f"No samples provided to plot for {plot_suffix}."); return

    # Create a reverse map: global index -> local test index
    global_to_local_test_map = {g_idx: l_idx for l_idx, g_idx in enumerate(test_global_indices_map)}

    ncols = 2
    nrows = (n_samples + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 4), sharex=True, squeeze=False) # Ensure axes is 2D
    axes = axes.flatten()

    seq_len = signals_original_all.shape[1]
    time_steps = np.arange(seq_len)

    plotted_count = 0
    for i, global_idx in enumerate(indices_to_plot_global):
        if plotted_count >= len(axes): break # Should not happen if layout is correct

        # Get original signal and metadata
        if global_idx >= len(signals_original_all) or global_idx >= len(all_metadata):
            print(f"Warning: Global index {global_idx} out of bounds for original signals/metadata. Skipping."); continue
        original = signals_original_all[global_idx].flatten()
        metadata_entry = all_metadata[global_idx]
        true_detailed_label = metadata_entry.get('label', 'N/A')
        key = metadata_entry.get('source_key', 'N/A')

        # Find corresponding local test index, reconstructed signal, and error
        local_idx = global_to_local_test_map.get(global_idx)
        if local_idx is None or local_idx >= len(signals_reconstructed_all) or local_idx >= len(errors_test):
            print(f"Warning: Could not find local test data for global index {global_idx}. Skipping reconstruction plot.")
            reconstructed = np.full_like(original, np.nan) # Plot original only
            error = np.nan
            anomaly_detected = False
            legend_label = 'Reconstructed (N/A)'
        else:
            reconstructed = signals_reconstructed_all[local_idx].flatten()
            error = errors_test[local_idx]
            anomaly_detected = error > threshold
            legend_label = f'Reconstructed (Err: {error:.4f})'

        ax = axes[plotted_count]
        ax.plot(time_steps, original, label='Original Signal', color='blue', linewidth=1.0)
        ax.plot(time_steps, reconstructed, label=legend_label, color='red', linestyle='--', linewidth=1.0)
        ax.set_title(f'Window {global_idx} - True: {true_detailed_label} (Key: {key}) - Detected: {"Anomaly" if anomaly_detected else "Normal"}', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=':')
        ax.set_ylabel('Signal Value')
        plotted_count += 1

    if plotted_count > 0:
        for j in range(plotted_count, nrows * ncols): fig.delaxes(axes[j]) # Remove unused subplots
        start_xlabel_row = max(0, plotted_count - ncols) # Add xlabel only on the bottom-most used plots
        for ax_idx in range(start_xlabel_row, plotted_count): axes[ax_idx].set_xlabel('Time Step')

        fig.suptitle(f'Original vs. Reconstructed Signals ({plot_suffix} Examples)', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if config.SAVE_PLOTS:
            fname = f'reconstruction_{plot_suffix}.{config.PLOT_FORMAT}'
            plt.savefig(os.path.join(config.RESULTS_DIR, fname))
            print(f"Saved plot: {fname}")
        plt.show()
    else:
         plt.close(fig) # Close empty figure

def plot_confusion_matrix(cm, class_names, config):
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='Blues')
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if config.SAVE_PLOTS:
        fname = f'confusion_matrix.{config.PLOT_FORMAT}'
        plt.savefig(os.path.join(config.RESULTS_DIR, fname))
        print(f"Saved plot: {fname}")
    plt.show()

def plot_performance_metrics(accuracy, precision, recall, f1, config):
    metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
    plt.figure(figsize=(8, 5))
    colors = ['skyblue', 'lightgreen', 'salmon', 'gold']
    bars = plt.bar(metrics.keys(), metrics.values(), color=colors)
    plt.ylabel('Score')
    plt.title('Anomaly Detection Performance Metrics (Test Set)')
    plt.ylim([0, 1.05]) # Slightly more space at top
    for bar in bars: # Add text labels
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom')
    if config.SAVE_PLOTS:
        fname = f'performance_metrics.{config.PLOT_FORMAT}'
        plt.savefig(os.path.join(config.RESULTS_DIR, fname))
        print(f"Saved plot: {fname}")
    plt.show()


# --- Main Inference Function ---
def main(config):
    """Main function for anomaly detection."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(config.RESULTS_DIR, config.REPORT_NAME)
    report_lines = ["Anomaly Detection Report\n", "="*30 + "\n"]
    start_total_time = time.time()
    random.seed(config.RANDOM_SEED) # Seed for reproducible plotting samples

    # 1. Load Model, Config, Graph Artifacts
    try:
        model_instance, train_config_dict, inference_artifacts, all_state_embeddings_tensor = load_inference_artifacts(config)
        report_lines.append(f"Loaded Model: {config.BEST_MODEL_NAME}\n")
        report_lines.append(f"Loaded Inference Artifacts: {config.FINAL_GRAPH_STRUCTURE_NAME}\n")
        report_lines.append(f"Using device: {config.DEVICE}\n")
        # Extract key components
        signal_scaler = inference_artifacts['signal_scaler']
        feature_scaler = inference_artifacts['feature_scaler']
        feature_centroids = inference_artifacts['feature_centroids']
        n_clusters = inference_artifacts['n_clusters']
        seq_len_loaded = inference_artifacts['seq_len']
        if seq_len_loaded != config.WINDOW_SIZE:
             print(f"Warning: Artifact seq_len {seq_len_loaded} != config window size {config.WINDOW_SIZE}. Check consistency.")
        report_lines.append(f"Model Sequence Length: {seq_len_loaded}\n")
        report_lines.append(f"Number of Clusters (Nodes): {n_clusters}\n")
        report_lines.append(f"Signal Scaler Loaded: {'Yes' if signal_scaler else 'No'}\n")
        report_lines.append(f"Feature Scaler Loaded: {'Yes' if feature_scaler else 'No'}\n")

    except Exception as e:
        print(f"FATAL ERROR loading artifacts: {e}"); traceback.print_exc(); return

    # 2. Load and Prepare Inference Data (All files, windowing)
    try:
        inference_windows_raw, inference_metadata = create_inference_windows(
            config.NPZ_FILE_PATHS_INFERENCE, config.SIGNAL_COLUMN_INDEX,
            config.WINDOW_SIZE, config.STRIDE, config.PADDING_VALUE
        )
        report_lines.append(f"Loaded {len(config.NPZ_FILE_PATHS_INFERENCE)} files for inference.\n")
        report_lines.append(f"Generated {len(inference_metadata)} inference windows.\n")

        # Apply signal scaling (if used during training)
        if signal_scaler:
            print("Applying loaded SIGNAL scaler to inference windows...")
            inference_windows_scaled = dataset_utils.scale_data(inference_windows_raw, signal_scaler, config.PADDING_VALUE)
        else:
            print("Skipping signal scaling (no scaler loaded).")
            inference_windows_scaled = inference_windows_raw

        # Assign cluster labels to inference windows
        assigned_labels = assign_cluster_labels(inference_windows_raw, feature_scaler, feature_centroids) # Use RAW for features, then scale inside
        # Update metadata with assigned labels
        for i, meta in enumerate(inference_metadata): meta['cluster_label'] = assigned_labels[i]

    except Exception as e:
        print(f"FATAL ERROR during inference data preparation: {e}"); traceback.print_exc(); return

    # 3. Identify Healthy Validation Windows (for thresholding)
    # Find the global indices of the 40 healthy windows used for validation during training
    print("\n--- Identifying Healthy Validation Windows for Thresholding ---")
    try:
        # Need to map the SPLIT_COUNTS definition back to the originally generated healthy windows
        # This requires loading the healthy file again and applying the same windowing/splitting logic
        # OR loading the `all_metadata_with_labels` saved during training. Let's assume the latter is in inference_artifacts.
        if 'all_metadata_with_labels' not in inference_artifacts:
             raise ValueError("Training metadata ('all_metadata_with_labels') not found in artifacts, cannot identify validation set.")
        train_meta_full = inference_artifacts['all_metadata_with_labels']
        # Re-run the splitting logic *conceptually* on the indices used during training
        temp_train_indices, temp_val_indices = dataset_utils.split_data_indices(
             train_meta_full, config.SPLIT_COUNTS_VALIDATION_SETUP, config.RANDOM_SEED
        )
        val_indices_global_for_thresh = temp_val_indices # These are the global indices relative to the original healthy window generation
        print(f"Identified {len(val_indices_global_for_thresh)} healthy validation indices for thresholding.")
        if len(val_indices_global_for_thresh) != config.SPLIT_COUNTS_VALIDATION_SETUP['val']['Healthy']:
             print(f"Warning: Identified {len(val_indices_global_for_thresh)} validation indices, but expected {config.SPLIT_COUNTS_VALIDATION_SETUP['val']['Healthy']}.")

    except Exception as e:
        print(f"Error identifying validation set indices: {e}. Cannot determine threshold accurately.")
        # Fallback: Use all healthy inference samples for threshold? Less ideal.
        val_indices_global_for_thresh = [m['global_index'] for m in inference_metadata if m['label'] == 'Healthy']
        if not val_indices_global_for_thresh: print("FATAL: No healthy samples found for fallback thresholding."); return
        print(f"Warning: Using {len(val_indices_global_for_thresh)} healthy inference samples as fallback for thresholding.")

    # 4. Create Datasets and Dataloaders for Inference
    try:
        # Map string labels ('Healthy', 'UH_noisy', etc.) to binary (0: Normal, 1: Anomaly)
        inference_label_map = {label: (0 if label == 'Healthy' else 1) for label in config.NPZ_FILE_PATHS_INFERENCE.values()}

        # Full inference dataset (all windows)
        inference_dataset_full = InferenceDataset(inference_windows_scaled, inference_metadata, list(range(len(inference_metadata))), inference_label_map)
        inference_loader = DataLoader(inference_dataset_full, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
        print(f"Full Inference DataLoader: {len(inference_loader)} batches ({len(inference_dataset_full)} samples)")

        # Validation subset dataset (only the 40 healthy windows)
        # Need to map the global indices (val_indices_global_for_thresh) to the current inference set indices
        # This is tricky because the original indices might not exist if windowing changed subtly.
        # SAFER APPROACH: Use the errors calculated for the *healthy* samples within the main inference run.
        # We will calculate all errors first, then filter.

    except Exception as e:
        print(f"FATAL ERROR creating inference datasets/loaders: {e}"); traceback.print_exc(); return


    # 5. Calculate Reconstruction Errors for ALL Inference Windows
    criterion = nn.L1Loss(reduction='none') # Use reduction='none' for per-sample error calc
    inf_errors, inf_true_labels, inf_global_indices = calculate_reconstruction_errors(
        model_instance, inference_loader, criterion, config.DEVICE,
        all_state_embeddings_tensor, inference_metadata # Pass metadata with assigned cluster labels
    )

    if len(inf_errors) != len(inference_dataset_full):
         print(f"FATAL ERROR: Mismatch in number of errors ({len(inf_errors)}) and inference dataset size ({len(inference_dataset_full)}). Cannot proceed.")
         return

        # 6. Determine Threshold using ALL Healthy Inference Errors
    print("\n--- Determining Anomaly Threshold ---")
    # Use errors from ALL healthy samples identified in the inference run
    healthy_inf_mask = (inf_true_labels == 0)
    healthy_inference_errors = inf_errors[healthy_inf_mask]

    if len(healthy_inference_errors) == 0:
        print("FATAL: No healthy samples found in the inference set to determine threshold.")
        return

    print(f"Using {len(healthy_inference_errors)} errors from ALL healthy inference samples for thresholding.")
    threshold_method_used = config.THRESHOLD_METHOD
    try:
        if config.THRESHOLD_METHOD == 'QUANTILE':
            threshold = find_quantile_threshold(healthy_inference_errors, config.QUANTILE)
        elif config.THRESHOLD_METHOD == 'MAX_HEALTHY':
            threshold = find_max_healthy_threshold(healthy_inference_errors)
        else:
            print(f"Warning: Unsupported threshold method '{config.THRESHOLD_METHOD}'. Defaulting to QUANTILE.")
            threshold_method_used = 'QUANTILE'
            threshold = find_quantile_threshold(healthy_inference_errors, config.QUANTILE)
    except ValueError as e: # Catch potential errors from quantile/max functions
         print(f"Error calculating threshold: {e}. Using median of healthy inference errors as fallback.")
         threshold = np.median(healthy_inference_errors) if len(healthy_inference_errors) > 0 else 0.0
    except Exception as e: # Catch any other unexpected errors
         print(f"Unexpected error during threshold calculation: {e}. Using median fallback.")
         threshold = np.median(healthy_inference_errors) if len(healthy_inference_errors) > 0 else 0.0


    # Update report line to reflect the change
    # Find and replace the old report line generation or add a new one like this:
    idx_to_replace = -1
    for idx, line in enumerate(report_lines):
        if line.startswith("Threshold Method:"):
            idx_to_replace = idx
            break
    new_thresh_line = f"Threshold Method: {threshold_method_used} (Based on All {len(healthy_inference_errors)} Healthy Inference Samples)\n"
    if idx_to_replace != -1:
        report_lines[idx_to_replace] = new_thresh_line
    else:
        report_lines.append(new_thresh_line) # Add if not found

    report_lines.append(f"Anomaly Threshold value: {threshold:.6f}\n") # Keep this line
    print(f"--- Threshold Determined: {threshold:.6f} ---")

    # ... rest of the code (classification, evaluation, plotting) ...

    # 7. Classify Inference Samples and Evaluate
    inf_predictions = (inf_errors >= threshold).astype(int)
    report_lines.append(f"Total Inference Windows: {len(inf_true_labels)}\n")
    report_lines.append(f"Healthy Windows: {np.sum(inf_true_labels == 0)}\n")
    report_lines.append(f"Anomaly Windows: {np.sum(inf_true_labels == 1)}\n")
    report_lines.append(f"Predicted Anomalies: {np.sum(inf_predictions)}/{len(inf_predictions)}\n")

    accuracy, precision, recall, f1, cm = evaluate_performance(inf_true_labels, inf_predictions, report_lines)

    # 8. Generate Plots
    print("\n--- Generating Plots ---")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Plot 1: Error distribution (all inference samples)
    plot_error_distribution(inf_errors, inf_true_labels, threshold, config, plot_suffix="(Inference Set)")

    # Plot 2: Errors over time/index (all inference samples)
    plot_errors_over_time(inf_errors, inf_true_labels, threshold, inf_global_indices, config)

    # Plot 3: Signal Reconstructions
    print("Generating reconstructed signals for plotting...")
    signals_reconstructed_list = []
    with torch.no_grad():
        for signals_scaled, _, batch_global_indices in inference_loader:
            signals_scaled = signals_scaled.to(config.DEVICE)
            try:
                state_indices_list = [inference_metadata[g_idx]['cluster_label'] for g_idx in batch_global_indices.tolist()]
                state_indices = torch.tensor(state_indices_list, dtype=torch.long).to(config.DEVICE)
            except Exception as e:
                print(f"Error getting state indices during plot reconstruction: {e}. Skipping batch.")
                continue
            output_scaled = model_instance(signals_scaled, state_indices, all_state_embeddings_tensor, tgt=signals_scaled)
            signals_reconstructed_list.append(output_scaled.cpu().numpy())

    if not signals_reconstructed_list:
        print("WARNING: No reconstructed signals generated for plotting.")
    else:
        signals_reconstructed_scaled = np.concatenate(signals_reconstructed_list, axis=0)
        # Ensure length matches errors/labels length (handle potential skipped batches)
        min_len = min(len(signals_reconstructed_scaled), len(inf_errors))
        signals_reconstructed_scaled = signals_reconstructed_scaled[:min_len]
        inf_errors_plot = inf_errors[:min_len]
        inf_global_indices_plot = inf_global_indices[:min_len]

        # Inverse transform if scaler exists
        signals_plot_reconstructed_unscaled = np.copy(signals_reconstructed_scaled)
        if signal_scaler:
            print("Applying inverse scaling to reconstructed signals for plotting...")
            for i in range(len(signals_plot_reconstructed_unscaled)):
                 # Simple inverse transform (ignores padding awareness for simplicity in plotting)
                 # Assumes shape [SeqLen, 1] for scaler
                 try:
                     reshaped_sig = signals_plot_reconstructed_unscaled[i].reshape(-1, 1)
                     unscaled_sig = signal_scaler.inverse_transform(reshaped_sig)
                     signals_plot_reconstructed_unscaled[i] = unscaled_sig.reshape(config.WINDOW_SIZE, 1)
                 except Exception as e:
                      print(f"Warning: Error during inverse scaling for plot sample {i}: {e}")
                      signals_plot_reconstructed_unscaled[i] = np.nan # Mark as NaN if inverse fails

        # Generate Specific Reconstruction Plots
        indices_by_label = defaultdict(list)
        for i, meta in enumerate(inference_metadata):
            if i < len(inf_global_indices_plot): # Check if this index was processed
                indices_by_label[meta.get('label', 'Unknown')].append(meta['global_index'])

        plot_categories = { # Map original label to plot suffix
            'Healthy': 'Normal',
            'UH_high_energy': 'HighEnergyAnomaly',
            'UH_low_energy': 'LowEnergyExtendedAnomaly',
            'UH_noisy': 'NoisyAnomaly',
            # UH_repeated_cycle is excluded based on config.NPZ_FILE_PATHS_INFERENCE
        }

        for detailed_label, plot_suffix in plot_categories.items():
            global_indices_category = indices_by_label.get(detailed_label, [])
            if not global_indices_category:
                print(f"Note: No inference samples found with label '{detailed_label}'.")
                continue

            n_available = len(global_indices_category)
            n_to_plot = min(config.N_PLOT_SAMPLES, n_available)
            plot_indices_global_sampled = random.sample(global_indices_category, n_to_plot)
            print(f"Plotting {n_to_plot}/{n_available} '{detailed_label}' examples ({plot_suffix})...")

            plot_signal_reconstruction(
                inference_windows_raw, # Original unscaled windows
                signals_plot_reconstructed_unscaled, # Reconstructed unscaled windows
                plot_indices_global_sampled, # Global indices of samples to plot
                inf_global_indices_plot, # Map from local index (0..N_test-1) to global index
                inference_metadata, # Metadata for all inference windows
                inf_errors_plot, # Errors for all test windows (indexed locally 0..N_test-1)
                threshold, config, plot_suffix
            )

    # Plot 4: Confusion Matrix
    print("Plotting confusion matrix...")
    plot_confusion_matrix(cm, class_names=['Normal (0)', 'Anomaly (1)'], config=config)

    # Plot 5: Performance Metrics Bar Chart
    print("Plotting performance metrics...")
    plot_performance_metrics(accuracy, precision, recall, f1, config=config)

    # --- 9. Save Report ---
    print(f"\n--- Saving Report to {report_path} ---")
    total_time = time.time() - start_total_time
    report_lines.append(f"\nTotal Anomaly Detection Time: {total_time:.2f} seconds\n")
    try:
        with open(report_path, 'w') as f: f.writelines(report_lines)
        print("Report saved.")
    except Exception as e: print(f"Error saving report: {e}")

    print("\n--- Anomaly Detection Script Completed ---")

# --- Run Inference ---
if __name__ == "__main__":
    config = InferConfig()
    try:
        main(config)
    except FileNotFoundError as fnf_error:
        print(f"\nFATAL ERROR: File Not Found. {fnf_error}")
        print(f"Ensure model ('{config.BEST_MODEL_NAME}'), artifacts ('{config.FINAL_GRAPH_STRUCTURE_NAME}'), "
              f"and config ('{config.CONFIG_SAVE_NAME}') exist in '{config.MODEL_SAVE_DIR}'.")
        print(f"Ensure NPZ files defined in InferConfig.NPZ_FILE_PATHS_INFERENCE exist.")
    except Exception as e:
        print(f"\nFATAL ERROR during anomaly detection execution: {e}")
        traceback.print_exc()

# --- END OF FILE anomaly_detection.py ---