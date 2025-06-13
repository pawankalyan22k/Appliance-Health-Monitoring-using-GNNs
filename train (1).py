# --- START OF FILE train.py ---

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import json
import pickle
import traceback
import networkx as nx
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F # Make sure F is imported

# Import custom modules
import dataset # To get dataloaders and parameters
import graph   # To get graph construction parameters/functions
import model   # To get model definitions (GCNTransformerAutoencoder)
import utils   # Potentially needed

print("\n--- GCN-Transformer Training Script (Using Conditional Autoencoder Model) ---")

# --- Configuration ---
class TrainConfig:
    # Data Parameters
    SIGNAL_COLUMN_INDEX = dataset.SIGNAL_COLUMN_INDEX
    BATCH_SIZE = dataset.BATCH_SIZE
    SCALE_DATA = dataset.SCALE_DATA
    NUM_WORKERS = dataset.NUM_WORKERS
    SEQ_LEN = dataset.WINDOW_SIZE # 480

    # Graph Construction Parameters
    GRAPH_ARTIFACTS_PATH = './models/graph_initial.pkl' # Path for initial graph data attempt
    CREATE_GRAPH_IF_MISSING = True
    DEFAULT_N_CLUSTERS = graph.DEFAULT_N_CLUSTERS # 9
    # NODE_FEATURE_DIM needed for graph.py if run inside, not directly for model
    VISUALIZE_GRAPH = False
    VISUALIZE_ASSIGNMENTS = False

    # *** Model Parameters (Matching GCNTransformerAutoencoder) ***
    # Input dim is implicitly 1 for the signal
    # N_CLUSTERS is derived from graph artifacts
    D_MODEL = 256           # Tuned Transformer embedding dimension
    NHEAD = 8               # Tuned Attention heads
    NUM_ENCODER_LAYERS = 2  # Tuned Transformer encoder layers
    NUM_DECODER_LAYERS = 2  # Tuned Transformer decoder layers
    DIM_FEEDFORWARD = 1024  # Tuned Feedforward dim (4 * D_MODEL)
    GCN_OUT_DIM = 64        # Tuned Dimension of GCN state embeddings
    DROPOUT = 0.1
    TRANSFORMER_ACTIVATION = 'gelu' # Tuned activation
    USE_DECONVOLUTION = False # <<< Set False to use the successful Linear output layer >>>

    # GCN Encoder Params (for offline step)
    GCN_HIDDEN_DIM = 64 # Tuned hidden dim for GCNEncoder
    GCN_LAYERS = 8      # Layers for GCNEncoder

    # Training Parameters
    LEARNING_RATE = 0.0001 # Tuned LR
    EPOCHS = 100           # Increased epochs
    PATIENCE = 20          # Increased patience
    OPTIMIZER = 'AdamW'
    LOSS_FN = 'MAE'        # MAE
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CLIP_GRAD_NORM = 0.7   # Tuned clipping

    # Paths
    MODEL_SAVE_DIR = './models'
    BEST_MODEL_NAME = 'gcn_transformer_best.pt' # Standard name for this model
    FINAL_MODEL_NAME = 'gcn_transformer_final.pt' # Standard name
    FINAL_GRAPH_STRUCTURE_NAME = 'graph_structure_final.pkl' # Output artifacts
    CONFIG_SAVE_NAME = 'train_config.json' # Standard name


# --- Helper Functions (save_checkpoint, save_inference_artifacts, save_config_dict - unchanged) ---
def save_checkpoint(state, is_best, save_dir, best_filename, final_filename):
    """Saves model and training parameters."""
    os.makedirs(save_dir, exist_ok=True)
    final_path = os.path.join(save_dir, final_filename)
    torch.save(state, final_path)
    if is_best:
        best_path = os.path.join(save_dir, best_filename)
        torch.save(state, best_path)
        print(f"  => Saved new best model to {best_path}")

def save_inference_artifacts(artifacts, save_dir, filename):
    """Saves artifacts needed for inference."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(artifacts, f)
        print(f"  => Saved inference artifacts to {filepath}")
    except Exception as e:
         print(f"Error saving inference artifacts to {filepath}: {e}")

def save_config_dict(config_dict, save_dir, filename):
    """Saves the configuration dictionary."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    try:
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)
        print(f"  => Saved final configuration to {filepath}")
    except Exception as e:
         print(f"Error saving config dictionary to {filepath}: {e}")


# --- Training and Evaluation Epochs for GCNTransformerAutoencoder ---

def train_epoch(model_ae, dataloader, optimizer, criterion, device,
                all_state_embeddings, all_metadata_with_labels, clip_norm):
    """Runs one training epoch for the GCNTransformerAutoencoder."""
    model_ae.train()
    total_loss = 0.0
    batch_count = 0
    num_batches = len(dataloader)

    for batch_idx, (signals, _, global_indices) in enumerate(dataloader):
        signals = signals.to(device) # Input AND Target Y_scaled
        # global_indices remain on CPU for lookup

        # --- Get state indices for the batch ---
        try:
             state_indices_list = [all_metadata_with_labels[g_idx]['cluster_label'] for g_idx in global_indices.tolist()]
             state_indices = torch.tensor(state_indices_list, dtype=torch.long).to(device)
        except IndexError as e:
             max_meta_idx = len(all_metadata_with_labels) - 1
             problem_indices = [g_idx for g_idx in global_indices.tolist() if g_idx > max_meta_idx]
             print(f"\nError: Global index out of bounds during training lookup. Max Meta Idx: {max_meta_idx}, Problem Indices: {problem_indices}")
             print(f"Skipping batch {batch_idx+1}/{num_batches}.")
             continue
        except KeyError as e:
             print(f"\nError: 'cluster_label' key not found during training lookup.")
             print(f"Skipping batch {batch_idx+1}/{num_batches}.")
             continue
        except Exception as e:
             print(f"\nUnexpected error getting state indices: {e}")
             traceback.print_exc(); print(f"Skipping batch {batch_idx+1}/{num_batches}."); continue

        optimizer.zero_grad()

        # *** FORWARD PASS: Input signal, state indices, ALL static embeddings, signal is target ***
        output = model_ae(src=signals,
                          state_indices=state_indices,
                          all_state_embeddings=all_state_embeddings,
                          tgt=signals) # Y_hat_scaled

        # *** LOSS CALCULATION: Compare reconstructed output with original signal ***
        loss = criterion(output, signals) # Compare Y_hat_scaled with Y_scaled

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss in batch {batch_idx+1}. Skipping update.")
            continue

        loss.backward()

        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model_ae.parameters(), clip_norm)

        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

        if (batch_idx + 1) % 20 == 0: # Print progress less frequently
            print(f'  Batch {batch_idx+1}/{num_batches} | Loss: {loss.item():.6f}')

    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    return avg_loss


def evaluate_epoch(model_ae, dataloader, criterion, device,
                   all_state_embeddings, all_metadata_with_labels):
    """Runs one evaluation epoch for the GCNTransformerAutoencoder."""
    model_ae.eval()
    total_loss = 0.0
    batch_count = 0
    num_batches = len(dataloader)

    with torch.no_grad():
        for batch_idx, (signals, _, global_indices) in enumerate(dataloader):
            signals = signals.to(device) # Input AND Target Y_scaled

            # --- Get state indices ---
            try:
                 state_indices_list = [all_metadata_with_labels[g_idx]['cluster_label'] for g_idx in global_indices.tolist()]
                 state_indices = torch.tensor(state_indices_list, dtype=torch.long).to(device)
            except IndexError as e:
                 max_meta_idx = len(all_metadata_with_labels) - 1
                 problem_indices = [g_idx for g_idx in global_indices.tolist() if g_idx > max_meta_idx]
                 print(f"Error: Global index out of bounds during validation lookup. Max Meta Idx: {max_meta_idx}, Problem Indices: {problem_indices}")
                 print(f"Skipping validation batch {batch_idx+1}/{num_batches}.")
                 continue
            except KeyError as e:
                 print(f"Error: 'cluster_label' key not found during validation lookup.")
                 print(f"Skipping validation batch {batch_idx+1}/{num_batches}.")
                 continue
            except Exception as e:
                 print(f"Unexpected error getting state indices in validation: {e}")
                 traceback.print_exc(); print(f"Skipping validation batch {batch_idx+1}/{num_batches}."); continue

            # *** FORWARD PASS ***
            output = model_ae(src=signals,
                              state_indices=state_indices,
                              all_state_embeddings=all_state_embeddings,
                              tgt=signals) # Y_hat_scaled

            # *** LOSS CALCULATION ***
            loss = criterion(output, signals) # Compare Y_hat_scaled with Y_scaled

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                batch_count += 1
            else:
                 print(f"Warning: NaN/Inf loss during validation batch {batch_idx+1}. Ignoring batch loss.")

    avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
    return avg_loss


# --- Main Training Function ---
def main(config):
    """Main function using GCNTransformerAutoencoder."""
    print(f"Using device: {config.DEVICE}")
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

    # --- 1. Load Data ---
    # (Same as before)
    print("\n--- Loading and Preprocessing Data ---")
    try:
        (train_loader, val_loader,
         all_signals_padded_processed, # Potentially scaled signals
         all_metadata_initial, train_indices, val_indices,
         seq_len, signal_scaler) = dataset.get_dataloaders(
             batch_size=config.BATCH_SIZE,
             scale=config.SCALE_DATA,
             num_workers=config.NUM_WORKERS
         )
        if seq_len != config.SEQ_LEN: raise ValueError("Seq len mismatch")
        print(f"Data loaded. Sequence length: {config.SEQ_LEN}")
    except Exception as e: print(f"FATAL ERROR during data loading: {e}"); traceback.print_exc(); return

    # --- 2. Load or Construct Graph Artifacts ---
    # (Same as before)
    print("\n--- Loading or Constructing Graph Artifacts ---")
    # ... (Paste the full loading/construction logic here again) ...
    # --- Snipped for brevity ---
    graph_artifacts = None
    if os.path.exists(config.GRAPH_ARTIFACTS_PATH):
        print(f"Attempting to load graph artifacts from: {config.GRAPH_ARTIFACTS_PATH}")
        try:
            with open(config.GRAPH_ARTIFACTS_PATH, 'rb') as f: graph_artifacts = pickle.load(f)
            # Basic validation
            if not all(k in graph_artifacts for k in ['n_clusters', 'seq_len']): raise KeyError("Missing keys")
            config.N_CLUSTERS = graph_artifacts['n_clusters']
            if graph_artifacts['seq_len'] != config.SEQ_LEN: print("Warning: Seq len mismatch in loaded artifact"); graph_artifacts['seq_len'] = config.SEQ_LEN
            print("Graph artifacts loaded.")
        except Exception as e: print(f"Error loading artifacts: {e}. Reconstructing..."); graph_artifacts = None

    if graph_artifacts is None and config.CREATE_GRAPH_IF_MISSING:
        print("Constructing graph artifacts...")
        try:
             signals_for_graph = all_signals_padded_processed
             graph_results = graph.construct_graph_nodes_and_adjacency(
                 signals_for_graph, all_metadata_initial.copy(), train_indices,
                 n_clusters_input=config.DEFAULT_N_CLUSTERS, perform_k_analysis=False,
                 visualize_graph=config.VISUALIZE_GRAPH, visualize_assignments=config.VISUALIZE_ASSIGNMENTS
             )
             G, feat_centroids, avg_signal_nodes, meta_with_labels, feat_scaler, n_clust = graph_results
             if G is None or n_clust <= 0: raise ValueError("Graph construction failed.")
             graph_artifacts = {'graph': G, 'feature_centroids': feat_centroids, 'average_signal_node_features': avg_signal_nodes,
                                'all_metadata_with_labels': meta_with_labels, 'feature_scaler': feat_scaler,
                                'n_clusters': n_clust, 'seq_len': config.SEQ_LEN}
             config.N_CLUSTERS = n_clust
             print(f"Graph artifacts constructed. N_Clusters: {config.N_CLUSTERS}")
        except Exception as e: print(f"FATAL ERROR during graph construction: {e}"); traceback.print_exc(); return
    elif graph_artifacts is None: print(f"FATAL ERROR: Graph artifacts not found and cannot create."); return

    try: # Extract components
        G = graph_artifacts['graph']
        feature_centroids = graph_artifacts['feature_centroids']
        average_signal_node_features = graph_artifacts['average_signal_node_features']
        all_metadata_with_labels = graph_artifacts['all_metadata_with_labels']
        feature_scaler = graph_artifacts['feature_scaler']
        config.N_CLUSTERS = graph_artifacts['n_clusters']
    except KeyError as e: print(f"FATAL ERROR: Missing key {e} in graph_artifacts."); return
    # --- End Graph Handling ---

    # --- 3. Generate Static GCN Embeddings ---
    # (Same as before - GCNEncoder part)
    print("\n--- Generating Static GCN State Embeddings ---")
    # ... (Paste the full GCN embedding generation logic here again) ...
     # --- Snipped for brevity ---
    try:
        if model.GCNConv is None: raise ImportError("PyTorch Geometric not available.")
        gcn_encoder = model.GCNEncoder(
            node_feature_dim=config.SEQ_LEN, hidden_dim=config.GCN_HIDDEN_DIM,
            output_dim=config.GCN_OUT_DIM, num_layers=config.GCN_LAYERS,
            dropout=config.DROPOUT, activation=F.gelu
        ).to(config.DEVICE)
        gcn_encoder.eval()
        gcn_node_features_tensor = torch.tensor(average_signal_node_features, dtype=torch.float).to(config.DEVICE)
        adj_matrix = nx.to_numpy_array(G, weight='weight')
        adj_matrix_tensor = torch.tensor(adj_matrix, dtype=torch.float).to(config.DEVICE)
        edge_index, edge_weight = model.dense_to_sparse(adj_matrix_tensor)
        if edge_weight is not None: edge_weight = edge_weight.float()
        print(f"Input node features shape to GCNEncoder: {gcn_node_features_tensor.shape}")
        with torch.no_grad():
            all_state_embeddings = gcn_encoder(gcn_node_features_tensor, edge_index, edge_weight)
        print(f"Generated static GCN state embeddings shape: {all_state_embeddings.shape}")
        if all_state_embeddings.shape != (config.N_CLUSTERS, config.GCN_OUT_DIM): raise ValueError("GCN embeddings shape mismatch.")
        all_state_embeddings = all_state_embeddings.detach()
    except Exception as e: print(f"FATAL ERROR during GCN embedding generation: {e}"); traceback.print_exc(); return
    # --- End GCN ---

    # --- 4. Initialize GCNTransformerAutoencoder Model ---
    print("\n--- Initializing GCNTransformerAutoencoder Model ---")
    try:
        # *** Instantiate the correct model ***
        model_instance = model.GCNTransformerAutoencoder(
            input_dim=1, # Hardcode 1 for signal input dim
            seq_len=config.SEQ_LEN,
            n_clusters=config.N_CLUSTERS,
            d_model=config.D_MODEL,
            nhead=config.NHEAD,
            num_encoder_layers=config.NUM_ENCODER_LAYERS,
            num_decoder_layers=config.NUM_DECODER_LAYERS,
            dim_feedforward=config.DIM_FEEDFORWARD,
            gcn_out_dim=config.GCN_OUT_DIM,
            dropout=config.DROPOUT,
            activation=config.TRANSFORMER_ACTIVATION,
            use_deconvolution=config.USE_DECONVOLUTION # Control deconv layer
        ).to(config.DEVICE)

        num_params = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
        print(f"Model initialized. Trainable parameters: {num_params:,}")

    except Exception as e:
        print(f"FATAL ERROR during model initialization: {e}"); traceback.print_exc(); return

    # --- 5. Define Loss and Optimizer ---
    # (Same as before)
    if config.LOSS_FN == 'MAE': criterion = nn.L1Loss()
    elif config.LOSS_FN == 'MSE': criterion = nn.MSELoss()
    else: raise ValueError(f"Unsupported loss function: {config.LOSS_FN}")

    if config.OPTIMIZER == 'Adam': optimizer = optim.Adam(model_instance.parameters(), lr=config.LEARNING_RATE)
    elif config.OPTIMIZER == 'AdamW': optimizer = optim.AdamW(model_instance.parameters(), lr=config.LEARNING_RATE)
    else: raise ValueError(f"Unsupported optimizer: {config.OPTIMIZER}")
    print(f"Using Loss: {config.LOSS_FN}, Optimizer: {config.OPTIMIZER}, LR: {config.LEARNING_RATE}")

    # --- 6. Training Loop ---
    # (Same structure, uses the correct train/eval functions for this model)
    print("\n--- Starting Training ---")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    training_start_time = time.time()
    train_losses, val_losses = [], []

    for epoch in range(1, config.EPOCHS + 1):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch}/{config.EPOCHS}")

        train_loss = train_epoch(model_instance, train_loader, optimizer, criterion, config.DEVICE,
                                 all_state_embeddings, all_metadata_with_labels, config.CLIP_GRAD_NORM)
        train_losses.append(train_loss)

        val_loss = evaluate_epoch(model_instance, val_loader, criterion, config.DEVICE,
                                  all_state_embeddings, all_metadata_with_labels)
        val_losses.append(val_loss)

        epoch_duration = time.time() - epoch_start_time
        print("-" * 89)
        print(f"| End of Epoch {epoch:3d} | Time: {epoch_duration:5.2f}s | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} |")
        print("-" * 89)

        # Checkpointing and Early Stopping (Same logic)
        if np.isnan(val_loss) or np.isinf(val_loss):
             print("Warning: Invalid validation loss. Skipping checks.")
             epochs_no_improve += 1
        elif val_loss < best_val_loss:
            print(f"Validation loss improved ({best_val_loss:.6f} --> {val_loss:.6f}). Saving best model...")
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint({
                'epoch': epoch, 'model_state_dict': model_instance.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss,
            }, True, config.MODEL_SAVE_DIR, config.BEST_MODEL_NAME, config.FINAL_MODEL_NAME)
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve. Early stopping counter: {epochs_no_improve}/{config.PATIENCE}")

        if epochs_no_improve >= config.PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break
    # --- End Training Loop ---

    training_duration = time.time() - training_start_time
    print(f"\n--- Training Finished ---")
    print(f"Total Training Time: {training_duration/60:.2f} minutes")
    print(f"Best Validation Loss: {best_val_loss:.6f}")

    # --- 7. Save Final Inference Artifacts ---
    # (Same saving logic)
    print("\n--- Saving Final Artifacts for Inference ---")
    final_artifacts = {
        'graph': G, 'feature_centroids': feature_centroids,
        'all_state_embeddings': all_state_embeddings.cpu().numpy(),
        'signal_scaler': signal_scaler, 'feature_scaler': feature_scaler,
        'all_metadata_with_labels': all_metadata_with_labels,
        'n_clusters': config.N_CLUSTERS, 'seq_len': config.SEQ_LEN
    }
    save_inference_artifacts(final_artifacts, config.MODEL_SAVE_DIR, config.FINAL_GRAPH_STRUCTURE_NAME)

    # --- 8. Save FINAL Config Dictionary ---
    # (Save relevant config for the Autoencoder)
    print("\n--- Saving Final Configuration Dictionary ---")
    config_dict_to_save = {
        'SIGNAL_COLUMN_INDEX': config.SIGNAL_COLUMN_INDEX, 'BATCH_SIZE': config.BATCH_SIZE,
        'SCALE_DATA': config.SCALE_DATA, 'WINDOW_SIZE': config.SEQ_LEN,
        'INPUT_DIM': 1, 'SEQ_LEN': config.SEQ_LEN, 'N_CLUSTERS': config.N_CLUSTERS,
        'D_MODEL': config.D_MODEL, 'NHEAD': config.NHEAD,
        'NUM_ENCODER_LAYERS': config.NUM_ENCODER_LAYERS, 'NUM_DECODER_LAYERS': config.NUM_DECODER_LAYERS,
        'DIM_FEEDFORWARD': config.DIM_FEEDFORWARD, 'GCN_OUT_DIM': config.GCN_OUT_DIM,
        'DROPOUT': config.DROPOUT, 'TRANSFORMER_ACTIVATION': config.TRANSFORMER_ACTIVATION,
        'USE_DECONVOLUTION': config.USE_DECONVOLUTION, # Record if deconv was used
        # Include GCN params for context
        'GCN_HIDDEN_DIM': config.GCN_HIDDEN_DIM, 'GCN_LAYERS': config.GCN_LAYERS,
        # Include Training params
        'LEARNING_RATE': config.LEARNING_RATE, 'EPOCHS_COMPLETED': epoch,
        'BEST_VAL_LOSS': best_val_loss, 'OPTIMIZER': config.OPTIMIZER, 'LOSS_FN': config.LOSS_FN,
        'CLIP_GRAD_NORM': config.CLIP_GRAD_NORM, 'DEVICE_USED': config.DEVICE,
    }
    # Add deconv params if used
    if config.USE_DECONVOLUTION:
         config_dict_to_save['DECONV_INTERMEDIATE_CHANNELS'] = config.DECONV_INTERMEDIATE_CHANNELS
         config_dict_to_save['DECONV_KERNEL_SIZE'] = config.DECONV_KERNEL_SIZE
         config_dict_to_save['DECONV_STRIDE'] = config.DECONV_STRIDE

    save_config_dict(config_dict_to_save, config.MODEL_SAVE_DIR, config.CONFIG_SAVE_NAME)

    print("\n--- Training Script Completed ---")

# --- Run Training ---
if __name__ == "__main__":
    config = TrainConfig()
    try:
        main(config)
    except ImportError as e: print(f"\nFATAL ERROR: Missing required library. {e}")
    except FileNotFoundError as e: print(f"\nFATAL ERROR: File not found. {e}")
    except Exception as e: print(f"\nFATAL ERROR during main training execution: {e}"); traceback.print_exc()