# --- START OF FILE dataset.py ---

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import ast
import random
import math
from collections import Counter
import traceback

print("\n--- Anomaly Detection Dataset Setup (Healthy Only Training) ---")

# --- Parameters ---
# <<< IMPORTANT: Define the path to your *single* healthy NPZ file >>>
paths_str = '''{ './dishwasher-dataset/11_REFIT_B2_DW_healthy_activations.npz': 'Healthy' }'''
try:
    NPZ_FILE_PATHS = ast.literal_eval(paths_str)
    if not isinstance(NPZ_FILE_PATHS, dict): raise TypeError("Paths not dict.")
    if len(NPZ_FILE_PATHS) != 1: raise ValueError("NPZ_FILE_PATHS should contain exactly one entry for the healthy data file.")
    HEALTHY_NPZ_PATH = list(NPZ_FILE_PATHS.keys())[0]
except Exception as e:
    print(f"FATAL Error parsing paths: {e}")
    sys.exit(1)

SIGNAL_COLUMN_INDEX = 2 # "Appliance Energy" column index
WINDOW_SIZE = 480       # Desired window length
STRIDE = 240            # Stride for sliding window
PADDING_VALUE = 0.0     # Value for padding shorter sequences/windows
RANDOM_SEED = 42
SCALE_DATA = True       # Whether to apply StandardScaler to signals
BATCH_SIZE = 32         # Batch size for DataLoaders
NUM_WORKERS = 0         # Number of workers for DataLoaders

# Data split counts for the generated *windows* (Healthy Only)
# Ensure total count (train + val) <= total windows generated
SPLIT_COUNTS = {
    'train': {'Healthy': 360},
    'val':   {'Healthy': 40},
    'test':  {'Healthy': 0} # No test set during healthy-only training
}

# Label mapping (Only Healthy class expected during training)
LABEL_MAP = {'Healthy': 0}
print(f"Using Label Map for Healthy-Only Training: {LABEL_MAP}")

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# --- Function Definitions ---

def load_and_create_windows(healthy_filepath, column_index, window_size, stride, padding_value):
    """
    Loads data from a single healthy NPZ file, concatenates series,
    creates sliding windows, and pads the last window if needed.

    Args:
        healthy_filepath (str): Path to the healthy NPZ file.
        column_index (int): Index of the column containing the signal.
        window_size (int): The desired length of each window.
        stride (int): The step size for the sliding window.
        padding_value (float): Value used for padding.

    Returns:
        tuple: (all_windows_array, all_metadata, window_size)
               - np.ndarray: Array of all generated windows [N_windows, window_size].
               - list: Metadata dictionaries for each window.
               - int: The window size used.
    """
    print(f"\n--- Loading & Creating Sliding Windows (Size: {window_size}, Stride: {stride}) ---")
    print(f"  Loading from: {healthy_filepath}")
    all_series_list = []
    original_source_info = [] # Store (key, length) for metadata

    if not os.path.exists(healthy_filepath):
        raise FileNotFoundError(f"Healthy data file not found: {healthy_filepath}")

    try:
        with np.load(healthy_filepath, allow_pickle=True) as data:
            keys = list(data.keys())
            if not keys:
                raise ValueError(f"No arrays found in NPZ file: {healthy_filepath}")

            for key in keys:
                array = data[key]
                if isinstance(array, np.ndarray) and array.ndim == 2 and array.shape[0] > 0 and array.shape[1] > column_index:
                    signal = array[:, column_index].astype(np.float32).copy()
                    if signal.size > 0:
                        all_series_list.append(signal)
                        original_source_info.append({'key': key, 'length': len(signal)})
                # Handle 1D arrays if necessary (as seen in previous attempts)
                elif isinstance(array, np.ndarray) and array.ndim == 1 and array.shape[0] > 0 and column_index == 0:
                     signal = array.astype(np.float32).copy()
                     all_series_list.append(signal)
                     original_source_info.append({'key': key, 'length': len(signal)})
                else:
                     print(f"    Warn: Skipping key '{key}'. Unexpected data format: ndim={array.ndim if isinstance(array, np.ndarray) else 'N/A'}, shape={array.shape if isinstance(array, np.ndarray) else 'N/A'}")

    except Exception as e:
        print(f"  Error processing file {healthy_filepath}: {e}")
        traceback.print_exc()
        raise

    if not all_series_list:
        raise ValueError(f"No valid signals extracted from column {column_index} in {healthy_filepath}")

    # Concatenate all series into one long sequence
    concatenated_series = np.concatenate(all_series_list).astype(np.float32)
    total_length = len(concatenated_series)
    print(f"  Concatenated total length: {total_length} time steps.")
    if total_length < window_size:
         print(f"Warning: Total concatenated length ({total_length}) is less than window size ({window_size}). Only one (padded) window will be created.")

    # Create windows using sliding window approach
    all_windows_list = []
    start_indices = range(0, total_length, stride)
    for i, start_idx in enumerate(start_indices):
        end_idx = start_idx + window_size
        window = concatenated_series[start_idx:end_idx]

        # Pad the last window if it's shorter than window_size
        if len(window) < window_size:
            if end_idx >= total_length: # Only pad if it's truly the end
                padding_len = window_size - len(window)
                padded_window = np.pad(window, (0, padding_len), 'constant', constant_values=padding_value)
                all_windows_list.append(padded_window)
                # print(f"  Padded last window (start: {start_idx}, original end: {len(window)})")
            else:
                 # This case should ideally not happen if stride < window_size unless total_length is very small
                 # It means a window starts but cannot be fully formed even before reaching the actual end
                 print(f"  Skipping incomplete window at start index {start_idx} (length {len(window)} < {window_size}) that is not the final window.")
                 continue # Skip incomplete windows that aren't the last one
        else:
            all_windows_list.append(window)

    if not all_windows_list:
        raise ValueError("No windows could be generated. Check window_size, stride, and data length.")

    # Create metadata for each generated window
    all_metadata = []
    for i in range(len(all_windows_list)):
         metadata = {
             'source_file': os.path.basename(healthy_filepath),
             'original_window_index': i, # Index within the generated windows
             'label': 'Healthy',
             'global_index': i # Use the window index as its global ID for this dataset
         }
         all_metadata.append(metadata)

    all_windows_array = np.stack(all_windows_list).astype(np.float32)
    n_windows_generated = all_windows_array.shape[0]

    print(f"  Generated {n_windows_generated} windows.")
    print(f"  Shape of final windows array: {all_windows_array.shape}") # [N_windows, window_size]
    print("--- Loading & Window Creation Done ---")
    return all_windows_array, all_metadata, window_size


def split_data_indices(all_metadata, split_counts, random_seed=RANDOM_SEED):
    """
    Splits the indices of the generated healthy windows into train and validation sets.

    Args:
        all_metadata (list): List of metadata dictionaries for all generated windows.
        split_counts (dict): Dictionary defining the number of samples for 'train' and 'val'.
        random_seed (int): Seed for random shuffling.

    Returns:
        tuple: (train_indices, val_indices)
               Lists of global window indices for each split.
    """
    print("\n--- Splitting Healthy Window Indices ---")
    random.seed(random_seed) # Ensure reproducibility for splitting

    total_windows = len(all_metadata)
    all_indices = list(range(total_windows))
    random.shuffle(all_indices) # Shuffle indices randomly

    n_train_requested = split_counts['train']['Healthy']
    n_val_requested = split_counts['val']['Healthy']

    if n_train_requested + n_val_requested > total_windows:
        print(f"Warning: Requested {n_train_requested} train + {n_val_requested} val = {n_train_requested + n_val_requested} windows, "
              f"but only {total_windows} were generated. Adjusting counts proportionally.")
        # Simple proportional adjustment (can be refined)
        ratio = total_windows / (n_train_requested + n_val_requested)
        n_train = int(n_train_requested * ratio)
        n_val = total_windows - n_train # Assign remaining to validation
        print(f"  Adjusted counts: Train={n_train}, Val={n_val}")
    else:
        n_train = n_train_requested
        n_val = n_val_requested

    train_indices = sorted(all_indices[:n_train])
    val_indices = sorted(all_indices[n_train : n_train + n_val])

    print(f"Total windows generated: {total_windows}")
    print(f"Train indices assigned: {len(train_indices)}")
    print(f"Val indices assigned:   {len(val_indices)}")

    # Verification: Check for overlap
    assert len(set(train_indices).intersection(set(val_indices))) == 0, "Overlap detected between train and val!"
    print("Verified: No overlap between splits.")
    print("--- Data Splitting Done ---")

    return train_indices, val_indices


class ApplianceWindowDataset(Dataset):
    """PyTorch Dataset for appliance energy signal windows."""
    def __init__(self, signals, metadata, indices, label_map):
        """
        Args:
            signals (np.ndarray): Padded and potentially scaled windows [N_total, seq_len].
            metadata (list): List of metadata dictionaries for all windows.
            indices (list): List of global window indices belonging to this split.
            label_map (dict): Dictionary mapping string labels to numerical labels.
        """
        self.signals = signals[indices] # Select only windows for this split
        self.metadata_split = [metadata[i] for i in indices]
        self.indices = indices # Store the original global window indices
        self.label_map = label_map

        # Add channel dimension: [num_samples_split, seq_len, 1]
        self.signals = np.expand_dims(self.signals, axis=-1)

        print(f"Created Dataset split with {len(self.indices)} samples. Signal shape: {self.signals.shape}")

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Retrieves a single sample (signal window, label, global_window_index) from the dataset.

        Args:
            idx (int): Index of the sample within this split.

        Returns:
            tuple: (signal_tensor, label_tensor, global_index)
                   - torch.Tensor: Signal window data [seq_len, 1].
                   - torch.Tensor: Numerical label (always 0 for healthy training).
                   - int: The original global index of the window.
        """
        signal = self.signals[idx] # Shape [seq_len, 1]
        meta = self.metadata_split[idx]
        original_label = meta['label'] # Should always be 'Healthy' here
        global_index = meta['global_index'] # The index of this window among all generated windows

                # Map original label to numerical label using the provided label_map
        numerical_label = self.label_map.get(original_label, -1) # Default to -1 if label not in map

        if numerical_label == -1:
            # Print a warning if a label from the data isn't in the map provided to the dataset
            print(f"Warning: Label '{original_label}' for global index {global_index} not found in provided LABEL_MAP ({self.label_map}). Assigning label -1.")
            # Depending on downstream use, -1 might need handling, but it's better than forcing to 0.

        # Convert to PyTorch tensors
        signal_tensor = torch.from_numpy(signal).float()
        # Ensure the label tensor uses the correctly mapped numerical label
        label_tensor = torch.tensor(numerical_label, dtype=torch.long)

        return signal_tensor, label_tensor, global_index


def fit_scaler(all_windows_array, train_indices, padding_value=0.0):
    """
    Fits a StandardScaler only on the non-padded parts of the training windows.

    Args:
        all_windows_array (np.ndarray): Array of all generated windows [N_windows, seq_len].
        train_indices (list): List of global window indices for the training set.
        padding_value (float): The value used for padding.

    Returns:
        sklearn.preprocessing.StandardScaler: The fitted scaler object, or None if fitting fails.
    """
    print("\n--- Fitting Scaler on Training Windows (Non-Padded Values) ---")
    if not train_indices:
        print("Warning: No training indices provided. Cannot fit scaler.")
        return None
    train_windows = all_windows_array[train_indices]
    print(f"  Fitting on {len(train_windows)} training windows.")

    # Extract non-padded values from training data windows
    # Flatten the sequences and filter out padding
    non_padded_values = train_windows[train_windows != padding_value].reshape(-1, 1)

    if non_padded_values.size == 0:
        print("Warning: No non-padded values found in training windows to fit scaler. Returning unfitted scaler.")
        # Return unfitted scaler, maybe log a more severe warning depending on expectations
        return StandardScaler()

    scaler = StandardScaler()
    try:
        scaler.fit(non_padded_values)
        print(f"  Scaler fitted with mean: {scaler.mean_[0]:.4f}, scale (std dev): {scaler.scale_[0]:.4f}")
    except Exception as e:
        print(f"Error fitting scaler: {e}")
        return None # Return None if fitting fails
    print("--- Scaler Fitting Done ---")
    return scaler

def scale_data(all_windows_array, scaler, padding_value=0.0):
    """
    Applies a pre-fitted StandardScaler to the windows, handling padding.

    Args:
        all_windows_array (np.ndarray): Windows array [N_windows, seq_len].
        scaler (sklearn.preprocessing.StandardScaler): Fitted scaler object.
        padding_value (float): The value used for padding.

    Returns:
        np.ndarray: Scaled windows array [N_windows, seq_len]. Returns original if scaler is None.
    """
    if scaler is None:
        print("\n--- Scaler is None, skipping scaling ---")
        return all_windows_array

    print("\n--- Scaling Window Data (Handling Padding) ---")
    # Check if scaler is fitted
    if not hasattr(scaler, 'mean_') or not hasattr(scaler, 'scale_'):
         print("Warning: Scaler provided is not fitted. Skipping scaling.")
         return all_windows_array

    scaled_windows = np.copy(all_windows_array) # Avoid modifying original data

    # Iterate through each window to scale only non-padded parts
    num_skipped = 0
    for i in range(scaled_windows.shape[0]):
        window = scaled_windows[i]
        non_padded_mask = (window != padding_value)
        if np.any(non_padded_mask): # Check if there are any non-padded values
            try:
                # Reshape needed for scaler's transform method
                values_to_scale = window[non_padded_mask].reshape(-1, 1)
                scaled_values = scaler.transform(values_to_scale)
                window[non_padded_mask] = scaled_values.flatten()
            except Exception as e:
                 print(f"Warning: Error scaling window {i}. Skipping. Error: {e}")
                 num_skipped += 1
                 # Optionally reset the window to original values if scaling failed partially
                 # scaled_windows[i] = all_windows_array[i]
        # No action needed if window is all padding

    if num_skipped > 0:
        print(f"Warning: Skipped scaling for {num_skipped} windows due to errors.")

    # Handle potential NaNs/Infs introduced by scaling (e.g., if scale_ is near zero)
    # Use padding_value to replace NaNs/Infs
    scaled_windows = np.nan_to_num(scaled_windows, nan=padding_value, posinf=padding_value, neginf=padding_value)
    print("--- Scaling Done ---")
    return scaled_windows


def get_dataloaders(batch_size=BATCH_SIZE, scale=SCALE_DATA, num_workers=NUM_WORKERS):
    """
    Main function to load healthy data, create windows, preprocess, split,
    and create DataLoaders for healthy-only training.

    Args:
        batch_size (int): Batch size for the DataLoaders.
        scale (bool): Whether to apply StandardScaler to the signal windows.
        num_workers (int): Number of workers for DataLoader.

    Returns:
        tuple: Contains:
            - train_loader (DataLoader): DataLoader for training windows.
            - val_loader (DataLoader): DataLoader for validation windows.
            - all_signals_padded (np.ndarray): All generated windows [N_windows, seq_len], potentially scaled.
            - all_metadata (list): Metadata for all generated windows.
            - train_indices (list): Global window indices for training split.
            - val_indices (list): Global window indices for validation split.
            - max_len (int): Window size (e.g., 480).
            - signal_scaler (StandardScaler or None): Fitted scaler object if scale=True, else None.
    """
    # 1. Load Data and Create Windows
    all_signals_padded, all_metadata, max_len = load_and_create_windows(
        HEALTHY_NPZ_PATH, SIGNAL_COLUMN_INDEX, WINDOW_SIZE, STRIDE, PADDING_VALUE
    )

    # 2. Split Indices (Train/Val for Healthy Windows)
    train_indices, val_indices = split_data_indices(
        all_metadata, SPLIT_COUNTS, RANDOM_SEED
    )

    # 3. Scaling (Optional)
    signal_scaler = None
    if scale:
        signal_scaler = fit_scaler(all_signals_padded, train_indices, PADDING_VALUE)
        # Scale the *original* padded signals using the fitted scaler
        signals_to_use = scale_data(all_signals_padded, signal_scaler, PADDING_VALUE)
    else:
        signals_to_use = all_signals_padded # Use the original padded signals
        print("\n--- Skipping Signal Scaling ---")


    # 4. Create Datasets
    print("\n--- Creating Datasets ---")
    train_dataset = ApplianceWindowDataset(signals_to_use, all_metadata, train_indices, LABEL_MAP)
    val_dataset = ApplianceWindowDataset(signals_to_use, all_metadata, val_indices, LABEL_MAP)
    # test_dataset creation is skipped as test_indices are empty during healthy-only training
    print("--- Datasets Created ---")

    # 5. Create DataLoaders
    print("\n--- Creating DataLoaders ---")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    # test_loader is None or empty
    print(f"Train loader: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"Val loader:   {len(val_loader)} batches ({len(val_dataset)} samples)")
    print("--- DataLoaders Created ---")

    # Return necessary components for subsequent phases
    return (train_loader, val_loader,
            signals_to_use, # Use the potentially scaled signals
            all_metadata, train_indices, val_indices,
            max_len, signal_scaler)

# --- Example Usage (for testing this script directly) ---
if __name__ == "__main__":
    print("\n--- Running dataset.py directly for testing ---")
    try:
        (train_loader, val_loader,
         all_windows, metadata, train_idx, val_idx,
         win_size, scaler) = get_dataloaders(batch_size=BATCH_SIZE, scale=SCALE_DATA)

        print(f"\n--- Example Batch from Train Loader ---")
        for signals, labels, global_indices in train_loader:
            print(f"Signals batch shape: {signals.shape}") # Should be [batch_size, WINDOW_SIZE, 1]
            print(f"Labels batch shape: {labels.shape}")   # Should be [batch_size]
            print(f"Labels batch example: {labels[:5]}") # Should be all 0s
            print(f"Global indices batch shape: {global_indices.shape}")
            print(f"Global indices example: {global_indices[:5]}")
            break # Only show one batch

        print(f"\nTotal generated windows: {len(metadata)}")
        print(f"Shape of all windows array: {all_windows.shape}")
        print(f"Window size (max_len): {win_size}")
        print(f"Number of training indices: {len(train_idx)}")
        print(f"Number of validation indices: {len(val_idx)}")
        print(f"Signal Scaler object: {'Fitted' if scaler and hasattr(scaler, 'mean_') else 'None or Unfitted'}")
        print("\n--- dataset.py Test Finished Successfully ---")

    except FileNotFoundError as fnf_error:
        print(f"\nFATAL ERROR: File Not Found. {fnf_error}")
        print(f"Please ensure the healthy NPZ file specified in NPZ_FILE_PATHS exists: '{HEALTHY_NPZ_PATH}'")
    except ValueError as val_error:
        print(f"\nFATAL ERROR: Value Error. {val_error}")
        traceback.print_exc()
    except Exception as e:
        print(f"\nFATAL ERROR: An unexpected error occurred.")
        traceback.print_exc()

# --- END OF FILE dataset.py ---