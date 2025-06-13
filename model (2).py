# --- START OF FILE model.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F
# Ensure utils.py is in the same directory or accessible via PYTHONPATH
try:
    from utils import PositionalEncoding
except ImportError:
    print("Error: utils.py not found. Please ensure it's in the same directory.")
    # Define a dummy PositionalEncoding if utils cannot be imported,
    # to allow the script to load, but training will fail.
    class PositionalEncoding(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.dropout = nn.Identity()
        def forward(self, x): return self.dropout(x)

import math

# Check if torch_geometric is installed
try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import dense_to_sparse, add_self_loops
except ImportError:
    GCNConv = None
    dense_to_sparse = None
    add_self_loops = None
    print("="*50)
    print("WARNING: PyTorch Geometric not found or import failed.")
    print("         GCNEncoder functionality will be disabled.")
    print("="*50)

print("\n--- Model Definitions (Conditional Autoencoder Architecture) ---")


class GCNEncoder(nn.Module):
    """
    Graph Convolutional Network Encoder. (Unchanged - needed for offline step)
    Takes node features (average signals) and graph structure.
    Outputs static enriched state embeddings. Run offline.
    Uses GELU activation and no final output activation.
    """
    def __init__(self, node_feature_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 8, dropout: float = 0.1, activation=F.gelu): # Default GELU
        super().__init__()
        if GCNConv is None: raise ImportError("PyTorch Geometric (GCNConv) required for GCNEncoder.")
        if num_layers < 1: raise ValueError("GCN layers must be >= 1.")

        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.activation = activation
        self.convs = nn.ModuleList()

        self.convs.append(GCNConv(node_feature_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, output_dim))
        else: # Only 1 layer
            self.convs[0] = GCNConv(node_feature_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        current_edge_index = edge_index
        current_edge_weight = edge_weight

        for i in range(self.num_layers):
            x = self.convs[i](x, current_edge_index, current_edge_weight)
            if i < self.num_layers - 1: # Apply activation and dropout to all but the output layer
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # NO final activation applied to GCN output
        return x

# ==============================================================================
# CORRECTED MODEL - GCNTransformerAutoencoder (Conditional Autoencoder)
# ==============================================================================

class GCNTransformerAutoencoder(nn.Module):
    """
    GCN-Transformer Autoencoder for time-series reconstruction (Conditional AE).

    Encodes the input sequence (Y_scaled) using a Transformer Encoder.
    Looks up a pre-computed GCN state embedding (H_c) based on the input's assigned cluster.
    Combines the sequence memory and H_c into a context vector for the Decoder's memory.
    Uses a Transformer Decoder (with Y_scaled as target) to reconstruct the input sequence.
    Optionally inserts a Deconvolution layer before the final output projection.
    """
    def __init__(self,
                 input_dim: int,          # Dimension of input sequence elements (1)
                 seq_len: int,            # Length of input sequences (e.g., 480)
                 n_clusters: int,         # Number of nodes/clusters in the graph (K=9)
                 d_model: int,            # Transformer embedding dimension (e.g., 256 or 512)
                 nhead: int,              # Number of attention heads (e.g., 8)
                 num_encoder_layers: int, # Number of Transformer encoder layers (e.g., 2)
                 num_decoder_layers: int, # Number of Transformer decoder layers (e.g., 2)
                 dim_feedforward: int,    # Dimension of feedforward network (e.g., 1024 or 2048)
                 gcn_out_dim: int,        # Dimension of the pre-computed GCN state embeddings (G=64)
                 dropout: float = 0.1,
                 activation: str = 'gelu', # Activation for Transformer layers ('gelu')
                 use_deconvolution: bool = False, # <<< CONTROL: Set True to add Deconv layer >>>
                 # Deconvolution parameters (only used if use_deconvolution=True)
                 deconv_intermediate_channels: int = 64, # Example intermediate channel dim
                 deconv_kernel_size: int = 7,
                 deconv_stride: int = 1
                 ):
        super().__init__()

        if d_model % nhead != 0:
             raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.d_model = d_model
        self.gcn_out_dim = gcn_out_dim
        self.n_clusters = n_clusters
        self.nhead = nhead
        self.use_deconvolution = use_deconvolution

        # --- Shared Components ---
        self.input_embed = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max(seq_len + 100, 5000))

        # --- Encoder Components ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation=activation, batch_first=True, norm_first=False
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        # --- Context Combination Components ---
        decoder_memory_input_dim = d_model + gcn_out_dim
        self.memory_projection = nn.Linear(decoder_memory_input_dim, d_model)

        # --- Decoder Components ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation=activation, batch_first=True, norm_first=False
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm
        )

        # --- Final Output Layers ---
        if self.use_deconvolution:
            print("INFO: Using Deconvolution layer in the final stage.")
            # Optional Deconvolution Path
            # 1. Permute [B, L, D] -> [B, D, L]
            self.pre_deconv_permute = lambda x: x.permute(0, 2, 1)
            # 2. ConvTranspose1d: Map D -> intermediate_channels, preserve L
            deconv_padding = deconv_kernel_size // 2
            self.deconv_layer = nn.ConvTranspose1d(
                in_channels=d_model,
                out_channels=deconv_intermediate_channels,
                kernel_size=deconv_kernel_size,
                stride=deconv_stride,
                padding=deconv_padding
            )
            self.deconv_activation = nn.GELU()
            # 3. Final Projection/Conv: Map intermediate_channels -> output_dim=1
            # Using a 1x1 convolution is equivalent to a Linear layer applied channel-wise
            self.final_conv = nn.Conv1d(
                in_channels=deconv_intermediate_channels,
                out_channels=input_dim, # input_dim should be 1
                kernel_size=1
            )
            # 4. Permute back: [B, 1, L] -> [B, L, 1]
            self.post_deconv_permute = lambda x: x.permute(0, 2, 1)
        else:
            # Direct Linear Path
            print("INFO: Using direct Linear layer for final output.")
            self.output_linear = nn.Linear(d_model, input_dim)

        # Initialize weights
        self.init_weights()

    def init_weights(self) -> None:
        """Initializes weights for linear layers."""
        initrange = 0.1
        self.input_embed.weight.data.uniform_(-initrange, initrange)
        self.memory_projection.bias.data.zero_()
        self.memory_projection.weight.data.uniform_(-initrange, initrange)
        if hasattr(self, 'output_linear'):
            self.output_linear.bias.data.zero_()
            self.output_linear.weight.data.uniform_(-initrange, initrange)
        # Conv layers have their own default init

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generates a square causal mask for the sequence."""
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)


    def forward(self,
                src: torch.Tensor,           # Input sequence windows: [B, L, 1]
                state_indices: torch.Tensor, # Cluster index for each window: [B]
                all_state_embeddings: torch.Tensor, # PRE-COMPUTED GCN embeddings: [K, G]
                tgt: torch.Tensor = None,    # Target sequence for decoder (usually src) [B, L, 1]
                # Optional masks can be added here if padding is handled
                src_key_padding_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None
                ) -> torch.Tensor:
        """Forward pass of the GCNTransformerAutoencoder."""
        device = src.device
        batch_size = src.size(0)
        seq_len = src.size(1) # Get L from input

        # Validate inputs
        if seq_len != self.seq_len: print(f"Warning: Runtime seq_len {seq_len} != init seq_len {self.seq_len}")
        # Add other validations if needed

        # --- Select GCN State Embedding ---
        if torch.any(state_indices < 0) or torch.any(state_indices >= self.n_clusters):
            raise IndexError(f"State indices out of bounds [0, {self.n_clusters-1}]")
        # Input: [B], [K, G] -> Output: [B, G]
        batch_state_embeddings = all_state_embeddings[state_indices]

        # --- Transformer Encoder ---
        # Input: [B, L, 1] -> [B, L, D]
        src_embedded = self.input_embed(src) * math.sqrt(self.d_model)
        src_embedded = self.pos_encoder(src_embedded)
        # Output: [B, L, D]
        sequence_memory = self.transformer_encoder(
            src_embedded,
            src_key_padding_mask=src_key_padding_mask
        )

        # --- Combine Sequence Memory and GCN State ---
        # Input: [B, G] -> Output: [B, L, G]
        expanded_state_embeddings = batch_state_embeddings.unsqueeze(1).expand(-1, seq_len, -1)
        # Input: [B, L, D], [B, L, G] -> Output: [B, L, D+G]
        combined_memory_features = torch.cat((sequence_memory, expanded_state_embeddings), dim=-1)
        # Input: [B, L, D+G] -> Output: [B, L, D]
        projected_memory = self.memory_projection(combined_memory_features) # This is decoder 'memory'

        # --- Transformer Decoder ---
        if tgt is None: tgt = src # Use source for autoencoding target

        # Input: [B, L, 1] -> [B, L, D]
        tgt_embedded = self.input_embed(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_encoder(tgt_embedded)

        # Create causal mask for target self-attention
        tgt_mask = self._generate_square_subsequent_mask(seq_len, device=device)

        # Input: [B, L, D], [B, L, D] -> Output: [B, L, D]
        decoder_output = self.transformer_decoder(
            tgt=tgt_embedded,
            memory=projected_memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask # Should match src padding if used
        )

        # --- Final Output Stage ---
        if self.use_deconvolution:
            # Deconvolution Path
            # Input: [B, L, D] -> Permute -> [B, D, L]
            permuted_output = self.pre_deconv_permute(decoder_output)
            # Input: [B, D, L] -> Deconv -> [B, C_interm, L]
            deconv_out = self.deconv_activation(self.deconv_layer(permuted_output))
            # Input: [B, C_interm, L] -> Final Conv -> [B, 1, L]
            final_out_permuted = self.final_conv(deconv_out)
            # Input: [B, 1, L] -> Permute -> [B, L, 1]
            output = self.post_deconv_permute(final_out_permuted)
        else:
            # Direct Linear Path
            # Input: [B, L, D] -> Output: [B, L, 1]
            output = self.output_linear(decoder_output)

        return output


# --- Example Usage ---
if __name__ == "__main__":
    print("\n--- Testing GCNTransformerAutoencoder (Conditional AE) ---")

    # Config Parameters (Match successful run)
    _INPUT_DIM = 1; _SEQ_LEN = 480; _N_CLUSTERS = 9; _D_MODEL = 256; _NHEAD = 8
    _NUM_ENC_LAYERS = 2; _NUM_DEC_LAYERS = 2; _DIM_FFWD = 1024; _GCN_OUT = 64
    _DROPOUT = 0.1; _ACTIVATION = 'gelu'; _BATCH_SIZE = 4
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {_DEVICE}")

    # --- Test WITHOUT Deconvolution ---
    print("\nTesting WITHOUT Deconvolution Layer:")
    try:
        autoencoder_linear = GCNTransformerAutoencoder(
            input_dim=_INPUT_DIM, seq_len=_SEQ_LEN, n_clusters=_N_CLUSTERS, d_model=_D_MODEL, nhead=_NHEAD,
            num_encoder_layers=_NUM_ENC_LAYERS, num_decoder_layers=_NUM_DEC_LAYERS, dim_feedforward=_DIM_FFWD,
            gcn_out_dim=_GCN_OUT, dropout=_DROPOUT, activation=_ACTIVATION, use_deconvolution=False
        ).to(_DEVICE)
        print(f"Model Structure (Linear Output):\n{autoencoder_linear}")
        num_params = sum(p.numel() for p in autoencoder_linear.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {num_params:,}")

        sim_src = torch.randn(_BATCH_SIZE, _SEQ_LEN, _INPUT_DIM).to(_DEVICE)
        sim_state_indices = torch.randint(0, _N_CLUSTERS, (_BATCH_SIZE,)).to(_DEVICE)
        sim_all_state_embeddings = torch.randn(_N_CLUSTERS, _GCN_OUT).to(_DEVICE)
        output_recon = autoencoder_linear(sim_src, sim_state_indices, sim_all_state_embeddings, tgt=sim_src)
        assert output_recon.shape == sim_src.shape
        print("Conditional AE (Linear Output) test successful.")
    except Exception as e: print(f"Error: {e}"); traceback.print_exc()

    # --- Test WITH Deconvolution ---
    print("\nTesting WITH Deconvolution Layer:")
    try:
        autoencoder_deconv = GCNTransformerAutoencoder(
            input_dim=_INPUT_DIM, seq_len=_SEQ_LEN, n_clusters=_N_CLUSTERS, d_model=_D_MODEL, nhead=_NHEAD,
            num_encoder_layers=_NUM_ENC_LAYERS, num_decoder_layers=_NUM_DEC_LAYERS, dim_feedforward=_DIM_FFWD,
            gcn_out_dim=_GCN_OUT, dropout=_DROPOUT, activation=_ACTIVATION, use_deconvolution=True,
            # Pass relevant deconv params (example values)
            deconv_intermediate_channels=32, deconv_kernel_size=7, deconv_stride=1
        ).to(_DEVICE)
        print(f"\nModel Structure (Deconv Output):\n{autoencoder_deconv}")
        num_params = sum(p.numel() for p in autoencoder_deconv.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {num_params:,}")

        sim_src = torch.randn(_BATCH_SIZE, _SEQ_LEN, _INPUT_DIM).to(_DEVICE)
        sim_state_indices = torch.randint(0, _N_CLUSTERS, (_BATCH_SIZE,)).to(_DEVICE)
        sim_all_state_embeddings = torch.randn(_N_CLUSTERS, _GCN_OUT).to(_DEVICE)
        output_recon_deconv = autoencoder_deconv(sim_src, sim_state_indices, sim_all_state_embeddings, tgt=sim_src)
        assert output_recon_deconv.shape == sim_src.shape
        print("Conditional AE (Deconv Output) test successful.")
    except Exception as e: print(f"Error: {e}"); traceback.print_exc()

    print("\n--- model.py Test Finished ---")

# --- END OF FILE model.py ---