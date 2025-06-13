# --- START OF FILE utils.py ---

import torch
import torch.nn as nn
import math

print("\n--- Utilities Setup ---") # Simple print to confirm import

class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input sequence embeddings.

    This module generates fixed sinusoidal positional encodings based on the
    formula from "Attention Is All You Need". It adds these encodings to the
    input embeddings to give the model information about the relative or
    absolute position of the tokens (time steps) in the sequence.

    Attributes:
        dropout (nn.Dropout): Dropout layer applied after adding positional encodings.
        pe (torch.Tensor): Buffer storing the pre-calculated positional encodings.
                           Shape: `[1, max_len, d_model]` if batch_first=True perspective,
                           or `[max_len, 1, d_model]` if batch_first=False perspective.
                           We'll adjust based on model's batch_first setting.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model (int): The embedding dimension of the model. Must match the
                           dimension of the input embeddings to the Transformer.
            dropout (float): The dropout probability applied to the final encoded embeddings.
            max_len (int): The maximum possible sequence length for which positional
                           encodings will be pre-calculated. Should be >= the actual
                           sequence length used during training/inference.
        """
        super().__init__()
        # Input validation (optional but good practice)
        if d_model <= 0:
             raise ValueError("d_model must be a positive integer.")
        # The original paper formulation works slightly more elegantly with even d_model,
        # but it's not strictly required for the math to work. We'll keep it flexible.
        # if d_model % 2 != 0:
        #     raise ValueError("d_model should ideally be an even number for standard PositionalEncoding.")

        self.dropout = nn.Dropout(p=dropout)

        # --- Pre-calculate Positional Encodings ---
        # `position`: Tensor of sequence positions [0, 1, ..., max_len-1]. Shape: [max_len, 1]
        position = torch.arange(max_len).unsqueeze(1)

        # `div_term`: Term for calculating frequencies. Shape: [d_model / 2]
        # Creates terms like 1/10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Initialize positional encoding tensor `pe`. Shape: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # Calculate sin for even indices and cos for odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Handle odd d_model case: ensure we don't go out of bounds for cosine calculation
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
             # Calculate cosine only up to the second-to-last dimension
             pe[:, 1::2] = torch.cos(position * div_term[:-1]) # Use div_term excluding the last element if d_model is odd (though less common)


        # Add batch dimension for easier broadcasting: [max_len, d_model] -> [1, max_len, d_model]
        # This suits `batch_first=True` format often used in PyTorch Transformers.
        pe = pe.unsqueeze(0) # Shape becomes [1, max_len, d_model]

        # Register `pe` as a buffer. Buffers are part of the model's state_dict
        # but are not optimized by the optimizer. They are moved to device with `.to(device)`.
        self.register_buffer('pe', pe) # Name it 'pe'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor containing sequence embeddings.
                              Expected shape: [batch_size, seq_len, d_model] (if batch_first=True).

        Returns:
            torch.Tensor: The input tensor with added positional encoding.
                          Output shape: [batch_size, seq_len, d_model].
        """
        # x shape: [batch_size, seq_len, d_model]
        # self.pe shape: [1, max_len, d_model]
        # We need positional encodings for the length of the input sequence (`x.size(1)`).
        # Slicing `self.pe[:, :x.size(1)]` gives shape [1, seq_len, d_model].
        # This broadcasts correctly during addition with x.
        x = x + self.pe[:, :x.size(1), :] # Use buffer directly
        return self.dropout(x)

# --- Example Usage (for testing this script directly) ---
if __name__ == "__main__":
    print("\n--- Testing PositionalEncoding ---")
    d_model_test = 512 # Example embedding dimension
    seq_len_test = 100  # Example sequence length
    batch_size_test = 4 # Example batch size
    max_len_test = 500  # Max length for PE calculation

    # Create dummy input tensor (batch_first=True)
    dummy_input = torch.randn(batch_size_test, seq_len_test, d_model_test)
    print(f"Input shape: {dummy_input.shape}")

    # Initialize Positional Encoding
    pos_encoder = PositionalEncoding(d_model=d_model_test, max_len=max_len_test)
    print(f"PositionalEncoding module initialized (max_len={max_len_test}, d_model={d_model_test}).")
    print(f"Internal 'pe' buffer shape: {pos_encoder.pe.shape}") # Should be [1, max_len_test, d_model_test]

    # Apply positional encoding
    output = pos_encoder(dummy_input)
    print(f"Output shape after PositionalEncoding: {output.shape}") # Should match input shape

    # Check if values changed (they should have)
    print(f"Input equals Output check (should be False): {torch.equal(dummy_input, output)}")

    # Check positional encoding values are different across sequence dimension
    print(f"PE for timestep 0 vs 1 (should be different):")
    print(f"  PE[0, 0, :8]: {pos_encoder.pe[0, 0, :8].numpy()}")
    print(f"  PE[0, 1, :8]: {pos_encoder.pe[0, 1, :8].numpy()}")
    print(f"PE for timestep 99 vs 100 (using max_len):")
    print(f"  PE[0, 99, :8]: {pos_encoder.pe[0, 99, :8].numpy()}")
    # Need to check if max_len_test > 100
    if max_len_test > 100:
      print(f"  PE[0, 100, :8]: {pos_encoder.pe[0, 100, :8].numpy()}")


    print("\n--- utils.py Test Finished Successfully ---")


# --- END OF FILE utils.py ---