"""
Pure PyTorch replacements for flash_attn.bert_padding utilities.
Drop-in compatible with: pad_input, unpad_input, index_first_axis, rearrange
"""

import torch
import torch.nn.functional as F


def unpad_input(hidden_states, attention_mask):
    """Remove padding tokens from batched sequences.
    
    Args:
        hidden_states: (batch, seqlen, ...) - input tensor
        attention_mask: (batch, seqlen) - 1 for real tokens, 0 for padding
    
    Returns:
        hidden_states_unpad: (total_nnz, ...) - unpadded hidden states
        indices: (total_nnz,) - indices of non-padding tokens in flattened input
        cu_seqlens: (batch + 1,) - cumulative sequence lengths
        max_seqlen_in_batch: int - maximum sequence length in the batch
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    
    # Flatten batch and seqlen dims, then index
    flat = hidden_states.reshape(-1, *hidden_states.shape[2:])
    hidden_states_unpad = flat[indices]
    
    return hidden_states_unpad, indices, cu_seqlens, max_seqlen_in_batch


def pad_input(hidden_states, indices, batch, seqlen):
    """Pad unpadded hidden states back to (batch, seqlen, ...) shape.
    
    Args:
        hidden_states: (total_nnz, ...) - unpadded hidden states
        indices: (total_nnz,) - original indices from unpad_input
        batch: int - batch size
        seqlen: int - sequence length
    
    Returns:
        output: (batch, seqlen, ...) - padded hidden states
    """
    output = torch.zeros(
        batch * seqlen, *hidden_states.shape[1:],
        device=hidden_states.device, dtype=hidden_states.dtype
    )
    output[indices] = hidden_states
    return output.reshape(batch, seqlen, *hidden_states.shape[1:])


def index_first_axis(x, indices):
    """Index into the first axis of a tensor.
    
    Args:
        x: (total, ...) - input tensor  
        indices: (n,) - indices to select
    
    Returns:
        output: (n, ...) - selected elements
    """
    return x[indices]


def rearrange(x, pattern):
    """Simplified rearrange that handles 'b s ... -> (b s) ...' pattern.
    
    This only supports the specific pattern used in the codebase:
    merging batch and sequence dimensions.
    """
    if "b s" in pattern and "(b s)" in pattern:
        return x.reshape(-1, *x.shape[2:])
    raise NotImplementedError(f"Pattern '{pattern}' not supported. Only 'b s ... -> (b s) ...' is supported.")
