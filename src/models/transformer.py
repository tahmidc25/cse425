"""
Task 3: Transformer for Long-Form Music Generation
Architecture:
- Decoder-only Transformer (GPT-style)
- d_model=512, nhead=8, num_layers=6, feedforward_dim=2048
- Causal (triangular) masking for autoregressive generation
- Sinusoidal positional encoding
- Vocabulary size: 228 (tokens)
- Max sequence length: 512
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int = 512, max_len: int = 512):
        """
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor (batch, seq_len, d_model)
        
        Returns:
            x + positional encoding
        """
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    """Single transformer decoder block."""
    
    def __init__(self, d_model: int = 512, nhead: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input (batch, seq_len, d_model)
            attn_mask: Attention mask for causality
        
        Returns:
            Output (batch, seq_len, d_model)
        """
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + self.dropout1(attn_out)
        
        # Feedforward with pre-norm
        x_norm = self.norm2(x)
        ff_out = self.linear2(F.relu(self.linear1(x_norm)))
        x = x + self.dropout2(ff_out)
        
        return x


class MusicTransformer(nn.Module):
    """
    Decoder-only Transformer for music token generation.
    """
    
    def __init__(
        self,
        vocab_size: int = 228,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        """
        Args:
            vocab_size: Token vocabulary size
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal attention mask (lower triangular).
        
        Args:
            seq_len: Sequence length
            device: Device
        
        Returns:
            Causal mask (seq_len, seq_len) with -inf for future positions
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
        return mask
    
    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            token_ids: Token indices (batch, seq_len)
            attention_mask: Optional mask for padding tokens
        
        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.size()
        
        # Embedding
        x = self.embedding(token_ids)  # (batch, seq_len, d_model)
        x = x * math.sqrt(self.d_model)
        
        # Positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len, token_ids.device)
        
        # Transformer layers
        for block in self.transformer_blocks:
            x = block(x, attn_mask=causal_mask)
        
        # Output projection
        logits = self.output_proj(x)  # (batch, seq_len, vocab_size)
        
        return logits
    
    def generate(
        self,
        start_token: int = 0,  # BOS token
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Autoregressive token generation.
        
        Args:
            start_token: Starting token ID
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling (if None, regular sampling)
            device: Device to generate on
        
        Returns:
            Generated token sequence (1, max_length)
        """
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        
        # Start with BOS token
        tokens = torch.tensor([[start_token]], dtype=torch.long, device=device)
        
        with torch.no_grad():
            for _ in range(max_length - 1):
                # Forward pass
                logits = self.forward(tokens)
                
                # Get last token logits
                next_logits = logits[:, -1, :]  # (batch, vocab_size)
                
                # Apply temperature
                next_logits = next_logits / temperature
                
                # Top-k sampling
                if top_k is not None:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float('-inf')
                
                # Convert to probabilities
                next_probs = F.softmax(next_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(next_probs, num_samples=1)
                
                # Append token
                tokens = torch.cat([tokens, next_token], dim=1)
                
                # Stop if EOS token (226)
                if next_token.item() == 226:
                    break
        
        return tokens


def cross_entropy_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    pad_token_id: int = 227,
    ignore_index: bool = True
) -> torch.Tensor:
    """
    Compute cross-entropy loss for token prediction.
    
    Args:
        logits: Model output (batch, seq_len, vocab_size)
        target_ids: Target token IDs (batch, seq_len)
        pad_token_id: Token ID for padding
        ignore_index: Whether to ignore padding tokens
    
    Returns:
        Scalar loss
    """
    batch_size, seq_len, vocab_size = logits.size()
    
    # Reshape for cross-entropy
    logits_flat = logits.view(batch_size * seq_len, vocab_size)
    target_flat = target_ids.view(batch_size * seq_len)
    
    if ignore_index:
        loss = F.cross_entropy(logits_flat, target_flat, ignore_index=pad_token_id)
    else:
        loss = F.cross_entropy(logits_flat, target_flat)
    
    return loss


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss
    
    Returns:
        Perplexity = exp(loss)
    """
    return math.exp(loss)


if __name__ == "__main__":
    # Test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MusicTransformer(
        vocab_size=228,
        d_model=512,
        nhead=8,
        num_layers=6,
        max_seq_len=512
    ).to(device)
    
    # Dummy input
    token_ids = torch.randint(0, 228, (2, 100)).to(device)
    
    logits = model(token_ids)
    print(f"Input shape: {token_ids.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Loss
    target = torch.randint(0, 228, (2, 100)).to(device)
    loss = cross_entropy_loss(logits, target)
    print(f"Loss: {loss.item():.4f}")
    print(f"Perplexity: {compute_perplexity(loss.item()):.4f}")
    
    # Generation
    generated = model.generate(start_token=0, max_length=50, device=device)
    print(f"Generated shape: {generated.shape}")
