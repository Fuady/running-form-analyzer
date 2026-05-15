"""
bilstm_model.py
───────────────
Modeling: Bidirectional LSTM with attention for running form classification.

Architecture:
  Input  : (batch, seq_len=30, n_features=18)
  → Input projection  : Linear(18, 36) → LayerNorm → GELU
  → BiLSTM layer 1    : hidden=128, bidirectional → Dropout(0.3)
  → BiLSTM layer 2    : hidden=64,  bidirectional → Dropout(0.3)
  → Temporal attention : softmax over timesteps
  → Linear(256, 128)  → GELU → Dropout(0.2)
  → Linear(128, 4)    → log-softmax
  Output : (batch, 4) class log-probabilities

Why BiLSTM?
  - Forward LSTM: reads the stride build-up (preparation → stance → push-off)
  - Backward LSTM: reads the follow-through back to preparation
  - Attention: learns which gait phase is most diagnostic per fault class

Smoke test:
    python src/modeling/bilstm_model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Scaled dot-product attention over the time axis."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, H)
        scores  = self.v(torch.tanh(self.W(x)))          # (B, T, 1)
        weights = F.softmax(scores, dim=1)               # (B, T, 1)
        context = (weights * x).sum(dim=1)               # (B, H)
        return context, weights.squeeze(-1)              # (B,H), (B,T)


class RunningFormClassifier(nn.Module):
    """
    Bidirectional LSTM + attention for 4-class running form classification.

    Parameters
    ----------
    input_size  : number of biomechanical features per timestep
    hidden_size : LSTM hidden units per direction
    num_layers  : stacked LSTM layers
    num_classes : number of form classes (default 4)
    dropout     : dropout probability
    """

    def __init__(
        self,
        input_size:  int = 18,
        hidden_size: int = 128,
        num_layers:  int = 2,
        num_classes: int = 4,
        dropout:     float = 0.3,
    ):
        super().__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.num_classes = num_classes

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.LayerNorm(input_size * 2),
            nn.GELU(),
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size   = input_size * 2,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            bidirectional= True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_size * 2  # bidirectional

        # Attention
        self.attention = TemporalAttention(lstm_out_dim)

        # Classifier head
        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p.data)
            elif "bias" in name:
                p.data.fill_(0)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x               : (B, T, input_size)
            return_attention : also return attention weights

        Returns:
            logits  : (B, num_classes)
            weights : (B, T)  — only if return_attention=True
        """
        x     = self.input_proj(x)                  # (B, T, input*2)
        out, _= self.lstm(x)                        # (B, T, hidden*2)
        ctx, w= self.attention(out)                  # (B, hidden*2), (B, T)
        logits= self.head(ctx)                      # (B, num_classes)

        if return_attention:
            return logits, w
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices (no grad)."""
        self.eval()
        with torch.no_grad():
            return self.forward(x).argmax(dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(
    input_size:  int   = 18,
    hidden_size: int   = 128,
    num_layers:  int   = 2,
    num_classes: int   = 4,
    dropout:     float = 0.3,
) -> RunningFormClassifier:
    model = RunningFormClassifier(input_size, hidden_size, num_layers, num_classes, dropout)
    print(f"RunningFormClassifier | {count_parameters(model):,} trainable parameters")
    return model


if __name__ == "__main__":
    B, T, F = 8, 30, 18
    x = torch.randn(B, T, F)
    model = build_model(input_size=F)

    logits, attn = model(x, return_attention=True)
    probs = F.softmax(logits, dim=1)

    print(f"Input   : {x.shape}")
    print(f"Logits  : {logits.shape}  range [{logits.min():.2f}, {logits.max():.2f}]")
    print(f"Probs   : {probs.shape}   sum={probs.sum(1).mean():.3f}")
    print(f"Attention: {attn.shape}")
    assert logits.shape == (B, 4)
    assert attn.shape   == (B, T)
    print("✅ Smoke test passed")
