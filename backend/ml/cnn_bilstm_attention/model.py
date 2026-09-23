import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class TemporalSelfAttention(nn.Module):
    """
    Additive Self-Attention mechanism over temporal feature frames.
    Computes an energy-weighted context vector summarizing the phonation sequence.
    """

    def __init__(self, input_dim: int, attention_dim: int = 64):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (Batch, Time, input_dim)
        Returns:
            context: Tensor of shape (Batch, input_dim)
            weights: Tensor of shape (Batch, Time, 1)
        """
        # scores: (Batch, Time, 1)
        scores = self.projection(x)
        # weights: (Batch, Time, 1) normalized across Time dimension
        weights = F.softmax(scores, dim=1)
        # context: (Batch, input_dim)
        context = torch.sum(weights * x, dim=1)
        return context, weights


class AudioCNNBilstmAttention(nn.Module):
    """
    2D CNN + BiLSTM + Attention architecture for laryngeal voice classification.
    
    Pipeline:
    1. Input: (Batch, 1, Freq=128, Time=128) Log-Mel Spectrogram.
    2. 2D CNN: Multiple Conv2d + BatchNorm2d + ReLU + MaxPool2d + Dropout2d blocks.
    3. Temporal Sequence Reshaping: Collapses frequency/channel dimensions per time frame.
    4. BiLSTM: Models recurrent temporal acoustic dynamics.
    5. Attention: Aggregates BiLSTM states into a fixed-length context vector.
    6. Dense Classifier: Fully connected layer with Dropout and 6-class Softmax logits.
    """

    def __init__(
        self,
        n_classes: int = 6,
        cnn_channels: List[int] = [32, 64, 128],
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        dense_dropout: float = 0.3,
        cnn_dropout: float = 0.1
    ):
        super().__init__()
        self.n_classes = n_classes
        self.cnn_channels = cnn_channels
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        # 1. 2D CNN Backbone
        layers = []
        in_c = 1
        for out_c in cnn_channels:
            layers.extend([
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(cnn_dropout)
            ])
            in_c = out_c
        self.cnn = nn.Sequential(*layers)

        # Calculate feature dimension after CNN pooling
        # Input: 128 x 128
        # Each block reduces spatial dimensions by 2:
        num_pools = len(cnn_channels)
        downsample_factor = 2 ** num_pools
        freq_out = 128 // downsample_factor
        time_out = 128 // downsample_factor

        feat_per_time = cnn_channels[-1] * freq_out
        self.seq_proj = nn.Linear(feat_per_time, 128)

        # 2. Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cnn_dropout if lstm_layers > 1 else 0.0
        )

        # 3. Temporal Attention Mechanism
        bilstm_output_dim = lstm_hidden * 2
        self.attention = TemporalSelfAttention(bilstm_output_dim, attention_dim=64)

        # 4. Dense Classifier
        self.classifier = nn.Sequential(
            nn.Linear(bilstm_output_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dense_dropout),
            nn.Linear(64, n_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: (Batch, 1, 128, 128)
        Returns:
            logits: (Batch, n_classes)
            attention_weights: (Batch, Time_steps, 1)
        """
        # CNN feature maps: (Batch, C, F, T)
        feat = self.cnn(x)
        B, C, F_dim, T_dim = feat.shape

        # Reshape to temporal sequence: (Batch, Time, Channels * Freq)
        seq = feat.permute(0, 3, 1, 2).contiguous().view(B, T_dim, C * F_dim)
        seq = F.relu(self.seq_proj(seq))

        # BiLSTM: (Batch, Time, 2 * lstm_hidden)
        lstm_out, _ = self.lstm(seq)

        # Attention: (Batch, 2 * lstm_hidden)
        context, attn_weights = self.attention(lstm_out)

        # Final logits: (Batch, n_classes)
        logits = self.classifier(context)
        return logits, attn_weights
