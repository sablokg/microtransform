#!/usr/bin/env python3
"""
Train a small Transformer autoencoder on DNA sequences.

Original script had several bugs that were fixed here:
  - `dataframe` argument was never used (code referenced an undefined `df`)
  - `df[labels]` referenced an undefined name `labels` instead of a column name string
  - `encode_sequence` was nested with broken indentation, and the rest of the
    script was accidentally left inside `transformer(...)`'s body
  - Sequences were padded with "N" but the base->index mapping only covered
    A/C/G/T, so encoding a padded sequence raised a KeyError
  - Missing imports (numpy, pandas, torch, sklearn, fastai)
  - Nothing was ever done with the trained model (no saving)

Usage:
    python dna_autoencoder_cli.py --input-csv data.csv --seq-col sequences
"""

import click
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.losses import CrossEntropyLossFlat

# Include "N" as a 5th token so padded positions are legal to encode.
BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
NUM_CLASSES = len(BASE_TO_IDX)
PAD_TOKEN = "N"
PAD_IDX = BASE_TO_IDX[PAD_TOKEN]


def encode_sequence(sequence):
    """Map a DNA string to a list of integer tokens."""
    return [BASE_TO_IDX[base] for base in sequence]


def pad_sequences(sequences, pad_char=PAD_TOKEN):
    """Right-pad all sequences to the length of the longest one."""
    maxlen = max(len(s) for s in sequences)
    return [s + pad_char * (maxlen - len(s)) for s in sequences]


def load_sequences(csv_path, seq_col, label_col=None):
    """Read sequences (and optional labels) from a CSV file."""
    dataframe = pd.read_csv(csv_path)

    if seq_col not in dataframe.columns:
        raise click.ClickException(
            f"Column '{seq_col}' not found in {csv_path}. "
            f"Available columns: {list(dataframe.columns)}"
        )

    sequences = dataframe[seq_col].astype(str).to_list()

    labels = None
    if label_col is not None:
        if label_col not in dataframe.columns:
            raise click.ClickException(
                f"Column '{label_col}' not found in {csv_path}. "
                f"Available columns: {list(dataframe.columns)}"
            )
        labels = dataframe[label_col].to_list()

    return sequences, labels


class DNATransformerAutoencoder(nn.Module):
    def __init__(
        self,
        seq_len,
        num_classes=NUM_CLASSES,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        latent_dim=32,
        dropout=0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.position = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.to_latent = nn.Linear(embed_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, embed_dim)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        self.output = nn.Linear(embed_dim, num_classes)

   def forward(self, x):
     padding_mask = x == PAD_IDX
     x = self.embedding(x)
     x = x + self.position
     x = self.dropout(x)
     x = self.transformer_encoder(
        x,
        src_key_padding_mask=padding_mask
     ) 
     z = self.to_latent(x)
     x = self.from_latent(z)
     x = self.transformer_decoder(
        x,
        src_key_padding_mask=padding_mask
     )
    return self.output(x).permute(0, 2, 1)


def build_dataloaders(sequences, test_size, seed, batch_size):
    padded = pad_sequences(sequences)
    X = np.array([encode_sequence(seq) for seq in padded], dtype=np.int64)

    X_train, X_valid = train_test_split(
        X,
        test_size=test_size,
        random_state=seed,
    )

    X_train = torch.tensor(X_train, dtype=torch.long)
    X_valid = torch.tensor(X_valid, dtype=torch.long)

    train_ds = TensorDataset(X_train, X_train)
    valid_ds = TensorDataset(X_valid, X_valid)

    dls = DataLoaders.from_dsets(train_ds, valid_ds, bs=batch_size)
    return dls, X_train.shape[1]


@click.command()
@click.option(
    "--input-csv",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="CSV file containing a column of DNA sequences.",
)
@click.option("--seq-col", default="sequences", show_default=True, help="Name of the sequence column.")
@click.option(
    "--label-col",
    default=None,
    help="Optional label column (loaded but unused by this unsupervised autoencoder).",
)
@click.option("--epochs", default=10, show_default=True, help="Number of training epochs.")
@click.option("--lr", default=1e-3, show_default=True, help="Learning rate.")
@click.option("--batch-size", default=64, show_default=True, help="Batch size.")
@click.option("--embed-dim", default=64, show_default=True, help="Embedding dimension.")
@click.option("--num-heads", default=4, show_default=True, help="Number of attention heads.")
@click.option("--num-layers", default=2, show_default=True, help="Number of transformer layers (encoder and decoder each).")
@click.option("--latent-dim", default=32, show_default=True, help="Bottleneck latent dimension.")
@click.option("--dropout", default=0.2, show_default=True, help="Dropout probability.")
@click.option("--valid-pct", default=0.2, show_default=True, help="Fraction of data held out for validation.")
@click.option("--seed", default=42, show_default=True, help="Random seed for the train/valid split.")
@click.option(
    "--out-model",
    default="dna_autoencoder.pth",
    show_default=True,
    type=click.Path(dir_okay=False),
    help="Path to save the trained model weights.",
)
def main(
    input_csv,
    seq_col,
    label_col,
    epochs,
    lr,
    batch_size,
    embed_dim,
    num_heads,
    num_layers,
    latent_dim,
    dropout,
    valid_pct,
    seed,
    out_model,
):
    """Train a Transformer autoencoder on DNA sequences from INPUT_CSV."""
    click.echo(f"Loading sequences from {input_csv} (column '{seq_col}')...")
    sequences, labels = load_sequences(input_csv, seq_col, label_col)
    click.echo(f"Loaded {len(sequences)} sequences.")

    dls, seq_len = build_dataloaders(sequences, valid_pct, seed, batch_size)
    click.echo(f"Padded sequence length: {seq_len}")

    model = DNATransformerAutoencoder(
        seq_len=seq_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        latent_dim=latent_dim,
        dropout=dropout,
    )

    xb, yb = dls.one_batch()
    out = model(xb)
    click.echo(f"input : {tuple(xb.shape)}")
    click.echo(f"output: {tuple(out.shape)}")

    learner = Learner(
        dls,
        model,
        loss_func=CrossEntropyLossFlat(ignore_index=PAD_IDX),
    )
    learner.fit(epochs, lr=lr)

    torch.save(model.state_dict(), out_model)
    click.echo(f"Saved model weights to {out_model}")


if __name__ == "__main__":
    main()