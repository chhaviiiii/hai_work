# VTNet Training & Evaluation

## Overview
This repository contains code for training and evaluating VTNet models on eye‑tracking `.pkl` data, as well as a SLURM script for easy submission to UBC’s HPC.

## 1. Data Loading & Preparation
Scan for .pkl files
We look in your output/ folder for all the eye‑tracking .pkl pickles you produced.

Per‐file processing
Each file is a DataFrame of time‑series gaze features (e.g. left/right gaze coordinates, pupil size, fixation points).

Label lookup
We match each pickle’s subject/task ID to its binary label (e.g. high vs low VerbalWM) from your CSV.

Sequence trimming / padding
To feed into the model, each sequence is truncated (or cyclically split) to a fixed length (max_seq_len) and padded if too short—so every example ends up with the same shape.

Train/Validation split
We randomly hold out ~20% of your subjects for validation, keeping 80% for training.

##2. Model Instantiation
We build a two‑branch VTNet:

Temporal branch: a small GRU that ingests the raw time series.

Spatial branch: a 2‑layer CNN that ingests a “scanpath image” generated on the fly from the same sequence.

Optionally we wrap one or both branches in a multi‑head attention layer.

## 3. Training Loop
Epochs: we repeat over the data for N passes (e.g. 20 epochs).

Mini‑batches: in each epoch we load batches of, say, 32 sequences + labels.

Forward pass: each batch goes through the model to produce a score (0…1).

Loss computation: we compute binary cross‑entropy between predictions and true labels.

Backward pass & optimizer step: gradients flow back and we update the model weights (e.g. via Adam).

Logging: after every batch or every few batches we print running loss and accuracy to the console.

## 4. Validation / Evaluation
At the end of each epoch (or at regular intervals), we run the model on the held‑out validation set, without updating weights.

We compute metrics:

Accuracy (percentage of correct high/low predictions)

AUC‑ROC (area under the ROC curve, for threshold‑independent performance).

We track these to detect overfitting and to pick the best checkpoint.

## 5. Model Checkpointing
Whenever validation AUC (or accuracy) improves, we save the model’s weights and optimizer state to trained/ so you can later reload the best version.
