"""
Audio Event Tagging for CochlScene Dataset using BEATs Model with FSD50K Compatibility

This script performs audio event tagging on CochlScene dataset using the pre-trained BEATs model
with compatibility for FSD50K's 200-class vocabulary through post-processing filtering.

The pipeline:
    1. Loads the BEATs model pre-trained on AudioSet and fine-tuned on AS2M.
    2. Loads the class label mapping from BEATs index to FSD50K-compatible event names.
    3. Processes a single audio file from CochlScene, resampled to 16 kHz.
    4. Predicts event probabilities and outputs the top-10 predicted labels with their probabilities.

References:
[1] BEATs: Audio Pre-Training with Acoustic Tokenizers (https://arxiv.org/abs/2212.09058)
[2] CochlScene: Acquisition of acoustic scene data using crowdsourcing
[3] FSD50K: An Open Dataset of Human-Labeled Sound Events
[4] AudioSet: A large-scale dataset of audio events (https://arxiv.org/abs/1709.05844)
"""

import sys
from pathlib import Path
import librosa
import pandas as pd
import torch
from model.beats.BEATs import BEATsConfig, BEATs


audio_dir = f".../CochlScene"  # Path to the CochlScene audio file or directory (to be modified)
sr = 16000                     # Target sampling rate (16 kHz, required by BEATs)
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

event_labels_df = pd.read_csv("model/beats/class_labels_indices.csv")
mid_to_label = {mid: label for mid, label in zip(event_labels_df["mid"], event_labels_df["display_name"])}

ckpt_event_model = ".../BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"  # Path to the BEATs checkpoint
checkpoint = torch.load(ckpt_event_model, map_location=device)            # Load checkpoint (weights and config)
cfg = BEATsConfig(checkpoint['cfg'])                                      # Extract model configuration
event_tagging_model = BEATs(cfg).to(device)                               # Instantiate model and move to device
event_tagging_model.load_state_dict(checkpoint['model'])                  # Load pre-trained weights
event_tagging_model.eval()                                                # Set to evaluation mode (no dropout etc.)

# Load audio file, resample to 16 kHz, convert to mono
audio_input_16khz, _ = librosa.load(audio_dir, sr=sr)

# Convert numpy array to torch tensor, add batch dimension, move to device
audio_input_16khz = torch.from_numpy(audio_input_16khz).float().unsqueeze(0).to(device)

# Create padding mask (BEATs expects a mask of shape (batch, time) indicating padded positions)
# Here we assume fixed-length input of 10 seconds (160000 samples) – adjust if needed.
padding_mask = torch.zeros(1, 160000).bool().to(device)

# The extract_features method returns a tuple; the first element contains class probabilities.
probs = event_tagging_model.extract_features(audio_input_16khz, padding_mask=padding_mask)[0]

# probs.topk(k=10) returns (values, indices) for the 10 highest probabilities
for i, (top5_label_prob, top5_label_idx) in enumerate(zip(*probs.topk(k=10))):
    # Convert model output indices to AudioSet MIDs using the checkpoint's label dictionary
    top5_mid = [checkpoint['label_dict'][label_idx.item()] for label_idx in top5_label_idx]
    # Map MID to human-readable label using the pre-loaded mapping
    top5_label = [mid_to_label[mid] for mid in top5_mid]
    print(f'Top 5 predicted labels of the {i}th audio are {top5_label} with probability of {top5_label_prob}')
