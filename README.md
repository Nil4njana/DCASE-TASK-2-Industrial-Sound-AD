# DCASE-TASK-2-Industrial-Sound-AD
End-to-end unsupervised anomaly detection pipeline for industrial machines under domain shift - transfer learning with PANNs, inverse-Mel features, and domain-aware k-NN scoring.

# 🔊 Industrial Anomaly Sound Detection — DCASE 2025 Task 2

## Problem Statement

Industrial machines produce characteristic sounds during normal operation.
Detecting deviations from these patterns (anomalies) is critical for predictive
maintenance. This project tackles **DCASE 2025 Challenge Task 2**, which imposes
two hard constraints:

- **No anomaly labels** during training — only normal sounds available
- **Domain shift** — test audio comes from an unseen target domain with
  different acoustic conditions than training (source domain)

## Architecture

- **Input**: Raw audio → 3-channel multi-resolution spectrogram (STFT 256/1024/4096) + inverse-Mel (500–4000 Hz)
- **Backbone**: PANNs CNN14 (frozen, pre-trained on AudioSet 2M clips) → 2048D embeddings
- **Layer 1**: Multi-k ensemble k-NN (k = 1, 3, 5, 7, 10) with weighted averaging
- **Layer 2**: Mahalanobis distance scoring per domain
- **Layer 3**: Score fusion — min-max normalized combination of Layers 1 & 2
- **Layer 4**: Domain-aware + section-aware z-score normalization
- **Output**: Final anomaly score per audio clip

## 📊 Results

- **Average AUC**: 57.16% across 7 machines (baseline: 50.00%)
- **Best machine**: Valve — 71.25% AUC, 28.90% pAUC, 41.12% official score
- **Other machines**: ToyTrain 57.20% · Gearbox 56.46% · Slider 55.18% · Fan 54.17% · ToyCar 53.38% · Bearing 52.50%

## Key Contributions

### 1. Multi-Resolution Spectrogram with Inverse-Mel
Standard mel-spectrograms compress frequency resolution. This pipeline uses
**three STFT resolutions simultaneously**:
- **Fine (256)** — transient events, high-frequency details
- **Mid (1024)** — rhythmic patterns, mechanical vibrations
- **Coarse (4096)** — overall harmonic structure, low-frequency content

Combined with **inverse-Mel reconstruction** to recover linear frequency
resolution in the industrial sound range (500–4000 Hz).

### 2. Transfer Learning Without Fine-Tuning
**PANNs CNN14**, pre-trained on 2 million AudioSet clips, is used as a
**completely frozen** feature extractor. No gradient updates — purely
zero-shot transfer to industrial audio.

### 3. 4-Layer Anomaly Scoring Pipeline
A custom multi-strategy scoring system designed to handle source/target
domain imbalance:
- **Multi-k ensemble**: weighted average of k-NN distances at k=1,3,5,7,10
- **Mahalanobis distance**: covariance-aware scoring per domain
- **Score fusion**: min-max normalized combination of both
- **Normalization**: domain-specific and section-specific z-score normalization

### 4. Supplemental Data Integration
Automatically detects and incorporates supplemental training audio
(up to 500 samples per machine) with configurable sample weighting.

