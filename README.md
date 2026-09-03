# Lunar Image Correspondence

## SIH 2026 — Problem Statement PS 26166

Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images.

### Objective

Develop a software prototype for correspondence and registration between Chandrayaan-2 optical imagery and lunar reference imagery.

The system aims to identify corresponding features between a moving/source image and a reference image despite differences in illumination, scale and viewing geometry.

---

## Core Pipeline

```text
Two Lunar Images
       ↓
Preprocessing
       ↓
Feature Extraction
(SIFT / SuperPoint)
       ↓
Feature Matching
(BF/FLANN / LightGlue)
       ↓
Match Filtering
       ↓
Geometric Verification
(RANSAC / MAGSAC++)
       ↓
Transformation Estimation
       ↓
Image Registration
       ↓
Refinement
       ↓
Evaluation
```

---

## Initial Data Sources

### Chandrayaan-2

- OHRC
- TMC-2
- IIRS

### Reference Imagery

- LROC NAC
- Other lunar reference imagery where appropriate

---

## Evaluation Metrics

The prototype will evaluate:

- Number of matches
- Number of inliers
- Inlier ratio
- RMSE
- Spatial distribution / uniformity
- Runtime

---

## Project Structure

```text
src/
├── preprocessing/
├── features/
├── matching/
├── geometry/
├── refinement/
├── registration/
└── metrics/

backend/
frontend/
experiments/
notebooks/
docs/
data/
```

---

## Development Strategy

The project will first establish a reliable classical computer-vision baseline using SIFT and robust geometric verification.

A pretrained deep-learning correspondence pipeline using SuperPoint + LightGlue will then be evaluated against the classical baseline.

No model training from scratch is planned for the 5-day prototype.

---

## Team

SIH 2026 Team

- Algorithm Lead
- Classical Computer Vision
- Deep Learning & Refinement
- Backend
- Frontend
- Research & Documentation

---

## Current Status

### Day 1

- Repository setup
- Dataset investigation
- Classical CV baseline preparation
- Deep-learning environment setup
- System architecture definition
