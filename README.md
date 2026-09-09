# Lunar Image Correspondence

## SIH 2026 — Problem Statement PS 26166

**Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC-2 and IIRS)**

A research-oriented software prototype for establishing reliable image correspondences and geometric registration between lunar optical imagery acquired under different imaging conditions.

The system is designed to handle differences in:

- Illumination caused by changing Sun azimuth and elevation
- Image scale and ground sampling distance
- Rotation and viewpoint
- Perspective and geometric distortion
- Sensor/modality characteristics
- Uneven spatial distribution of feature matches

The objective is not simply to maximize the number of matches, but to produce **geometrically consistent, spatially distributed and quantitatively evaluated correspondences** suitable for lunar image registration.

---

## Problem

Images of the same lunar region can appear substantially different because they may have been acquired:

- at different times,
- under different Sun angles,
- from different viewing geometries,
- at different scales,
- using different optical instruments.

Traditional feature matching can therefore produce few correspondences, clustered matches, or geometrically incorrect matches.

This project investigates a hybrid registration pipeline combining classical computer vision, learned feature matching and robust geometric verification.

---

## Proposed Pipeline

```text
Moving / Source Image
        +
Reference / Fixed Image
        │
        ▼
┌───────────────────────────────┐
│ Sensor / Geometry Awareness   │
│ Metadata + Overlap Analysis   │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ Lunar Image Preprocessing     │
│                               │
│ • Grayscale / normalization   │
│ • Illumination handling       │
│ • CLAHE / contrast processing │
│ • Destriping where required   │
│ • Geographic overlap cropping │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ Feature Extraction            │
│                               │
│ • SIFT                        │
│ • SuperPoint                  │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ Feature Matching              │
│                               │
│ • BF / FLANN                  │
│ • LightGlue                   │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ Match Filtering               │
│                               │
│ • Ratio / confidence checks   │
│ • Duplicate filtering         │
│ • Candidate correspondence set│
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ Geometric Verification        │
│                               │
│ • RANSAC / MAGSAC++           │
│ • Similarity                  │
│ • Affine                      │
│ • Homography                  │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ Spatial Correspondence        │
│ Selection                     │
│                               │
│ • Grid-based distribution     │
│ • Coverage measurement        │
│ • Uniformity analysis         │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ Registration / Refinement     │
│                               │
│ • Image warping               │
│ • Local / sub-pixel refinement│
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ Evaluation                    │
│                               │
│ • RMSE                        │
│ • Inlier count                │
│ • Inlier ratio                │
│ • Spatial coverage            │
│ • Uniformity                  │
│ • Runtime                     │
└───────────────┬───────────────┘
                ▼
        Registered Product
        + Correspondence Points
        + Evaluation Metrics




Core Design Philosophy
1. Classical + learned methods

The project does not assume that a single registration method will work reliably across all lunar imaging conditions.

A classical SIFT pipeline provides:

A strong and interpretable baseline
Scale and rotation robustness
A fallback when learned methods fail
A useful reference for evaluating learned approaches

A learned pipeline based on SuperPoint + LightGlue is investigated for more difficult correspondence cases.

2. Preprocessing is a first-class component

Preprocessing is treated as part of the scientific experiment rather than a fixed collection of image operations.

Different preprocessing versions can be evaluated against the same correspondence algorithms.

Examples include:

intensity normalization
CLAHE
gradient representations
destriping
geometric/geographic overlap cropping
scale normalization

This allows us to determine whether improvements come from the representation, the matching algorithm, or both.

3. Robust geometric verification

Raw feature matches are not treated as valid correspondences.

Candidate matches are geometrically verified using robust estimation methods such as:

RANSAC
MAGSAC++

Depending on the image pair, candidate transformation models include:

Similarity
Affine
Homography

The transformation model should be selected based on the actual image geometry rather than assuming that a homography is always physically appropriate.

4. Spatially distributed correspondences

A large number of matches concentrated in one small image region is not necessarily useful for registration.

The system therefore evaluates correspondence distribution using spatial grids and coverage metrics.

The goal is to obtain reliable correspondences distributed across the overlapping region while avoiding the introduction of weak matches merely to satisfy a coverage target.

5. Scientific evaluation

The system is designed to report measurable results rather than claiming registration success based only on visual appearance.

Primary metrics include:

RMSE
Inlier match count
Inlier ratio

Additional diagnostics include:

Median geometric error
P95 geometric error
Spatial coverage
Uniformity coefficient of variation
Runtime
Failure/warning status

Metrics are reported from actual experiments. No performance values are hard-coded or claimed without reproducible evidence.

Data

The project investigates lunar imagery including:

Chandrayaan-2
OHRC
TMC-2
IIRS
Reference imagery

Reference datasets may include:

LROC NAC imagery
Lunar reference mosaics
Other appropriate lunar cartographic products

Official Chandrayaan-2 data sources include the ISRO/ISS DC data systems.

Large raw datasets and generated experiment artifacts are intentionally excluded from Git version control.

Current Research Direction

The current development process follows:

Dataset investigation
        ↓
Geometric / geographic overlap validation
        ↓
Preprocessing experiments
        ↓
Reliable SIFT baseline
        ↓
SuperPoint + LightGlue baseline
        ↓
Robust geometric verification
        ↓
Spatial correspondence analysis
        ↓
Preprocessing comparison
        ↓
Refinement / sub-pixel correspondence
        ↓
Quantitative evaluation
        ↓
Integrated prototype

Synthetic transformations may be used during development and debugging because they provide known transformations and therefore controlled evaluation.

Final performance claims should be based on real lunar image pairs wherever possible.

Repository Structure
lunar-image-correspondence/
│
├── src/
│   ├── preprocessing/
│   ├── features/
│   ├── matching/
│   ├── geometry/
│   ├── refinement/
│   ├── registration/
│   └── metrics/
│
├── backend/
│   └── ...
│
├── frontend/
│   └── ...
│
├── experiments/
│   ├── deep_learning/
│   ├── multimodal/
│   ├── rift/
│   └── ...
│
├── docs/
│   ├── architecture.md
│   ├── dataset.md
│   └── experiments.md
│
├── notebooks/
│
├── tests/
│
└── README.md

The repository separates:

reusable pipeline code
backend/API infrastructure
frontend/demo interface
research experiments
documentation
tests
Development Architecture

The system is being developed as separate components with stable interfaces.

Frontend
   │
   ▼
Backend / API
   │
   ▼
Registration Orchestrator
   │
   ├──────────────► Preprocessing Pipeline
   │
   └──────────────► Registration Algorithm
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
             SIFT          SuperPoint + LightGlue
              │                     │
              └──────────┬──────────┘
                         ▼
                  Standardized Result
                         │
                         ▼
                 Evaluation / Artifacts

This allows CV, deep-learning, preprocessing and application-development work to progress independently before final integration.

Team Development Workflow

The repository uses:

main
  │
  └── stable / demo-ready code

develop
  │
  └── integration branch

feature/*
  │
  ├── feature/preprocessing
  ├── feature/cv
  ├── feature/dl
  ├── feature/backend
  ├── feature/frontend
  └── feature/evaluation

Development workflow:

Feature branch
      ↓
Implementation
      ↓
Experiment / testing
      ↓
Pull Request
      ↓
develop
      ↓
Integration testing
      ↓
main

Large datasets, model weights, virtual environments and generated experiment outputs should not be committed to the repository.

Reproducibility

Each significant experiment should record, where applicable:

Dataset / image pair
Preprocessing version
Algorithm
Algorithm version
Configuration
Transformation model
Match counts
Inlier counts
Inlier ratio
RMSE
Spatial metrics
Runtime
Failure cases
Git commit

The objective is to make improvements attributable and reproducible rather than relying on isolated successful examples.

Status
Current
Repository initialized
GitHub remote configured
main and develop branches established
Feature branches established for the six workstreams
Initial preprocessing and correspondence research code added
Geographic overlap and real lunar-pair investigation underway
SIFT baseline under development
SuperPoint + LightGlue experiments underway
Backend architecture being developed independently
Frontend/demo integration planned
Next milestones
Finalize preprocessing pipeline
Establish reproducible SIFT baseline
Establish reproducible learned-matching baseline
Integrate robust geometric verification
Implement correspondence refinement
Define final evaluation protocol
Integrate backend + algorithms
Integrate frontend
Validate on real lunar image pairs
Prepare final SIH demonstration
Scientific Integrity

This project prioritizes measurable and reproducible results.

In particular:

A visually aligned image is not automatically considered a successful registration.
A high number of raw matches is not automatically considered good correspondence.
Sub-pixel coordinates are not treated as evidence of sub-pixel accuracy.
Synthetic experiments are distinguished from real lunar-image evaluation.
Failed experiments are retained as part of the research process.
Performance numbers are reported only when supported by actual experiments.
```
