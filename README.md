# Lunar Image Correspondence

## SIH 2026 — Problem Statement SIH26166

### Multi-modal, Sun Angle and Scale Invariant Image Correspondence using Chandrayaan-2 Optical Images (OHRC, TMC-2 and IIRS)

A geometry-aware, multi-cue lunar image correspondence and registration system designed to establish reliable correspondences between Chandrayaan-2 optical imagery and reference lunar datasets acquired under different illumination, scale, viewpoint, and sensor conditions.

The system is designed around a central principle:

> Reliable lunar registration is not simply a feature-matching problem. It is a geometry, scale, illumination, modality, correspondence, and verification problem that must be solved as an integrated pipeline.

The proposed system combines planetary geometry, GSD-aware normalization, illumination-robust image representations, classical and learned correspondence methods, lunar structural cues, robust geometric verification, spatial quality control, and sub-pixel refinement to produce scientifically measurable registration results.

---

# 1. Problem Statement

Lunar images of the same geographic region can exhibit substantial differences even when they depict the same terrain.

These differences arise from:

- Different Sun azimuth and elevation
- Different illumination and shadow conditions
- Different spacecraft viewing geometry
- Different spatial resolutions / GSD
- Different image scales
- Different sensors and spectral characteristics
- Perspective and geometric distortions
- Low-texture lunar surfaces
- Strong crater and terrain relief
- Image striping and sensor-specific artifacts
- NoData / invalid pixels
- Uncertain or imperfect geographic overlap
- Very large image dimensions

The target problem is therefore:

```text
Given:

    Moving / Source Lunar Image
              +
    Fixed / Reference Lunar Image

Find:

    Reliable corresponding points
              +
    Geometrically consistent transformation
              +
    Registered image
              +
    Quantitative quality measurements

The system is intended to support correspondence between Chandrayaan-2 imagery such as:

OHRC
TMC-2
IIRS-derived products

and appropriate lunar reference datasets such as:

LROC NAC
Lunar cartographic products
Other compatible lunar reference imagery
2. Objectives

The system is designed to achieve the following objectives.

2.1 Multi-Sensor Correspondence

Support correspondence between different lunar imaging sensors instead of assuming identical image characteristics.

2.2 Illumination Robustness

Reduce sensitivity to changes in:

Sun elevation
Sun azimuth
Shadows
Surface brightness
Contrast
2.3 Scale Robustness

Handle differences in:

Ground Sampling Distance
Image resolution
Image dimensions
Image pyramid level
2.4 Geometric Robustness

Account for:

Rotation
Translation
Scale
Viewpoint changes
Local geometric distortion
Perspective effects
2.5 Reliable Correspondences

The objective is not simply to maximize raw feature matches.

The system instead seeks:

High-quality
      +
Geometrically consistent
      +
Spatially distributed
      +
Quantitatively validated
correspondences
2.6 Sub-Pixel Refinement

After obtaining geometrically reliable correspondences, refine selected points to sub-pixel precision where the image evidence supports it.

2.7 Scientific Evaluation

Every registration result should be accompanied by measurable evidence rather than being judged only by visual alignment.

3. Final System Architecture

The final system follows a single integrated pipeline:

                 ┌──────────────────────────────┐
                 │     MULTI-SENSOR INPUT       │
                 │ OHRC / TMC-2 / IIRS / LROC  │
                 │ + Metadata / PDS4           │
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │ GEOMETRY + OVERLAP           │
                 │ SPICE / DEM / Georeferencing │
                 │ Common Lunar Reference Frame │
                 │ Overlap ROI + Uncertainty    │
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │ GSD-AWARE SCALE NORMALIZATION│
                 │ Physical scale estimation    │
                 │ Common working resolution    │
                 │ Multi-scale pyramid          │
                 │ OHRC strip-aware tiling      │
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │ ILLUMINATION + MODALITY      │
                 │ NORMALIZATION                │
                 │ Robust intensity             │
                 │ CLAHE / gradient / phase    │
                 │ Destriping / shadow handling│
                 │ Spectral → structural cues  │
                 └──────────────┬───────────────┘
                                │
                                ▼
        ┌─────────────────────────────────────────────────┐
        │          MULTI-CUE CORRESPONDENCE               │
        │                                                 │
        │  Local Features      Lunar Structure            │
        │  SIFT / RootSIFT    Crater / terrain cues       │
        │  SuperPoint         Structural representation   │
        │                                                 │
        │            Area / Correlation                   │
        │            Phase correlation                    │
        │            Mutual information                   │
        └──────────────────────┬──────────────────────────┘
                               │
                               ▼
                 ┌──────────────────────────────┐
                 │ MATCH FUSION + SPATIAL QC    │
                 │ Mutual consistency           │
                 │ Confidence filtering         │
                 │ Cross-cue agreement          │
                 │ Duplicate removal            │
                 │ Spatial grid / coverage      │
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │ ROBUST GEOMETRIC VERIFICATION│
                 │ RANSAC / MAGSAC              │
                 │ Similarity / Affine          │
                 │ Projective / Homography      │
                 │ Model validation             │
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │ SUB-PIXEL REFINEMENT         │
                 │ Local patch optimization     │
                 │ Gradient / phase / MI cues   │
                 │ Residual minimization        │
                 │ Point uncertainty            │
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │ QUALITY GATE + FAILURE       │
                 │ ACCEPT / RETRY / REJECT      │
                 │ Inliers / RMSE / coverage    │
                 │ Median / P95 error           │
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │       REGISTERED PRODUCT     │
                 │ Registered Image             │
                 │ Verified Tie Points          │
                 │ Transformation + Uncertainty │
                 │ Quality Report               │
                 └──────────────────────────────┘
4. Detailed Technical Approach
4.1 Multi-Sensor Input and Metadata

The system accepts a moving/source image and a fixed/reference image together with available metadata.

Metadata is used to identify:

Sensor
Acquisition information
Spatial resolution
Image geometry
Geographic coordinates
Projection
Valid data region
Illumination information where available

The pipeline is designed so that missing or uncertain metadata does not immediately terminate processing.

Instead:

Metadata available
       ↓
Use geometric localization

Metadata uncertain
       ↓
Expand search / overlap uncertainty

Metadata insufficient
       ↓
Image-based coarse localization fallback
5. Geometry and Overlap Localization

Geometric information is used before expensive correspondence processing wherever possible.

The geometry stage establishes:

Sensor geometry
       ↓
Common lunar coordinate system
       ↓
Approximate geographic footprint
       ↓
Candidate overlap region
       ↓
Reduced search area

Planetary geometry may incorporate:

SPICE-based geometry
Sensor metadata
Lunar DEM information
Geographic footprints
Polar projections where appropriate

The overlap is treated as an uncertain region, not automatically as pixel-level ground truth.

This distinction is important because geographic projection and image correspondence have different error characteristics.

6. GSD-Aware Scale Normalization

Different sensors and reference datasets may have substantially different spatial resolutions.

Instead of directly comparing raw pixels:

Sensor A
  ↓
Estimated physical GSD

Sensor B
  ↓
Estimated physical GSD

        ↓

Common working scale

The system can construct:

GSD-normalized working images
Multi-scale pyramids
Resolution-aware image representations
Sensor-specific processing branches

For large OHRC strips, processing can additionally use:

Large image
     ↓
Geographic / geometric ROI
     ↓
Tile extraction
     ↓
Overlap-aware processing

This reduces unnecessary computation and memory consumption.

7. Illumination and Modality Normalization

The same lunar terrain may appear substantially different under different illumination conditions.

The preprocessing system therefore generates multiple complementary representations.

Intensity Representation

Preserves the overall image structure.

Robust Normalization

Uses percentile-based normalization to reduce the influence of extreme pixels.

CLAHE

Improves local contrast in low-contrast terrain.

Gradient Representation

Emphasizes terrain boundaries and structural changes rather than absolute brightness.

Multi-Scale Structural Representation

Combines image structures observed at different spatial scales.

Destriped Representation

For sensors such as OHRC where column-wise striping can interfere with feature extraction.

Phase / Structural Representation

Provides a representation less dependent on absolute intensity.

The system does not assume that one preprocessing method is universally optimal.

Instead:

Original
   │
   ├── Intensity
   ├── CLAHE
   ├── Gradient
   ├── Multi-scale
   ├── Destriped
   └── Structural / phase representation
             │
             ▼
       Correspondence engine
8. Multi-Cue Correspondence Engine

A major design principle is to avoid depending on a single feature detector or matcher.

The correspondence engine combines multiple complementary cues.

8.1 Local Feature Branch

Potential methods include:

SIFT
RootSIFT
SuperPoint

Classical features provide an interpretable and computationally accessible baseline.

Learned features provide an additional correspondence mechanism for difficult image pairs.

8.2 Learned Matching Branch

Where applicable:

SuperPoint
    ↓
Feature descriptors
    ↓
LightGlue
    ↓
Candidate correspondences

The learned branch is not assumed to be universally successful.

It is treated as one component of the multi-cue system.

8.3 Lunar Structural Branch

The lunar surface contains strong geometric structures such as:

Crater rims
Crater interiors
Ridges
Valleys
Terrain boundaries
Large-scale topographic structures

These structures can provide useful correspondence evidence even when raw intensity changes substantially.

8.4 Area / Correlation Branch

When sparse local features are insufficient, area-based techniques can provide complementary evidence.

Potential methods include:

Phase correlation
Mutual information
Local correlation
Gradient-based similarity

The goal is to combine sparse feature correspondence with broader image evidence.

9. Match Fusion

Candidate matches generated by different branches are not automatically accepted.

The system evaluates:

Local feature evidence
        +
Learned matching confidence
        +
Structural agreement
        +
Area/correlation evidence
        +
Geometric consistency

Candidate matches can then be filtered using:

Descriptor confidence
Ratio tests where applicable
Mutual / bidirectional consistency
Duplicate removal
Cross-branch agreement
Neighborhood consistency
Local geometric consistency
10. Spatial Quality Control

A high match count alone does not guarantee a useful registration.

For example:

     XXXXX
     XXXXX
     XXXXX

        mostly one region

may be less useful than:

X          X

      X

X          X

with reliable points covering the overlap.

Therefore the system evaluates:

Spatial coverage
Grid occupancy
Distribution uniformity
Regional match density
Clustered-match detection

The goal is to obtain reliable points distributed throughout the usable overlap.

The system does not artificially add weak matches merely to satisfy a coverage target.

11. Robust Geometric Verification

Raw correspondences are treated as hypotheses.

They must pass geometric verification.

Candidate models include:

Similarity

Useful when the dominant transformation is approximately:

translation + rotation + uniform scale
Affine

Useful when moderate geometric distortion exists.

Projective / Homography

Useful when the image geometry requires a projective model.

The system can evaluate multiple models rather than blindly assuming that every lunar image pair should use a homography.

Robust estimation can use:

RANSAC
MAGSAC++
Other robust estimators where appropriate

The result is:

Candidate matches
      ↓
Robust model estimation
      ↓
Inliers
      ↓
Residual analysis
      ↓
Validated transformation
12. Sub-Pixel Refinement

After robust geometric verification, high-confidence correspondences can be refined locally.

The refinement stage operates on image patches around the selected points.

Possible optimization signals include:

Gradient alignment
Phase consistency
Local correlation
Mutual information
Residual minimization

The refinement stage produces:

Initial correspondence
        ↓
Local optimization
        ↓
Sub-pixel coordinate
        +
Uncertainty / residual

Sub-pixel coordinates are not themselves treated as proof of sub-pixel accuracy.

Accuracy must be evaluated against appropriate geometric or experimental evidence.

13. Quality Gate

Every result passes through a final validation stage.

Important measurements include:

Inlier count
Inlier ratio
RMSE
Median geometric error
P95 geometric error
Spatial coverage
Distribution uniformity
Transformation consistency
Runtime
Failure / warning state

The final decision is:

                 ┌──────────────┐
                 │ Quality Gate │
                 └──────┬───────┘
                        │
              ┌─────────┼─────────┐
              ↓         ↓         ↓
           ACCEPT     RETRY     REJECT
ACCEPT

The correspondence set satisfies the configured quality requirements.

RETRY

The result is insufficient but another processing configuration may be appropriate.

Examples:

Alternate preprocessing representation
Alternate feature branch
Alternate scale
Alternate geometric model
Expanded overlap
Alternative matching strategy
REJECT

The available evidence is insufficient to establish a reliable registration.

The system reports the failure reason rather than producing an apparently valid but unreliable result.

14. Failure Recovery Strategy

The final architecture explicitly considers difficult cases.

Case 1 — Very Few Features
Few keypoints
      ↓
Alternate representation
      ↓
Gradient / structural representation
      ↓
Learned feature branch
      ↓
Area-based matching
Case 2 — Matches Concentrated in One Region
Clustered matches
      ↓
Spatial grid analysis
      ↓
Region-aware filtering
      ↓
Alternative representation / tile
      ↓
Re-estimation
Case 3 — Illumination Mismatch
Large brightness difference
      ↓
Intensity-independent representations
      ↓
Gradient / phase / structural matching
      ↓
Robust geometric verification
Case 4 — Large Scale Difference
Different GSD
      ↓
Physical scale estimation
      ↓
GSD-aware normalization
      ↓
Multi-scale correspondence
Case 5 — OHRC Striping
OHRC image
      ↓
Valid-pixel analysis
      ↓
Column-gain correction
      ↓
Structural representation
      ↓
Feature extraction
Case 6 — Large OHRC Image
Large strip
      ↓
Overlap localization
      ↓
ROI extraction
      ↓
Tile-based processing
      ↓
Local correspondence
      ↓
Global verification
Case 7 — Metadata Uncertainty
Metadata
   │
   ├── reliable → geometry-guided overlap
   │
   └── uncertain
          ↓
   uncertainty-expanded search
          ↓
   image-based fallback
Case 8 — False Correspondences
Raw matches
     ↓
Confidence filtering
     ↓
Mutual consistency
     ↓
Geometric verification
     ↓
Spatial QC
     ↓
Validated inliers
Case 9 — Visually Aligned but Scientifically Unreliable

The system does not consider visual alignment alone sufficient.

A result can be rejected when:

Inlier count is insufficient
Inlier ratio is poor
Error is excessive
Spatial coverage is inadequate
Transformation is geometrically inconsistent
Residual distribution is unstable
15. Output Products

For every successful registration, the system produces:

15.1 Registered Image

The moving image transformed into the reference coordinate system.

15.2 Verified Correspondence Points

A set of geometrically validated tie points.

Each point may contain:

Source coordinate
Reference coordinate
Match confidence
Geometric residual
Refinement status
Estimated uncertainty
15.3 Transformation Parameters

Including the selected geometric model and its parameters.

15.4 Quality Report

Containing:

Input information
Sensor information
Processing configuration
Transformation model
Raw matches
Verified inliers
Inlier ratio
RMSE
Median error
P95 error
Spatial coverage
Uniformity
Runtime
Warnings
Final decision
15.5 Machine-Readable Result

A structured JSON/CSV result can be generated for downstream processing.

16. Evaluation Metrics

The system is evaluated using quantitative measurements rather than visual appearance alone.

Primary Metrics
Inlier Count

Number of correspondences consistent with the estimated geometric model.

Inlier Ratio
Inlier Ratio =
Number of Inliers / Number of Candidate Matches
RMSE

Measures geometric registration error.

Spatial Coverage

Measures how much of the usable overlap is represented by verified correspondences.

Additional Metrics
Median geometric error
P95 geometric error
Spatial uniformity
Grid occupancy
Runtime
Failure rate
Processing memory where relevant

Synthetic transformations can additionally be used during development because the ground-truth transformation is known.

However:

Synthetic performance is not presented as equivalent to real lunar-image performance.

Final performance claims should be based on reproducible real lunar image pairs wherever possible.

17. Experimental Methodology

The development process follows a controlled progression:

Dataset investigation
        ↓
Geometry / geographic validation
        ↓
Preprocessing experiments
        ↓
Reliable SIFT baseline
        ↓
Learned matching baseline
        ↓
Robust geometric verification
        ↓
Spatial correspondence analysis
        ↓
Multi-cue fusion
        ↓
Sub-pixel refinement
        ↓
Quality gate
        ↓
Real lunar validation
        ↓
Integrated prototype

Every significant experiment should record:

Dataset / image pair
Sensor
Preprocessing configuration
Feature method
Matching method
Transformation model
Candidate match count
Inlier count
Inlier ratio
RMSE
Median error
P95 error
Spatial metrics
Runtime
Failure condition
Git commit / implementation version

This makes improvements attributable and reproducible.

18. Current Repository Implementation

The repository currently contains research and prototype components covering several stages of the final architecture.

Important existing components include:

src/
├── preprocessing/
│   ├── preprocess.py
│   ├── extract_p0_overlap.py
│   └── run_preprocessing_experiments.py
│
├── features/
│   └── sift_features.py
│
├── matching/
│   └── sift_benchmark.py
│
└── pipeline.py

Research and geometry investigation scripts include:

build_true_1m_ohrc_crop.py
diagnose_ohrc_geometry.py
official_1m_working_pair.py
official_lroc_exact_overlap.py
official_ohrc_representation_benchmark.py

Testing includes:

test_preprocessing_correspondence.py
test_preprocessing_synthetic.py

Documentation is organized under:

docs/
├── architecture.md
├── dataset.md
└── experiments.md

The repository's current implementation contains substantial preprocessing and experimental work, while the final integrated architecture requires consolidation of these components into a stable end-to-end orchestration layer.

19. Target Software Architecture

The final software is organized into modular components:

Frontend / Portal
       │
       ▼
Backend / API
       │
       ▼
Registration Orchestrator
       │
       ├── Metadata / PDS4
       │
       ├── Geometry Engine
       │
       ├── Overlap Localization
       │
       ├── Scale / GSD Engine
       │
       ├── Preprocessing
       │
       ├── Feature Extraction
       │
       ├── Matching
       │
       ├── Match Fusion
       │
       ├── Geometric Verification
       │
       ├── Sub-Pixel Refinement
       │
       ├── Quality Gate
       │
       └── Metrics / Reporting
                │
                ▼
        Registered Product

Each module should expose a stable interface so that research experiments can be replaced or improved without rewriting the entire application.

20. Technology Stack
Programming
Python
Computer Vision
OpenCV
SIFT / RootSIFT
Gradient and structural representations
Geometric verification
Deep Learning
PyTorch
SuperPoint
LightGlue
Planetary Geometry
SPICE
Lunar DEM products
Geographic projections
GDAL
Data Processing
NumPy
Raster/image processing libraries
CSV / JSON metadata
Backend
Python-based API layer
Database / Metadata
PostgreSQL where required
Deployment
Linux
Docker
CPU execution
Optional GPU acceleration
21. Deployment Architecture

The system is designed to operate in multiple environments.

                    Lunar Registration System
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
          Local PC         Docker          Server/HPC
             │                │                │
             └────────────────┼────────────────┘
                              │
                              ▼
                       Registration API
                              │
                              ▼
                    Processing Pipeline

The same core registration engine should remain independent of the presentation layer.

This allows the prototype to evolve from:

Research scripts
      ↓
Integrated Python pipeline
      ↓
Backend API
      ↓
Web demonstration
      ↓
Operational / batch processing
22. Prototype Workflow

The intended demonstration workflow is:

1. Upload source image
          ↓
2. Upload reference image
          ↓
3. Read metadata
          ↓
4. Determine sensor / scale
          ↓
5. Estimate overlap
          ↓
6. Normalize scale and modality
          ↓
7. Run correspondence engine
          ↓
8. Verify matches geometrically
          ↓
9. Refine correspondences
          ↓
10. Evaluate quality
          ↓
11. Display registered image
          ↓
12. Display verified tie points
          ↓
13. Display metrics
          ↓
14. Download results
23. Scientific Integrity

The project follows several principles to avoid misleading registration results.

Principle 1

A visually aligned image is not automatically a successful registration.

Principle 2

A large number of raw matches is not automatically good correspondence.

Principle 3

Sub-pixel coordinates are not automatically evidence of sub-pixel accuracy.

Principle 4

Synthetic experiments and real lunar experiments are reported separately.

Principle 5

Geographic overlap is not automatically pixel-level ground truth.

Principle 6

Failed experiments are useful evidence and should be retained during development.

Principle 7

Performance numbers are reported only when supported by reproducible experiments.

24. Expected Benefits

The proposed system is intended to provide:

Multi-Sensor Registration

A common framework for Chandrayaan-2 OHRC, TMC-2 and IIRS-derived imagery and compatible lunar reference datasets.

Illumination Robustness

Reduced dependence on raw intensity through structural and multi-representation processing.

Scale Robustness

GSD-aware normalization and multi-scale processing.

Geometry Awareness

Use of available planetary geometry before expensive image matching.

Reliable Tie Points

Geometrically verified and spatially distributed correspondences rather than raw feature matches.

Sub-Pixel Refinement

Improved localization of high-confidence correspondence points where image evidence supports it.

Reproducible Evaluation

Quantitative metrics and experiment configurations rather than visual-only validation.

Extensibility

A modular architecture that can incorporate additional lunar missions, sensors, representations, and matching algorithms.

25. Future Scope

The architecture is designed to support future extensions such as:

Additional Chandrayaan datasets
Additional lunar optical sensors
Improved lunar DEM integration
Advanced learned correspondence models
RIFT-style modality-invariant representations
Improved crater / terrain structural matching
More sophisticated uncertainty estimation
GPU acceleration
Large-scale batch processing
Automated dataset benchmarking
Integration with planetary mapping workflows
Extension toward future lunar missions and datasets

These are future extensions of the architecture, not claims that every component is already operational.

26. Project Development Strategy

Development is organized into the following layers:

LAYER 1
Dataset + Geometry
        ↓
LAYER 2
Preprocessing + Scale Normalization
        ↓
LAYER 3
Correspondence Algorithms
        ↓
LAYER 4
Match Fusion + Geometric Verification
        ↓
LAYER 5
Sub-Pixel Refinement
        ↓
LAYER 6
Quality Gate + Metrics
        ↓
LAYER 7
Registration Orchestrator
        ↓
LAYER 8
Backend / API
        ↓
LAYER 9
Frontend / Portal
        ↓
LAYER 10
Deployment + Demonstration

This prevents the project from becoming a collection of disconnected experimental scripts.

27. Repository Development Workflow

The repository uses:

main
  │
  └── Stable / demonstration-ready code

develop
  │
  └── Integration branch

feature/*
  │
  ├── preprocessing
  ├── geometry
  ├── correspondence
  ├── deep-learning
  ├── registration
  ├── backend
  ├── frontend
  └── evaluation

Development flow:

Feature branch
      ↓
Implementation
      ↓
Testing
      ↓
Experiment
      ↓
Pull Request
      ↓
develop
      ↓
Integration testing
      ↓
main

Large datasets, model weights, generated artifacts and virtual environments should remain outside Git version control.

28. Final System Objective

The final system aims to transform:

Different lunar images
        ↓
Different sensors
        ↓
Different scales
        ↓
Different illumination
        ↓
Different viewing conditions

into:

Common geographic context
        ↓
Reliable correspondence
        ↓
Geometrically verified tie points
        ↓
Sub-pixel refined matches
        ↓
Validated registration
        ↓
Scientifically measurable output

The central objective is therefore:

To build a robust, geometry-aware and quantitatively validated lunar image correspondence system capable of establishing reliable cross-sensor and cross-condition correspondences for Chandrayaan-2 imagery and compatible lunar reference datasets.

29. Project Status
Research / Prototype Components
Repository structure established
OHRC preprocessing investigation
OHRC destriping research
Geographic overlap investigation
GSD-aware OHRC working-image generation
Multi-representation preprocessing
SIFT baseline development
Correspondence experiments
Synthetic testing
Real lunar-pair investigation
Robust registration architecture defined
Integration Work

The next engineering stage is to consolidate the existing research components into the final modular architecture:

Existing Research Code
        ↓
Validated Modules
        ↓
Unified Interfaces
        ↓
Registration Orchestrator
        ↓
Quality Gate
        ↓
Backend API
        ↓
Portal / Demonstration
30. Final Deliverables

The completed system is intended to provide:

┌─────────────────────────────────────────────┐
│              FINAL DELIVERABLES             │
├─────────────────────────────────────────────┤
│                                             │
│  1. Lunar Image Registration Software       │
│                                             │
│  2. Registered Image Product                │
│                                             │
│  3. Verified Correspondence / Tie Points    │
│                                             │
│  4. Transformation Parameters               │
│                                             │
│  5. Sub-Pixel Refined Points                │
│                                             │
│  6. Quantitative Quality Metrics            │
│                                             │
│  7. Failure / Warning Diagnostics           │
│                                             │
│  8. Reproducible Experiment Configuration  │
│                                             │
│  9. API / Demonstration Interface           │
│                                             │
│ 10. Technical Documentation                 │
│                                             │
└─────────────────────────────────────────────┘
Project Philosophy

The system is designed to move beyond simple feature matching toward a geometry-aware, multi-cue, failure-aware, quantitatively validated lunar registration framework.

The final pipeline is:

INPUT + METADATA
       ↓
GEOMETRY + OVERLAP
       ↓
GSD-AWARE NORMALIZATION
       ↓
ILLUMINATION / MODALITY NORMALIZATION
       ↓
MULTI-CUE CORRESPONDENCE
       ↓
MATCH FUSION + SPATIAL QC
       ↓
ROBUST GEOMETRIC VERIFICATION
       ↓
SUB-PIXEL REFINEMENT
       ↓
QUALITY GATE
       ↓
REGISTERED PRODUCT

Reliable correspondence first. Registration second. Quantitative validation always.
