# Vyom Drishti

## SIH 2026 — Problem Statement SIH26166
### Multi-modal, Sun Angle and Scale Invariant Image Correspondence using Chandrayaan-2 Optical Images (OHRC, TMC-2 and IIRS)

Vyom Drishti is a geometry-aware, multi-cue lunar image correspondence and registration system for Chandrayaan-2 optical imagery and compatible lunar reference datasets.

The system is designed for the difficult case where the same lunar terrain is observed with different illumination, viewing geometry, scale/GSD, sensor characteristics, contrast, shadows, and imaging artifacts.

> **Core principle:** Reliable lunar registration is not only a feature-matching problem. It is a geometry, scale, illumination, modality, correspondence, verification, and uncertainty problem.

---

# 1. Problem Statement

Lunar image registration aligns a moving/source image with a fixed/reference image so that corresponding lunar surface locations occupy a common geometric frame.

The main challenges are:

- **Illumination variation:** different Sun azimuth/elevation, shadows, contrast, and polar lighting.
- **Viewpoint variation:** changes in camera position/orientation and local terrain relief.
- **Scale variation:** different GSD, resolution, and image dimensions.
- **Multi-modal imagery:** different spatial, spectral, and radiometric characteristics across OHRC, TMC-2, IIRS, and reference imagery.
- **Lunar terrain:** craters, ridges, valleys, low-texture regions, repetitive structures, and shadows.
- **Artifacts:** striping, NoData, invalid pixels, and large image dimensions.
- **Overlap uncertainty:** metadata can guide localization but does not automatically provide pixel-level correspondence ground truth.

---

# 2. Objectives

Vyom Drishti aims to provide:

1. Multi-sensor lunar image correspondence.
2. Robustness to illumination and shadow changes.
3. GSD-aware scale normalization.
4. Reliable and spatially distributed tie points.
5. Robust geometric transformation estimation.
6. Sub-pixel refinement of selected correspondences.
7. Quantitative validation and failure diagnostics.
8. A modular architecture suitable for research, prototype, and deployment.

---

# 3. System Architecture

```text
MULTI-SENSOR INPUT
OHRC / TMC-2 / IIRS / LROC + Metadata/PDS4
        ↓
GEOMETRY + OVERLAP
SPICE / DEM / Georeferencing → Common Lunar Frame → Overlap ROI
        ↓
GSD-AWARE SCALE NORMALIZATION
Physical Scale → Common Working Resolution → Multi-Scale / Tiling
        ↓
ILLUMINATION + MODALITY NORMALIZATION
Normalization / CLAHE / Gradient / Phase / Destriping / Structural Views
        ↓
MULTI-CUE CORRESPONDENCE
Local Features + Lunar Structure + Area/Correlation
        ↓
MATCH FUSION + SPATIAL QC
Confidence / Mutual Consistency / Cross-Cue Agreement / Coverage
        ↓
ROBUST GEOMETRIC VERIFICATION
RANSAC / MAGSAC → Similarity / Affine / Projective Models
        ↓
SUB-PIXEL REFINEMENT
Local Patch Optimization / Gradient / Phase / MI / Residuals
        ↓
QUALITY GATE + RECOVERY
ACCEPT / RETRY / REJECT
        ↓
REGISTERED PRODUCT
Image + Verified Tie Points + Transform + Metrics + Report
```

---

# 4. Technical Approach

## 4.1 Input and Metadata

The pipeline accepts moving/source and fixed/reference imagery and extracts, where available:

- Sensor and acquisition information
- GSD and image dimensions
- Geographic coordinates and projection
- Camera/observation geometry
- Valid-data and NoData information
- Illumination metadata

If metadata is uncertain, the system can move from geometry-guided localization to uncertainty-expanded search and then image-based coarse localization.

## 4.2 Geometry and Overlap

Planetary geometry is used to establish a common lunar context and reduce the correspondence search space.

Typical components include:

- SPICE-based geometry where available
- Lunar DEM information
- Geographic footprints
- Common lunar coordinates
- Projection conversion
- Uncertainty-expanded overlap regions

Geographic overlap is treated as a localization guide, not as automatic pixel-level ground truth.

## 4.3 GSD-Aware Scale Normalization

The system uses physical image resolution rather than arbitrary resizing:

```text
Sensor GSD
   ↓
Physical Scale
   ↓
Common Working Resolution
   ↓
Multi-Scale Pyramid
   ↓
Resolution-Aware Features
```

Large OHRC data can additionally use overlap-based ROI extraction and strip-aware tiling.

## 4.4 Illumination and Modality Normalization

Multiple image representations are generated so that correspondence does not depend only on absolute intensity.

Possible representations include:

- Robust intensity normalization
- CLAHE
- Gradient images
- Multi-scale structural views
- Phase-based representations
- Controlled OHRC destriping
- Valid-pixel masks
- Shadow-aware processing

## 4.5 Multi-Cue Correspondence

### Local Features

- SIFT
- RootSIFT
- SuperPoint

### Learned Matching

```text
SuperPoint → Descriptors → LightGlue → Confidence-Aware Matches
```

### Lunar Structural Cues

- Crater rims and interiors
- Ridges and valleys
- Terrain boundaries
- Large-scale morphology
- Local terrain neighborhoods

### Area / Correlation Cues

- Phase correlation
- Mutual information
- Local correlation
- Gradient similarity

The branches are complementary rather than mutually exclusive.

---

# 5. Match Fusion and Spatial Quality Control

Correspondences from different branches are filtered and fused using:

- Descriptor or matcher confidence
- Ratio and mutual-nearest-neighbor checks
- Forward/backward consistency
- Duplicate removal
- Cross-cue agreement
- Local neighborhood consistency
- Geometric consistency

A high raw match count is not sufficient. Vyom Drishti also evaluates spatial distribution using:

- Spatial coverage
- Grid occupancy
- Regional density
- Cluster detection
- Distribution uniformity

Weak matches are not added merely to improve coverage statistics.

---

# 6. Robust Geometric Verification

Candidate correspondences are tested using suitable geometric models, including:

- Similarity
- Affine
- Projective
- Homography

Robust estimation can use RANSAC or MAGSAC/MAGSAC++.

The objective is to separate geometrically consistent inliers from false matches and validate the resulting transformation.

Key measurements include:

- Inlier count
- Inlier ratio
- Residual/reprojection error
- Model stability
- Spatial distribution of inliers

---

# 7. Sub-Pixel Refinement

Selected verified points can be refined using local image patches.

```text
Verified Match
     ↓
Source + Reference Patches
     ↓
Gradient / Phase / Correlation / MI
     ↓
Local Optimization
     ↓
Sub-Pixel Coordinate
     ↓
Residual + Uncertainty
```

Sub-pixel coordinate precision is not treated as proof of sub-pixel accuracy; residual and geometric validation are still required.

---

# 8. Quality Gate and Failure Recovery

Every result passes through a final quality gate using:

- Inlier count
- Inlier ratio
- RMSE
- Median error
- P95 error
- Spatial coverage
- Spatial uniformity
- Transformation consistency
- Runtime and warnings

### Decision

**ACCEPT:** quality requirements are satisfied.

**RETRY:** try an alternate configuration, such as another representation, feature branch, working scale, tile, overlap, or geometric model.

**REJECT:** evidence remains insufficient for reliable registration; the failure reason is retained.

### Major recovery cases

| Failure | Recovery direction |
|---|---|
| Few features | CLAHE, gradient, structural, learned, correlation, multi-scale |
| Clustered matches | Spatial filtering, alternate representation, additional tiles |
| Illumination mismatch | Gradient, phase, structural, multi-cue matching |
| Scale mismatch | GSD normalization, pyramid, resolution-aware features |
| OHRC striping | Valid masks, controlled destriping, structural views |
| Large images | Overlap ROI, tiling, local matching, global verification |
| Metadata uncertainty | Geometry search → expanded search → image fallback |
| False matches | Confidence, mutual checks, cross-cue agreement, RANSAC/MAGSAC |

---

# 9. Output Products

Vyom Drishti is intended to produce:

### Registered Image

The transformed source image aligned with the reference image.

### Verified Tie Points

For each point, the system can retain source/reference coordinates, confidence, residual, refinement state, and uncertainty.

### Transformation

Selected geometric model and transformation parameters.

### Quality Report

Includes input information, configuration, raw matches, inliers, inlier ratio, RMSE, median/P95 error, spatial coverage, runtime, warnings, retries, and final decision.

### Machine-Readable Results

JSON/CSV and related image/transform products where applicable.

---

# 10. Evaluation Metrics

Primary metrics:

- **Inlier Count** — number of geometrically consistent correspondences.
- **Inlier Ratio** — inliers relative to candidate matches.
- **RMSE** — root mean square geometric residual.
- **Spatial Coverage** — extent of the overlap supported by reliable correspondences.

Additional metrics:

- Median error
- P95 error
- Grid occupancy
- Spatial uniformity
- Transformation stability
- Runtime
- Memory usage
- Failure rate
- Retry frequency

Synthetic ground truth is useful for controlled development but is kept separate from claims about real lunar performance.

---

# 11. Experimental Methodology

The development sequence is:

```text
Dataset Investigation
        ↓
Geometry Validation
        ↓
Preprocessing Experiments
        ↓
SIFT Baseline
        ↓
Learned Matching Baseline
        ↓
Geometric Verification
        ↓
Spatial Analysis
        ↓
Multi-Cue Fusion
        ↓
Sub-Pixel Refinement
        ↓
Quality Gate
        ↓
Real Lunar Validation
        ↓
Integrated Prototype
```

Each experiment should record the dataset, sensor, pair, GSD, preprocessing, feature/matcher, geometric model, candidate matches, inliers, ratio, error metrics, spatial metrics, runtime, failure reason, and code version.

---

# 12. Current Repository Implementation

The repository currently contains research and prototype components for different parts of the architecture.

### Preprocessing

- `src/preprocessing/preprocess.py`
- `src/preprocessing/extract_p0_overlap.py`
- `src/preprocessing/run_preprocessing_experiments.py`

### Features

- `src/features/sift_features.py`

### Matching

- `src/matching/sift_benchmark.py`

### Pipeline

- `src/pipeline.py`

### Research / Investigation

- `build_true_1m_ohrc_crop.py`
- `diagnose_ohrc_geometry.py`
- `official_1m_working_pair.py`
- `official_lroc_exact_overlap.py`
- `official_ohrc_representation_benchmark.py`

### Tests

- `test_preprocessing_correspondence.py`
- `test_preprocessing_synthetic.py`

The current repository is research/prototype-heavy. The next engineering stage is to consolidate validated components behind unified interfaces and connect them through the final registration orchestrator.

---

# 13. Target Software Architecture

```text
WEB PORTAL
    ↓
BACKEND / API
    ↓
REGISTRATION ORCHESTRATOR
    ↓
┌──────────────────────────────────────┐
│ Metadata / PDS4                      │
│ Geometry / SPICE / DEM               │
│ Overlap Localization                 │
│ GSD / Scale Normalization            │
│ Preprocessing                        │
│ Feature Extraction                   │
│ Matching                             │
│ Match Fusion                         │
│ Geometric Verification               │
│ Sub-Pixel Refinement                 │
│ Quality Gate                         │
│ Metrics / Reporting                  │
└──────────────────┬───────────────────┘
                   ↓
REGISTERED PRODUCTS + TIE POINTS + REPORT
```

---

# 14. Technology Stack

### Core

- Python
- NumPy
- OpenCV
- PyTorch

### Computer Vision

- SIFT / RootSIFT
- SuperPoint
- LightGlue
- Gradient and phase representations
- Correlation / mutual information

### Planetary / Geospatial

- SPICE
- Lunar DEM
- GDAL
- Geographic and projection utilities

### Data / Deployment

- PDS/PDS4
- CSV / JSON
- Python API
- PostgreSQL where required
- Linux
- Docker
- CPU with optional GPU acceleration

---

# 15. Prototype and Deployment

The prototype workflow is:

```text
Upload Source + Reference
        ↓
Read Metadata / Identify Sensor / GSD
        ↓
Geometry + Overlap
        ↓
Scale + Representation Normalization
        ↓
Correspondence + Fusion
        ↓
Geometric Verification
        ↓
Sub-Pixel Refinement
        ↓
Quality Gate
        ↓
Visualize + Download Results
```

Deployment can evolve from local execution to Docker, server/HPC processing, batch workflows, and a web/API interface.

---

# 16. Scientific Integrity

Vyom Drishti follows several safeguards:

- Visual alignment is not sufficient evidence of successful registration.
- Raw match count is not sufficient evidence of reliable correspondence.
- Spatial distribution is part of correspondence quality.
- Sub-pixel coordinates do not automatically prove sub-pixel accuracy.
- Geographic overlap is not pixel-level ground truth.
- Synthetic and real validation are kept separate.
- Failed runs and failure reasons are retained.
- Reported metrics should be reproducible from the dataset, configuration, and code version.

---

# 17. Project Development Strategy

The implementation is organized into ten layers:

1. **Dataset + Geometry** — metadata, GSD, footprints, overlap.
2. **Preprocessing + Scale** — masks, normalization, CLAHE, destriping, multi-scale views.
3. **Correspondence** — classical, learned, structural, and correlation methods.
4. **Fusion + Verification** — confidence, consistency, RANSAC/MAGSAC, model validation.
5. **Sub-Pixel Refinement** — local optimization and uncertainty.
6. **Quality + Metrics** — quantitative acceptance and failure classification.
7. **Registration Orchestrator** — unified end-to-end pipeline.
8. **Backend** — processing API/service.
9. **Frontend** — upload, visualization, metrics, download.
10. **Deployment** — local, Docker, server/HPC, batch.

---

# 18. Repository Development Workflow

Recommended branch structure:

```text
main
  ↑
develop
  ↑
feature/*   experiment/*   fix/*
```

Workflow:

Feature Branch → Implementation → Testing → Experiment → Pull Request → `develop` → Integration Testing → `main`

Large datasets, model weights, generated artifacts, and local environments should remain outside normal source control.

---

# 19. Final System Objective

```text
Different Lunar Images
        ↓
Different Sensors / Scales / Illumination / Viewpoints
        ↓
Common Geographic Context
        ↓
Reliable Multi-Cue Correspondence
        ↓
Geometrically Verified Tie Points
        ↓
Sub-Pixel Refinement
        ↓
Validated Registration
        ↓
Quantitative Quality Report
```

The central objective is to move from **"finding many image matches"** to **finding physically meaningful, spatially distributed, geometrically consistent, and quantitatively validated correspondences**.

---

# 20. Project Status and Deliverables

### Current Research / Prototype Work

- OHRC preprocessing and destriping investigation
- Geographic overlap and geometry diagnostics
- GSD-aware OHRC processing
- Multi-representation experiments
- SIFT feature extraction and benchmarking
- Synthetic preprocessing tests
- Real lunar pair investigation
- Architecture and experimental workflow definition

### Final Integration Direction

```text
Research Components
      ↓
Validated Modules
      ↓
Unified Interfaces
      ↓
End-to-End Pipeline
      ↓
Quality Gate
      ↓
Backend API
      ↓
Vyom Drishti Portal
```

### Final Deliverables

- Registration software
- Registered lunar image
- Verified tie points
- Transformation parameters
- Sub-pixel correspondence results where validated
- Quantitative quality metrics
- Failure diagnostics
- Reproducible experiment configuration
- API / demonstration portal
- Technical documentation

---

# Final Project Philosophy

**Geometry-Aware + Multi-Cue + Failure-Aware + Quantitatively Validated**

> **Reliable correspondence first. Registration second. Quantitative validation always.**
