Vyom Drishti
SIH 2026 — Problem Statement SIH26166
Multi-modal, Sun Angle and Scale Invariant Image Correspondence using Chandrayaan-2 Optical Images (OHRC, TMC-2 and IIRS)

Vyom Drishti is a geometry-aware, multi-cue image correspondence and registration system designed for multi-modal Chandrayaan-2 optical imagery and compatible lunar reference datasets.

The system addresses the difficulty of finding reliable correspondences between lunar images acquired under different:

Sun illumination conditions
Sun azimuth and elevation angles
Viewing geometries
Spatial resolutions and GSDs
Image scales
Sensor characteristics
Spectral responses
Contrast and radiometric conditions
Surface appearance and shadow conditions

The proposed system combines planetary geometry, GSD-aware scale normalization, illumination-robust representations, classical and learned feature matching, lunar structural cues, robust geometric verification, spatial quality control, and sub-pixel refinement into a single integrated registration pipeline.

Core principle: Reliable lunar registration is not simply a feature-matching problem. It is a geometry, scale, illumination, modality, correspondence, verification, and uncertainty problem that must be solved as an integrated pipeline.

1. Problem Statement

Image registration is the process of aligning a moving/source image with a fixed/reference image so that corresponding surface locations occupy the same geometric coordinate system.

For lunar imagery, this becomes difficult because images of the same terrain may differ significantly in acquisition conditions and sensor characteristics.

Major Challenges
1.1 Illumination Variation

The same lunar surface can appear substantially different because of:

Different Sun azimuth
Different Sun elevation
Different shadow directions
Different shadow lengths
Changing local contrast
Strongly illuminated and dark regions
Polar illumination conditions

A direct intensity comparison therefore cannot always be trusted.

1.2 Viewpoint Variation

Differences in:

Camera position
Camera orientation
Observation geometry
Local terrain relief
Perspective

can produce geometric distortions between corresponding regions.

1.3 Scale Variation

Images may have different:

Ground Sampling Distance (GSD)
Pixel resolution
Image dimensions
Acquisition scales

Therefore, the same lunar feature may occupy significantly different numbers of pixels.

1.4 Multi-Sensor / Multi-Modal Differences

The system is intended to support imagery from:

Chandrayaan-2 OHRC
Chandrayaan-2 TMC-2
Chandrayaan-2 IIRS-derived representations
LRO NAC/reference imagery
Compatible lunar cartographic/reference products

These sensors may have different spatial, spectral, radiometric, and imaging characteristics.

1.5 Lunar Surface Characteristics

The lunar surface contains:

Large craters
Small craters
Ridges
Valleys
Slopes
Mare regions
Low-texture terrain
Strong terrain boundaries
Repetitive textures
Shadowed regions

These characteristics can cause conventional local feature matching to fail or produce clustered or false correspondences.

1.6 Imaging Artifacts

The system must account for:

Striping
NoData regions
Invalid pixels
Uneven illumination
Low contrast
Large image dimensions
Sensor-specific artifacts
1.7 Metadata and Overlap Uncertainty

Geographic metadata can help localize overlap, but metadata-derived overlap should not automatically be treated as pixel-level ground truth.

The system therefore uses geometry as a guide while retaining uncertainty and image-based fallback mechanisms.

2. Objectives

The final system is designed around the following objectives.

2.1 Multi-Sensor Correspondence

Find reliable correspondences across OHRC, TMC-2, IIRS-derived imagery and compatible lunar reference imagery.

2.2 Illumination Robustness

Reduce dependence on absolute image intensity so that correspondence remains possible under different illumination and shadow conditions.

2.3 Scale Robustness

Handle large differences in image resolution and GSD through physical scale normalization and multi-scale processing.

2.4 Geometric Robustness

Handle translation, rotation, scale, affine and projective differences using robust geometric verification.

2.5 Reliable Correspondences

Prioritize high-confidence, geometrically consistent matches instead of simply maximizing raw match count.

2.6 Spatially Distributed Tie Points

Ensure that verified correspondences are distributed across the overlapping region instead of being concentrated in a single small area.

2.7 Sub-Pixel Refinement

Refine selected correspondences beyond integer-pixel coordinates using local image information.

2.8 Scientific Evaluation

Evaluate the registration using measurable metrics such as:

Inlier count
Inlier ratio
RMSE
Median error
P95 error
Spatial coverage
Match distribution
Runtime
Failure rate
3. Final System Architecture

The complete system follows one continuous pipeline:

MULTI-SENSOR INPUT

OHRC / TMC-2 / IIRS / LROC + Metadata/PDS4

↓

GEOMETRY + OVERLAP

SPICE / DEM / Georeferencing → Common Lunar Frame → Footprints → Overlap ROI + Uncertainty

↓

GSD-AWARE SCALE NORMALIZATION

Physical Scale → Common Working Resolution → Multi-Scale Pyramid → OHRC Strip-Aware Tiling

↓

ILLUMINATION + MODALITY NORMALIZATION

Robust Intensity / CLAHE / Gradient / Phase / Destriping / Shadow Handling / Structural Representation

↓

MULTI-CUE CORRESPONDENCE ENGINE

Local Features:
SIFT / RootSIFT / SuperPoint

Lunar Structure:
Craters / Rims / Ridges / Valleys / Terrain Boundaries

Area / Correlation:
Phase Correlation / Mutual Information / Local Correlation

↓

MATCH FUSION + SPATIAL QC

Mutual Consistency / Confidence / Cross-Cue Agreement / Duplicate Removal / Grid Coverage / Uniformity

↓

ROBUST GEOMETRIC VERIFICATION

RANSAC / MAGSAC → Similarity / Affine / Projective / Homography → Model Validation

↓

SUB-PIXEL REFINEMENT

Local Patch Optimization / Gradient / Phase / MI / Residual Minimization / Point Uncertainty

↓

QUALITY GATE + RECOVERY

ACCEPT / RETRY / REJECT

Inliers / Ratio / RMSE / Coverage / Median / P95

↓

REGISTERED PRODUCT

Registered Image + Verified Tie Points + Transformation + Uncertainty + Quantitative Quality Report

4. Detailed Technical Approach
4.1 Multi-Sensor Input and Metadata

The system accepts a moving/source image and a fixed/reference image.

Metadata may include:

Sensor
Acquisition time
Image dimensions
GSD
Camera geometry
Geographic coordinates
Projection
Observation geometry
Valid data region
Illumination information
Product-specific metadata

Where available, PDS4 metadata and planetary geometry information are used.

The input stage also performs:

Image validation
NoData detection
Valid-pixel mask creation
Sensor identification
Resolution estimation
Metadata consistency checks
Metadata Uncertainty Fallback

If metadata is incomplete or uncertain, the system can progressively use:

Geometry-guided localization

↓

Uncertainty-expanded overlap search

↓

Image-based coarse localization

This prevents the complete pipeline from depending on perfect metadata.

5. Geometry and Overlap Localization

Geometry is used before expensive correspondence processing.

The purpose is to determine where the two images are expected to overlap and to establish a common lunar spatial context.

Geometry Components
Sensor geometry
Geographic footprints
SPICE-based geometry where available
Lunar reference frame
Lunar DEM
Geographic coordinate transformations
Projection handling
Uncertainty expansion
Overlap Process

Source Footprint

Reference Footprint

↓

Common Lunar Coordinate System

↓

Intersection / Candidate Region

↓

Uncertainty Expansion

↓

Overlap ROI

The overlap region reduces the search space and allows the correspondence algorithms to operate on physically meaningful regions.

Important Scientific Constraint

Geographic overlap is used as a localization guide.

It is not automatically treated as pixel-level correspondence ground truth.

Pixel-level correspondence must still be established through image evidence and geometric verification.

6. GSD-Aware Scale Normalization

Scale normalization is based on physical image resolution rather than arbitrary image resizing.

If two sensors have different GSDs, the same lunar surface structure can occupy very different pixel areas.

The system therefore estimates the physical scale and creates a common working resolution.

Processing Strategy

Sensor GSD

↓

Physical Scale Estimation

↓

Common Working Resolution

↓

Multi-Scale Pyramid

↓

Resolution-Aware Feature Extraction

Additional Strategies
Multi-scale image pyramids
Sensor-specific scale branches
Resolution-aware feature extraction
Overlap ROI extraction
OHRC strip-aware tiling
Large-image memory management

This allows the correspondence engine to search across realistic feature scales instead of forcing unrelated pixel scales to match directly.

7. Illumination and Modality Normalization

Absolute intensity is not assumed to remain constant across lunar acquisitions.

The system therefore creates multiple representations of the same image.

Representation Branches

Original / Normalized Intensity

↓

CLAHE

↓

Gradient

↓

Multi-Scale Structural

↓

Phase-Based

↓

Destriped / Valid-Pixel Representation

Processing Components
Robust percentile normalization
Grayscale conversion where appropriate
CLAHE
Gradient representation
Multi-scale structural representation
Phase/structural representation
OHRC destriping
Shadow-aware processing
Valid-pixel masking

For IIRS-derived imagery, the system can emphasize structural information rather than relying only on raw spectral intensity.

8. Multi-Cue Correspondence Engine

Instead of depending on one feature matcher, the system uses complementary correspondence cues.

8.1 Local Feature Branch

Potential methods include:

SIFT
RootSIFT
SuperPoint

Local features provide keypoints and descriptors around identifiable image structures.

They are useful for:

Crater boundaries
Terrain edges
Local texture
Small-scale surface structures
8.2 Learned Matching Branch

Where suitable, the system can use:

SuperPoint

↓

Feature Descriptors

↓

LightGlue

↓

Confidence-Aware Correspondences

The learned branch is treated as one correspondence source rather than the only source.

8.3 Lunar Structural Branch

Lunar terrain contains structures that can remain meaningful even when intensity changes.

The structural branch can exploit:

Crater rims
Crater interiors
Ridges
Valleys
Slopes
Terrain boundaries
Large-scale morphological structures
Local terrain neighborhoods

This provides an additional cue when conventional texture-based matching becomes unreliable.

8.4 Area / Correlation Branch

Area-based techniques provide complementary evidence.

Potential methods include:

Phase correlation
Mutual information
Local correlation
Gradient similarity
Patch-based similarity

These methods are especially useful when local keypoint density is low.

9. Match Fusion

Correspondences from different branches are not blindly combined.

The system evaluates multiple evidence sources.

Fusion Evidence

Local Feature Evidence

Learned Matching Confidence

Lunar Structural Agreement

Area / Correlation Evidence

Geometric Consistency

↓

Final Candidate Correspondences

Match Filtering

Potential filters include:

Descriptor confidence
Ratio testing
Mutual nearest-neighbor consistency
Forward/backward consistency
Duplicate removal
Cross-branch agreement
Local neighborhood consistency
Geometric consistency

The objective is to reduce false correspondences before final transformation estimation.

10. Spatial Quality Control

A large number of matches does not automatically mean successful registration.

For example, 500 matches concentrated inside one small crater may be less useful for estimating a transformation over a large image than fewer matches distributed across the overlap.

The system therefore evaluates:

Spatial coverage
Grid occupancy
Regional density
Distribution uniformity
Clustered-match detection
Coverage of important overlap regions
Spatial QC Concept

Candidate Matches

↓

Divide Overlap into Spatial Grid

↓

Count Matches per Region

↓

Detect Clusters / Empty Regions

↓

Retain Reliable Spatial Distribution

The system must not artificially create weak matches simply to satisfy a coverage requirement.

11. Robust Geometric Verification

Candidate correspondences are tested against geometric models.

Potential models include:

Similarity transformation
Affine transformation
Projective transformation
Homography

The actual model is selected according to the observed geometry and validation results.

Robust Estimation

Potential methods include:

RANSAC
MAGSAC / MAGSAC++

The geometric verification stage separates:

Raw Correspondences

↓

Geometric Hypotheses

↓

Robust Estimation

↓

Inliers / Outliers

↓

Validated Transformation

Important measurements include:

Inlier count
Inlier ratio
Reprojection/residual error
Model stability
Spatial distribution of inliers
12. Sub-Pixel Refinement

After robust geometric verification, selected correspondences can be refined locally.

Refinement Strategy

Verified Integer-Pixel Match

↓

Local Source Patch
+
Local Reference Patch

↓

Gradient / Phase / Correlation / MI

↓

Local Optimization

↓

Sub-Pixel Coordinates

↓

Residual + Uncertainty

Possible refinement objectives include:

Gradient alignment
Phase consistency
Local correlation
Mutual information
Residual minimization

Each refined point can retain:

Source coordinate
Reference coordinate
Refined coordinate
Residual
Confidence
Uncertainty estimate
Refinement status
Scientific Constraint

Sub-pixel coordinate precision does not automatically prove sub-pixel accuracy.

The final quality must still be supported by independent residual and geometric validation.

13. Quality Gate

The system does not simply output a result after obtaining matches.

Every registration passes through a final quality gate.

Quality Measurements
Inlier Count
Inlier Ratio
RMSE
Median Error
P95 Error
Spatial Coverage
Spatial Uniformity
Transformation Consistency
Runtime
Warnings
Decision

QUALITY GATE

↓

ACCEPT / RETRY / REJECT

ACCEPT

The result satisfies the configured quality requirements.

RETRY

The result may be recoverable using an alternate configuration.

Possible retries:

Alternate preprocessing
Alternate feature branch
Different working scale
Different tile
Expanded overlap
Alternate geometric model
Structural representation
Learned matcher
REJECT

The evidence is insufficient for a scientifically reliable registration.

The failure reason is retained.

14. Failure Recovery Strategy

The pipeline is explicitly designed to handle failure cases.

14.1 Few Features
Problem

Low texture / low contrast

↓

Few keypoints

Recovery
Alternate normalization
CLAHE
Gradient representation
Structural representation
Learned features
Area/correlation methods
Multi-scale processing
14.2 Clustered Matches
Problem

Many matches

↓

All concentrated in one region

Recovery
Spatial grid analysis
Regional filtering
Alternate representation
Additional tiles
Re-estimation using distributed points
14.3 Illumination Mismatch
Problem

Different Sun angle

↓

Different brightness/shadows

Recovery
Gradient representation
Phase representation
Structural features
CLAHE
Illumination-robust normalization
Multi-cue matching
14.4 Scale Difference
Problem

Different GSD

↓

Same terrain structure has different pixel size

Recovery
GSD-aware normalization
Common working scale
Multi-scale pyramid
Scale-aware feature extraction
14.5 OHRC Striping
Problem

Column/strip artifacts

↓

False local structures

Recovery
Valid-pixel masks
Controlled column-gain correction
Destriping
Structural representations
Overlap-aware processing
14.6 Large OHRC Images
Problem

Very large image

↓

High memory / computation cost

Recovery
Geographic overlap ROI
Tiling
Local feature extraction
Tile-level correspondence
Global geometric verification
14.7 Metadata Uncertainty
Problem

Uncertain geographic localization

Recovery

Geometry-guided search

↓

Uncertainty-expanded search

↓

Image-based coarse localization

14.8 False Correspondences

Recovery:

Confidence filtering
Ratio tests
Mutual consistency
Cross-branch agreement
RANSAC/MAGSAC
Spatial quality control
Residual validation
14.9 Visually Aligned but Scientifically Unreliable

A result is not accepted merely because the overlay looks visually reasonable.

The system can reject results with:

Insufficient inliers
Low inlier ratio
High RMSE
Poor spatial coverage
Large residuals
Unstable transformation
Poor refinement consistency
15. Output Products

The final system produces several scientifically useful outputs.

15.1 Registered Image

The moving/source image transformed into the reference coordinate system.

15.2 Verified Correspondence Points

Each correspondence can contain:

Source coordinates
Reference coordinates
Confidence
Match type
Residual
Refinement status
Uncertainty
15.3 Transformation Parameters

The system records:

Selected geometric model
Transformation matrix/parameters
Model configuration
Inlier set
Validation information
15.4 Quality Report

The report contains:

Input images
Sensor information
Acquisition information
GSD
Preprocessing configuration
Feature method
Matching method
Geometric model
Raw matches
Verified inliers
Inlier ratio
RMSE
Median error
P95 error
Spatial coverage
Spatial uniformity
Runtime
Warnings
Retry history
Final decision
15.5 Machine-Readable Output

Where applicable:

JSON
CSV
Image products
Transformation parameters
Quality metrics
16. Evaluation Metrics

The system evaluates both correspondence quality and registration quality.

Primary Metrics
Inlier Count

Number of correspondences consistent with the estimated transformation.

Inlier Ratio

Inlier Ratio = Inliers / Candidate Matches

RMSE

Measures the root mean square geometric residual of verified correspondences.

Spatial Coverage

Measures how much of the overlap region contains reliable correspondences.

Additional Metrics
Median error
P95 error
Grid occupancy
Spatial uniformity
Transformation stability
Runtime
Memory usage
Failure rate
Retry frequency

Synthetic datasets may be used for development and controlled testing.

Synthetic ground truth must not automatically be treated as evidence of real lunar performance.

17. Experimental Methodology

The system is developed and evaluated incrementally.

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

Robust Geometric Verification

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

Each experiment should record:

Dataset
Sensor
Image pair
GSD
Preprocessing configuration
Feature method
Matcher
Geometric model
Candidate matches
Inliers
Inlier ratio
RMSE
Median error
P95 error
Spatial metrics
Runtime
Failure reason
Git commit/version

This ensures that improvements are measurable and reproducible.

18. Current Repository Implementation

The repository currently contains research and prototype components supporting different parts of the architecture.

Preprocessing

src/preprocessing/preprocess.py

src/preprocessing/extract_p0_overlap.py

src/preprocessing/run_preprocessing_experiments.py

These components support image loading, preprocessing, valid-pixel handling, normalization, OHRC-specific processing, overlap extraction and experimental evaluation.

Feature Extraction

src/features/sift_features.py

Contains the current SIFT-oriented feature extraction implementation.

Matching

src/matching/sift_benchmark.py

Contains SIFT correspondence benchmarking and experimental matching logic.

Pipeline

src/pipeline.py

Defines the intended main registration pipeline structure:

Preprocessing

↓

Feature Extraction

↓

Feature Matching

↓

Geometric Verification

↓

Transformation Estimation

↓

Image Registration

↓

Refinement

↓

Evaluation

Research / Investigation Scripts

build_true_1m_ohrc_crop.py

diagnose_ohrc_geometry.py

official_1m_working_pair.py

official_lroc_exact_overlap.py

official_ohrc_representation_benchmark.py

These scripts support:

OHRC geometry investigation
Geographic overlap analysis
GSD-aware processing
Representation experiments
Real lunar pair investigation
Registration benchmarking
Tests

test_preprocessing_correspondence.py

test_preprocessing_synthetic.py

These provide experimental and synthetic validation of preprocessing and correspondence behavior.

Current Implementation Status

The repository contains substantial research and experimental work.

The complete architecture is still being consolidated from individual research components into a unified production-style pipeline.

The intended progression is:

Research Scripts

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

Portal

19. Target Software Architecture

The final software architecture is designed as a modular system.

WEB PORTAL

↓

BACKEND / API

↓

REGISTRATION ORCHESTRATOR

↓

REGISTRATION ENGINE

The Registration Engine contains:

Metadata / PDS4
Geometry / SPICE / DEM
Overlap Localization
GSD / Scale Normalization
Preprocessing
Feature Extraction
Matching
Match Fusion
Geometric Verification
Sub-Pixel Refinement
Quality Gate
Metrics / Reporting

↓

REGISTERED PRODUCTS

TIE POINTS

METRICS

REPORT

20. Technology Stack
Core Programming
Python
NumPy
OpenCV
PyTorch
Classical Computer Vision
SIFT
RootSIFT
Gradient-based representations
Phase-based representations
Correlation methods
Mutual information
Learned Computer Vision
SuperPoint
LightGlue
Planetary Geometry
SPICE
Lunar DEM
Geographic coordinate transformations
Projection utilities
Geospatial Processing
GDAL
Raster processing
Geographic overlap analysis
Data Formats
PDS/PDS4 metadata
TIFF/GeoTIFF where applicable
CSV
JSON
Backend / Deployment
Python API
PostgreSQL where required
Linux
Docker
CPU execution
Optional GPU acceleration
21. Deployment Architecture

The system is designed to support multiple deployment modes.

LOCAL PC

CPU / Optional GPU

↓

DOCKER

Containerized Pipeline

↓

SERVER / HPC

Batch Processing

↓

REGISTRATION API

The development path is:

Research Code

↓

Integrated Python Pipeline

↓

Backend API

↓

Web Demonstration

↓

Operational / Batch Processing

22. Prototype Workflow

The demonstration prototype follows:

Upload Source Image
Upload Reference Image
Read Metadata
Identify Sensor / Scale
Estimate Geometry / Overlap
Normalize Resolution
Generate Representations
Extract Correspondences
Fuse and Filter Matches
Robust Geometric Verification
Sub-Pixel Refinement
Quality Evaluation
Display Results
Download Products

The portal can display:

Source image
Reference image
Registered image
Match points
Inliers/outliers
Transformation
Quality metrics
Processing status
Failure/warning information
23. Scientific Integrity

The project follows several principles to prevent misleading registration results.

Visual alignment is not sufficient

A visually convincing overlay can still contain incorrect correspondences.

Raw match count is not sufficient

A high number of matches does not guarantee a reliable transformation.

Spatial distribution matters

Correspondences should cover the useful overlap region.

Sub-pixel coordinates are not automatically sub-pixel accuracy

Refinement must be evaluated through residuals and uncertainty.

Synthetic and real evaluation are separated

Synthetic ground truth is useful for controlled development but should not be presented as real lunar validation.

Geographic overlap is not pixel-level ground truth

Geographic localization helps identify candidate overlap but does not prove pixel-to-pixel correspondence.

Failures are retained

Failed registration cases should be recorded rather than hidden.

Metrics must be reproducible

Reported values should be traceable to:

Dataset
Configuration
Code version
Processing parameters
Experimental run
24. Expected Benefits

The proposed architecture provides a foundation for:

Multi-Sensor Registration

Correspondence across different lunar sensors and reference datasets.

Illumination Robustness

Reduced dependence on absolute intensity under changing Sun conditions.

Scale Robustness

Physical GSD-aware normalization and multi-scale correspondence.

Geometry Awareness

Use of planetary geometry and geographic overlap before correspondence search.

Reliable Tie Points

Confidence-filtered and geometrically verified correspondence points.

Sub-Pixel Refinement

Higher coordinate precision for selected reliable correspondences.

Reproducible Evaluation

Quantitative metrics and machine-readable reports.

Extensibility

The architecture allows additional sensors, matching methods and planetary datasets to be integrated without redesigning the entire pipeline.

25. Future Scope

The architecture can be extended to support:

Additional Chandrayaan datasets
Additional lunar sensors
Improved lunar DEM integration
Advanced learned feature extractors
Advanced learned matchers
RIFT-style illumination-invariant representations
Crater-based structural matching
Terrain-neighborhood graph matching
Better uncertainty estimation
GPU acceleration
Large-scale batch processing
Automated benchmark generation
Planetary mapping workflows
Additional future lunar missions

These are future extensions and should not be interpreted as current implemented capabilities unless explicitly integrated and validated.

26. Project Development Strategy

The complete implementation is divided into ten layers.

Layer 1 — Dataset + Geometry
Dataset acquisition
Metadata parsing
Sensor identification
GSD extraction
Footprint analysis
Geometry validation
Overlap localization
Layer 2 — Preprocessing + Scale
Valid-pixel masks
Normalization
CLAHE
Destriping
Gradient representation
Structural representation
GSD-aware resampling
Multi-scale processing
Layer 3 — Correspondence
SIFT/RootSIFT
SuperPoint
LightGlue
Structural matching
Area/correlation matching
Layer 4 — Fusion + Geometric Verification
Match confidence
Mutual consistency
Cross-cue agreement
RANSAC/MAGSAC
Model validation
Inlier filtering
Layer 5 — Sub-Pixel Refinement
Local patch extraction
Gradient refinement
Phase refinement
Correlation refinement
Residual minimization
Uncertainty estimation
Layer 6 — Quality + Metrics
Inlier count
Inlier ratio
RMSE
Median error
P95 error
Spatial coverage
Uniformity
Runtime
Failure classification
Layer 7 — Registration Orchestrator

Integrate all validated modules into a single pipeline.

Layer 8 — Backend

Create the registration API and processing service.

Layer 9 — Frontend

Create the portal for:

Upload
Processing
Visualization
Metrics
Download
Layer 10 — Deployment

Package the system for:

Local execution
Docker
Server
HPC
Batch processing
27. Repository Development Workflow

The repository should maintain a controlled development workflow.

main

↓

develop

↓

feature/*

experiment/*

fix/*

Recommended workflow:

Feature Branch

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

Integration Testing

↓

main

Large datasets, model weights, generated artifacts and local environments should not be committed to the repository.

Use configuration files and documented download procedures where required.

28. Final System Objective

The final objective can be summarized as:

Different Lunar Images

↓

Different Sensors

↓

Different Scales / GSD

↓

Different Illumination

↓

Different Viewing Geometry

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

The system is therefore designed to transform the problem from:

"Find as many image matches as possible"

into:

"Find physically meaningful, spatially distributed, geometrically consistent and quantitatively validated correspondences."

29. Project Status

The current repository contains research and prototype components covering several parts of the proposed architecture.

Existing Work Includes
Repository structure
OHRC preprocessing
Valid-pixel handling
OHRC destriping investigation
Geographic overlap investigation
GSD-aware OHRC working-image generation
Multi-representation preprocessing
SIFT feature extraction
SIFT correspondence benchmarking
Synthetic preprocessing testing
Real lunar image-pair investigation
Geometry diagnostics
Experimental registration scripts
Architecture definition
Integration Work

The remaining engineering direction is:

Existing Research Code

↓

Validated Components

↓

Unified Interfaces

↓

End-to-End Registration Pipeline

↓

Match Fusion

↓

Geometric Verification

↓

Sub-Pixel Refinement

↓

Quality Gate

↓

Backend API

↓

Portal

The repository documentation should distinguish clearly between:

Currently implemented components
Experimentally validated components
Planned integration
Future research extensions
30. Final Deliverables

The final system is intended to provide:

Software

A modular lunar image correspondence and registration pipeline.

Registered Image

The transformed source image aligned with the reference image.

Verified Tie Points

Reliable source/reference correspondence points.

Transformation Parameters

Estimated geometric transformation and model information.

Sub-Pixel Points

Refined correspondence coordinates where refinement is successful.

Quantitative Metrics
Inlier count
Inlier ratio
RMSE
Median error
P95 error
Spatial coverage
Spatial uniformity
Runtime
Failure Diagnostics

Clear explanation of:

Insufficient features
Poor overlap
Clustered correspondences
Illumination mismatch
Scale mismatch
Metadata uncertainty
Geometric inconsistency
High residuals
Reproducible Experiment Configuration

Configuration and metadata required to reproduce experimental results.

API / Demonstration Portal

A user-facing workflow for uploading images, processing registration and visualizing results.

Technical Documentation

Architecture, methodology, experiments, limitations and reproducibility information.

Final Project Philosophy

The project follows three core principles:

GEOMETRY-AWARE

MULTI-CUE

FAILURE-AWARE

↓

QUANTITATIVELY VALIDATED REGISTRATION

The final pipeline is:

Geometry

↓

Scale

↓

Illumination / Modality

↓

Multi-Cue Correspondence

↓

Match Fusion

↓

Spatial Quality Control

↓

Robust Geometry

↓

Sub-Pixel Refinement

↓

Quality Gate

↓

Registered Product

Reliable correspondence first. Registration second. Quantitative validation always.
