# EPAM - Mobile Device Dirt Detection System

## Overview

This repository contains a computer vision system for detecting and localizing different types of dirt on the front and rear surfaces of mobile devices using semantic segmentation. The system utilizes high-resolution images from grading machines and employs a synthetically generated dataset with automatic pixel annotations to pre-train the segmentation model.

## Project Structure

### `/src` - Core Source Code

The main source directory containing all Python scripts for the synthetic dirt image data generation pipeline and model training:

#### Data Generation Pipeline
- **`arguments.py`** - Command-line argument parsing and configuration handling
- **`data_generation.py`** - Main orchestration script that controls the entire data generation process, including:
  - Loading clean background images
  - Applying synthetic dirt samples
  - Saving generated images and annotations
- **`data_generation_config.py`** - Default configuration parameters for the data generation process

#### Image Processing & Manipulation
- **`image_operations.py`** - Fundamental image processing utilities:
  - Image loading and saving
  - Scaling and rotation operations
  - Bounding box calculations
- **`dirt_placement.py`** - Core algorithms for placing dirt samples on images:
  - Background estimation
  - Mask creation with dual thresholds
  - Distorted boundary generation

#### Mask Generation & Utilities
- **`masks2.py`** - Creates and manipulates distorted polygon masks for realistic dirt placement
- **`utils.py`** - General helper functions:
  - File handling and image file listing
  - Image splitting into 1024x1024 patches with annotations
  - Dataset management utilities

#### Model Training & Visualization
- **`training.py`** - Implementation of the slim U-Net model for semantic segmentation
- **`visualization.py`** - Tools for visualizing various stages of the image processing and mask creation pipeline

## Technical Details

### Dataset Generation Approach
- Original clean images are downsampled 2x by width and height
- Synthetic dirt images are created using scaling, rotation, and translation augmentations
- Each generated sample contains multiple dirt instances superimposed over clean backgrounds
- Generated samples and corresponding binary masks are split into non-overlapping 1024x1024 patches

### Model Architecture
The project uses a slim U-Net model that:
- Inherits core ideas from the original U-Net architecture (Ronneberger et al.)
- Features fewer stages and lower channel numbers for speed optimization
- Utilizes dilated convolutions (2x) at every stage to increase receptive field
- Contains significantly fewer trainable parameters compared to the original U-Net

### Current Dataset Classes
1. **Dirt pixels** - Various types of dirt and contamination
2. **Clean pixels** - Background/clean surface areas
3. **Scratches** (planned) - To be added through false positive detection on clean samples

## Future Enhancements

- Implementation of patch overlapping to increase training data volume
- Fine-tuning with hard negative samples (scratches detected as false positives)
- Integration of manually annotated permanent defects with class labels
- Expansion to multi-class segmentation including various defect types

## Usage

[To be added: Instructions for running the data generation pipeline and training the model]

## Requirements

[To be added: Python version, dependencies, and hardware requirements]

## Contributing

This project is part of the feature-enhancement branch development. Please ensure all contributions maintain compatibility with the existing pipeline structure.