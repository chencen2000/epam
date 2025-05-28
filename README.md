The src folder contains the core Python scripts for a synthetic dirt image data generation pipeline. This includes modules for:

- Argument Parsing (arguments.py): Handles command-line argument definition and parsing for configuration.
- Data Generation Orchestration (data_generation.py): The main script that orchestrates the data generation process, including loading images, applying dirt, and saving synthetic data and annotations.
- Configuration (data_generation_config.py): Defines default parameters for the data generation process.
- Dirt Placement Logic (dirt_placement.py): Implements the algorithms for placing dirt samples on images, including background estimation and mask creation with dual thresholds and distorted boundaries.
- Image Operations (image_operations.py): Provides fundamental image processing utilities such as loading, saving, scaling, rotation, and bounding box calculations.
- Mask Generation (masks2.py): Focuses on creating and manipulating distorted polygon masks used for dirt placement.
- General Utilities (utils.py): Contains helper functions for file handling (listing image files) and splitting images into patches with annotations.
- Visualization (visualization.py): Offers tools to visualize various stages of the image processing and mask creation.