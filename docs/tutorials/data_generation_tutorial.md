# Tutorial: Running the Data Generation Script (Work in progress)

This tutorial will guide you through the process of running the `scripts/generate_data.py` script using a configuration file to generate synthetic dirt images for the mobile device dirt detection system.

## Prerequisites

Before running the data generation script, ensure you have:

1. Python 3.7 or higher installed
2. All required dependencies installed (see `requirements.txt`)
3. Clean background images of mobile devices
4. Dirt sample images for superimposition
5. A properly configured environment

## Directory Structure Setup

First, organize your project directories:

```
epam/
├── scripts/
│   └── generate_data.py
├── config/
│   └── data_generation_config.yaml
├── data/
│   ├── clean_images/      # Place clean device images here
│   ├── dirt_samples/      # Place dirt sample images here
│   └── output/           # Generated images will be saved here
│       ├── synthetic_images/
│       └── masks/
└── src/
    └── ... (source files)
```

## Configuration File Setup

Create a configuration file `config/data_generation_config.yaml`:

```yaml
# Data Generation Configuration

# Input paths
input:
  clean_images_path: "data/clean_images"
  dirt_samples_path: "data/dirt_samples"
  
# Output paths
output:
  synthetic_images_path: "data/output/synthetic_images"
  masks_path: "data/output/masks"
  patches_path: "data/output/patches"

# Generation parameters
generation:
  # Number of synthetic images to generate
  num_images: 1000
  
  # Image downsampling factor
  downsample_factor: 2
  
  # Number of dirt instances per image
  dirt_instances:
    min: 3
    max: 10
  
  # Patch generation
  patch_size: 1024
  patch_overlap: 0  # Set to >0 for overlapping patches

# Augmentation parameters
augmentation:
  # Dirt sample augmentation
  scale_range: [0.9, 1.1]
  rotation_range: [-180, 180]  # degrees
  
  # Translation range (as fraction of image size)
  translation_range: [0.0, 0.3]
  
  # Opacity range for dirt blending
  opacity_range: [0.6, 0.9]

# Mask generation parameters
mask:
  # Threshold values for mask creation
  lower_threshold: 50
  upper_threshold: 200
  
  # Distortion parameters for realistic boundaries
  distortion_amplitude: 5
  distortion_frequency: 0.1

# Processing parameters
processing:
  # Number of parallel workers
  num_workers: 4
  
  # Random seed for reproducibility
  random_seed: 42
  
  # Verbose output
  verbose: true


```

## Running the Script

### Basic Usage

1. **Activate your virtual environment:**
   ```bash
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

2. **Run with default configuration:**
   ```bash
   python scripts/generate_data.py --config config/data_generation_config.yaml
   ```


## Example Workflow

Here's a complete example workflow:

```bash
# 1. Set up directories
mkdir -p data/{clean_images,dirt_samples,output}

# 2. Copy your images
cp /path/to/clean/images/* data/clean_images/
cp /path/to/dirt/samples/* data/dirt_samples/

# 3. Create configuration
cp config/data_generation_config.yaml config/my_experiment.yaml
# Edit my_experiment.yaml as needed

# 4. Test run with small dataset
python scripts/generate_data.py --config config/my_experiment.yaml 

# 5. Run full generation
python scripts/generate_data.py --config config/my_experiment.yaml
```

## Next Steps

After generating your synthetic dataset:

1. Use the generated patches for model training
2. Evaluate the quality of synthetic images
3. Adjust configuration parameters based on results
4. Consider creating multiple datasets with different parameters

For more information about the model training process, see the training documentation.