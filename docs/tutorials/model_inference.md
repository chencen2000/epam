# Complete Tutorial for inf.py - Multi-Class Dirt Detection Inference

## Overview

The `inf.py` script is a comprehensive inference system for multi-class dirt and scratch detection on mobile phone screens. It supports various inference modes including single image prediction, batch processing, and full phone screen analysis with patch-based processing.

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Configuration](#configuration)
3. [Usage Modes](#usage-modes)
4. [Command Line Examples](#command-line-examples)
5. [Configuration File Structure](#configuration-file-structure)
6. [Output Structure](#output-structure)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Features](#advanced-features)

## System Architecture

The inference system consists of several key components:

- **SingleImagePredictor**: Handles individual image predictions
- **FullScreenPredictor**: Manages patch-based full phone screen analysis
- **BatchPredictor**: Processes multiple images in batch mode
- **RegionAnalyzer**: Analyzes detected regions and calculates statistics
- **ImageOperations**: Handles image preprocessing and operations

### Multi-Class Support

The system supports 3 classes:
- **Class 0**: Background (clean areas)
- **Class 1**: Dirt (particles, dust, fingerprints)
- **Class 2**: Scratches (physical damage, hairline cracks)

## Configuration

### Default Configuration File
The script uses YAML configuration files located in `config/inference/`. The default configuration is:
```
config/inference/full_phone_batch_predictor.yaml
```

### Key Configuration Sections

#### Model Configuration
```yaml
model:
  path: "path/to/your/model.pth"  # Path to trained model
  device: "auto"                  # auto, cuda, cpu
  threshold: 0.5                  # Confidence threshold
```

#### Input Configuration
```yaml
input:
  path: "path/to/input"           # File or directory path
  image_extensions: ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
  max_samples: null               # Limit number of images (null = no limit)
```

#### Processing Configuration
```yaml
processing:
  batch_mode: true                # Enable batch processing
  new_images: true                # Process new images vs ground truth comparison
  full_phone:
    enabled: true                 # Enable full phone processing
    patch_size: 1024             # Size of patches for processing
    overlap: 0.2                 # Overlap between patches (0-1)
    min_dirt_threshold: 0.01     # Minimum dirt percentage to consider
```

## Usage Modes

### Mode 1: Single Image Prediction (New Images)
Process a single new image without ground truth.

**Configuration:**
```yaml
processing:
  batch_mode: false
  new_images: true
  full_phone:
    enabled: false
```

**Use Case:** Quick analysis of individual phone screen images.

### Mode 2: Batch Processing (New Images)
Process multiple new images in a directory.

**Configuration:**
```yaml
processing:
  batch_mode: true
  new_images: true
  full_phone:
    enabled: false
```

**Use Case:** Bulk processing of phone screen images for quality control.

### Mode 3: Full Phone Analysis (Single)
Analyze a complete phone screen using patch-based processing.

**Configuration:**
```yaml
processing:
  batch_mode: false
  new_images: true
  full_phone:
    enabled: true
    patch_size: 1024
    overlap: 0.2
```

**Use Case:** Detailed analysis of high-resolution phone images.

### Mode 4: Full Phone Batch Analysis
Process multiple full phone images with patch-based analysis.

**Configuration:**
```yaml
processing:
  batch_mode: true
  new_images: true
  full_phone:
    enabled: true
```

**Use Case:** Production-level quality assessment of multiple devices.

### Mode 5: Ground Truth Comparison
Compare predictions against ground truth masks.

**Configuration:**
```yaml
processing:
  batch_mode: false
  new_images: false
```

**Use Case:** Model validation and performance evaluation.

## Command Line Examples

### Basic Usage
```bash
# Use default configuration
python inf.py

# Use custom configuration
python inf.py --config config/inference/my_config.yaml
```

### Example Configurations

#### Single Image Analysis
```bash
# Create config file: single_image_config.yaml
python inf.py --config config/inference/single_image_config.yaml
```

#### Batch Processing
```bash
# Process all images in a directory
python inf.py --config config/inference/batch_config.yaml
```

#### Full Phone Analysis
```bash
# Analyze complete phone screens
python inf.py --config config/inference/full_phone_config.yaml
```

## Configuration File Structure

### Complete Example Configuration
```yaml
# config/inference/example_config.yaml
model:
  path: "models/best_model.pth"
  device: "auto"
  threshold: 0.5

input:
  path: "data/test_images/"
  image_extensions: ['.jpg', '.png']
  max_samples: 100

output:
  dir: "./output/inference_results"

processing:
  batch_mode: true
  new_images: true
  full_phone:
    enabled: true
    patch_size: 1024
    overlap: 0.2
    min_dirt_threshold: 0.01

logging:
  file: "logs/inference.log"
```

### Configuration Parameters Explained

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model.path` | string | Path to trained model file | Required |
| `model.device` | string | Device for inference (auto/cuda/cpu) | "auto" |
| `model.threshold` | float | Confidence threshold for predictions | 0.5 |
| `input.path` | string | Input file or directory path | Required |
| `input.max_samples` | int/null | Maximum images to process | null |
| `processing.batch_mode` | bool | Enable batch processing | false |
| `processing.new_images` | bool | Process new images vs comparison | true |
| `full_phone.enabled` | bool | Enable patch-based processing | false |
| `full_phone.patch_size` | int | Size of patches | 1024 |
| `full_phone.overlap` | float | Overlap between patches | 0.2 |

## Output Structure

### Single Image Output
```
output/
├── image_name/
│   ├── image_name_multiclass_detection.png     # Visualization
│   ├── image_name_results.json                 # Detailed results
│   ├── image_name_class_prediction.png         # Class prediction map
│   ├── image_name_background_mask.png          # Background mask
│   ├── image_name_dirt_mask.png               # Dirt mask
│   ├── image_name_scratches_mask.png          # Scratches mask
│   ├── image_name_background_probability.png   # Background probabilities
│   ├── image_name_dirt_probability.png        # Dirt probabilities
│   └── image_name_scratches_probability.png   # Scratch probabilities
```

### Full Phone Output
```
output/
├── image_name/
│   ├── image_name_full_phone_multiclass_analysis.png  # Complete analysis
│   ├── image_name_full_analysis.json                  # Comprehensive results
│   ├── image_name_screen_class_prediction.png         # Screen class map
│   ├── image_name_screen_background_mask.png          # Screen masks
│   ├── image_name_screen_dirt_mask.png
│   ├── image_name_screen_scratches_mask.png
│   └── image_name_full_phone_multiclass_overlay.png   # Full image overlay
```

### Batch Output
```
output/
├── batch_summary.json              # Batch processing summary
├── image1/
│   └── [individual results]
├── image2/
│   └── [individual results]
└── ...
```

## Result File Structure

### JSON Results Format
```json
{
  "image_info": {
    "filename": "phone_image.jpg",
    "original_size": [1920, 1080],
    "total_pixels": 2073600
  },
  "prediction": {
    "class_statistics": {
      "background_percentage": 85.2,
      "dirt_percentage": 12.8,
      "scratches_percentage": 2.0,
      "background_pixels": 1766668,
      "dirt_pixels": 265420,
      "scratches_pixels": 41512
    },
    "inference_time": 0.234,
    "confidence_threshold": 0.5,
    "num_classes": 3
  },
  "region_analysis": {
    "dirt": {
      "num_regions": 15,
      "largest_region_area": 1250,
      "average_region_area": 234.5
    },
    "scratches": {
      "num_regions": 3,
      "largest_region_area": 890,
      "average_region_area": 456.7
    }
  },
  "model_info": {
    "architecture": "lightweight",
    "device": "cuda:0",
    "num_classes": 3,
    "class_names": ["background", "dirt", "scratches"]
  }
}
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Model Loading Errors
```bash
Error: Model file not found: models/best_model.pth
```
**Solution:** Verify the model path in your configuration file.

#### 2. CUDA Out of Memory
```bash
RuntimeError: CUDA out of memory
```
**Solutions:**
- Reduce batch size in configuration
- Use smaller patch sizes for full phone processing
- Set device to "cpu" in configuration

#### 3. Input Path Not Found
```bash
Error: Input path not specified in configuration
```
**Solution:** Ensure `input.path` is correctly set in your YAML config.

#### 4. Invalid Image Format
```bash
Error: Unsupported image format
```
**Solution:** Check that your images are in supported formats (jpg, png, bmp, tiff, tif).

### Debug Mode
Enable debug logging by setting environment variable:
```bash
export DEBUG=true
python inf.py --config your_config.yaml
```

### Memory Issues
For large images or batch processing:
1. Reduce patch size: `patch_size: 512`
2. Reduce overlap: `overlap: 0.1`
3. Limit batch size: `max_samples: 10`
4. Use CPU: `device: "cpu"`

## Advanced Features

### 1. Custom Patch Processing
For very large images, adjust patch parameters:
```yaml
full_phone:
  patch_size: 512      # Smaller patches for memory efficiency
  overlap: 0.3         # Higher overlap for better coverage
  min_dirt_threshold: 0.005  # Lower threshold for sensitivity
```

### 2. Multi-Class Visualization
The system generates comprehensive visualizations:
- **Class Prediction Maps**: Color-coded segmentation
- **Probability Maps**: Confidence heatmaps for each class
- **Overlay Visualizations**: Predictions overlaid on original images
- **Error Analysis**: Comparison with ground truth (when available)

### 3. Performance Optimization
- **GPU Acceleration**: Automatic GPU detection and usage
- **Batch Processing**: Efficient processing of multiple images
- **Memory Management**: Automatic cleanup and optimization
- **Parallel Processing**: Multi-threaded image loading

### 4. Quality Assessment Metrics
For each processed image:
- **Coverage Statistics**: Percentage of each class
- **Region Analysis**: Number and size of detected regions
- **Confidence Analysis**: Distribution of prediction confidence
- **Performance Metrics**: Processing time and speed

## Model Requirements

### Input Specifications
- **Image Format**: Grayscale (automatically converted if RGB)
- **Image Size**: Automatically resized to model input size (typically 1024x1024)
- **Supported Formats**: JPG, PNG, BMP, TIFF, TIF

### Model Architecture
- **Input Channels**: 1 (grayscale)
- **Output Classes**: 3 (background, dirt, scratches)
- **Architecture**: U-Net based with configurable depth
- **Output Format**: Multi-class segmentation maps

## Best Practices

### 1. Configuration Management
- Keep separate configs for different use cases
- Use descriptive names for configuration files
- Document custom parameters in config comments

### 2. Batch Processing
- Process images in chunks for memory efficiency
- Monitor system resources during large batch jobs
- Use appropriate patch sizes based on image resolution

### 3. Quality Control
- Always review visualizations for validation
- Check region analysis for reasonable results
- Monitor inference times for performance issues

### 4. Result Management
- Organize output directories by date/batch
- Archive results for future reference
- Use JSON results for automated analysis

## Integration Examples

### Python Integration
```python
from src.inference.single_predictor import SingleImagePredictor
from src.inference.full_phone_prediction import FullScreenPredictor

# Initialize predictors
single_predictor = SingleImagePredictor(...)
full_predictor = FullScreenPredictor(...)

# Single prediction
result = single_predictor.single_prediction_pipeline(
    "path/to/image.jpg",
    save_results=True,
    output_dir="output/"
)

# Full phone prediction
result = full_predictor.single_prediction_pipeline(
    "path/to/phone.jpg",
    patch_size=1024,
    overlap=0.2
)
```



## Conclusion

The `inf.py` script provides a comprehensive and flexible inference system for multi-class dirt and scratch detection. By understanding the different modes and configuration options, you can optimize the system for your specific use case, whether it's single image analysis, batch processing, or detailed full phone screen assessment.

For additional support or advanced customization, refer to the source code documentation and configuration examples provided in the repository.