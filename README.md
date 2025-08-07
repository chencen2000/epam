# \<epam\>
![image](https://github.com/user-attachments/assets/c564dc1b-ef6e-49ef-9771-c8db051600ac)

## Task and approach
Business task is to detect and localize different types of dirt and defects on frontal and rear surfaces of the mobile devices.
For that purpose, we can utilize high-resolution images (7296x16000 pixels) available from the grading machine.
Due to very limited compute resources (3s processing time on Intel i7-6700 CPU), we downscale these images to 1/2 of the original resolution for processing. 
We selected semantic segmentation approach with simplified post-processing of network outputs to detect dirt and defects.
Synthetically generated dataset with automatic pixel annotations is utilized to pre-train the semantic segmentation model.

Currently, dataset has four class labels: `condensation`, `dirt`, `scratches` and `clean` (background) pixels.
For model training one can utilize either clean samples without any noticeable defects or scratched samples with only scratches on them as a reference background images for dataset generation.
On clean samples patches of all three types, condensation, dirt and scratch are being superimposed over reference background.
Once we get the pre-trained model, capable of differentiating everything from background, we can run it (inference) on clean scratchy samples containing only scratches and use model predictions on them as automatic annotations for scratches.
Then, on these scratched samples only condensation and dirt patches are being superimposed, and model can be re-trained (fine-tuned) on this dataset, containing real samples with real scratches.
Patches used for sumerimposing were collected from training batches Batch1 and Batch2.
Please, refer to the [**Dataset generation**](https://github.com/chencen2000/epam/edit/main/README.md#dataset-generation) section for further information about current dataset generation approach.

## Dataset generation
- We downsample original clean images x2 by width and height to generate synthetic images. We use additional scaling, rotation and translation of randomly selected dirt samples as an augmantation technique to superimpose dirt on clean images.
- Each generated sample contains multiple dirt instances superimposed over clean background.
- Generated sample and corresponding masks is then split into non-overlapping patches of `1792x1792` resolution. Those patches are then utilized for semantic segmentation model pre-training.

### Python modules description
The [**src**](https://github.com/chencen2000/epam/tree/develop) folder contains the core Python scripts for a synthetic dirt image data generation pipeline. This includes modules for:

- **Argument Parsing** ([arguments.py](https://github.com/chencen2000/epam/blob/main/src/arguments.py)): Handles command-line argument definition and parsing for configuration.
- **Data Generation Orchestration** ([data_generation.py](https://github.com/chencen2000/epam/blob/main/src/data_generation.py)): The main script that orchestrates the data generation process, including loading images, applying dirt, and saving synthetic data and annotations.
- **Configuration** ([data_generation_config.py](https://github.com/chencen2000/epam/blob/main/src/data_generation_config.py)): Defines default parameters for the data generation process.
- **Dirt Placement Logic** ([dirt_placement.py](https://github.com/chencen2000/epam/blob/main/src/dirt_placement.py)): Implements the algorithms for placing dirt samples on images, including background estimation and mask creation with dual thresholds and distorted boundaries.
- **Image Operations** ([image_operations.py](https://github.com/chencen2000/epam/blob/main/src/image_operations.py)): Provides fundamental image processing utilities such as loading, saving, scaling, rotation, and bounding box calculations.
- **Mask Generation** ([masks2.py](https://github.com/chencen2000/epam/blob/main/src/masks2.py)): Focuses on creating and manipulating distorted polygon masks used for dirt placement.
- **General Utilities** ([utils.py](https://github.com/chencen2000/epam/blob/main/src/utils.py)): Contains helper functions for file handling (listing image files) and splitting images into patches with annotations.
- **Visualization** ([visualization.py](https://github.com/chencen2000/epam/blob/main/src/visualization.py)): Offers tools to visualize various stages of the image processing and mask creation.

## Model architecture and pre-training process
Our slim model inherits the ideas from original [U-net](https://arxiv.org/abs/1505.04597) as proposed by Ronneberger et al. However, it has fewer stages and lower number of channels to meet speed requirements. It utilizes dilated convolitions (x2) at every stage to increase the receptive field of the model. Overall, the model has only a fraction of number of trainable parameters, compared to the original U-net model.

![UNet-architecture-This-diagram-is-based-on-the-original-UNet-publication-20 (1)](https://github.com/user-attachments/assets/3d8394fc-00ff-4c15-a83b-a6d9fb6a6e9e)

Slim model architecture diagram. The architecture diagram is approximate, serves as an illustration of the approach, the exact number of channels on some of the layers can be different in the training [code](https://github.com/chencen2000/epam/blob/main/src/training.py).

