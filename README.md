# \<epam\>
![image](https://github.com/user-attachments/assets/c564dc1b-ef6e-49ef-9771-c8db051600ac)

## Task and approach
Business task is to detect and localize different types of dirt on the front and rear surfaces of the mobile devices.
We will utilize high-resolution images available from the grading machine.
We selected semantic segmentation approach with simplified post-processing of network outputs to detect instances of dirt and their classes.
Synthetically generated dataset with automatic pixel annotations is utilized to pre-train the semantic segmentation model.

Currently, dataset has only two class labels: `dirt` pixels and `clean` (background) pixels, as we utilize only clean samples without noticeable permanent defects as a background for dataset generation. We utilize a unclassified subset of dirt samples from the original dataset, provided by FutureDial team.
Please, refer to the [**Dataset generation**](https://github.com/chencen2000/epam/edit/main/README.md#dataset-generation) section for further information about current dataset generation approach.

### TODO: next steps
Once we get the pre-trained model, capable of differentiating all the dirt from background, we plan to apply it to clean samples containing various scratches.
We  expect that the model will have false positive detections on them. Thus, we will obtain hard negative samples for further model fine-tuning and automatic labels for the third distinctive class, `scratches`.
We are currently in the process of annotating additional dirt samples and some permanent defects together with their class labels. We do not plan to provide separate manual annotations for scratches - we expect to get enough scratch samples from original dataset, model false positives and xml files.

## Dataset generation
- We downsample original clean images x2 by width and height to generate synthetic dirt images. We use additional scaling, rotation and translation of randomly selected dirt samples as an augmantation technique to superimpose dirt om clean images.
- Each generated sample contains multiple dirt instances superimposed over clean background.
- Generated sample and corresponding binary mask is then split into non-overlapping patches of `1024x1024` resolution (TODO: add patch overlapping to increase the amount of data). Those patches are then utilized for semantic segmentation model pre-training.

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

![UNet-architecture-This-diagram-is-based-on-the-original-UNet-publication-20 (1)](https://github.com/user-attachments/assets/d38f1a8c-9f54-4bf4-b292-041244796a40)

Slim model architecture diagram. The architecture diagram is preliminary and approximate, serves as an illustration of the approach, the exact number of channels on some of the layers can be different in the training [code](https://github.com/chencen2000/epam/blob/main/src/training.py).

