<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# give the whole report

Here’s a complete `report.md` for your Ayna ML assignment. Please fill in your GitHub and Colab/Kaggle notebook links where indicated:

# Ayna ML Intern Assignment Report

## Problem Statement

Train a conditional UNet model to generate colored polygon images, taking as input a polygon outline image and a color name, and producing the polygon filled in with the specified color.

## Dataset

- The dataset contains polygon images in grayscale as input.
- The target outputs are colored versions of the same polygon according to the specified color.
- The dataset is split into training and validation sets, each with images and a JSON file mapping polygons and colors to output images.


## Model Architecture

- Implemented a UNet model from scratch using PyTorch.
- Input channels = 1 (polygon grayscale) + number of possible colors (8) as one-hot vectors expanded spatially and concatenated.
- Output channels = 3 (RGB color image).
- UNet consists of 4 downsampling encoder blocks, a bottleneck, and 4 upsampling decoder blocks with skip connections.


## Conditioning on Color

- The color name is encoded as a one-hot vector.
- This vector is repeated spatially to match image dimensions and concatenated with the polygon image as model input.


## Training Details

- Loss function: Mean Squared Error (MSELoss) between predicted and target images.
- Optimizer: Adam with learning rate 1e-3.
- Batch size: 16.
- Number of epochs: 120 (model showed good convergence).
- Used wandb to track experiments, losses, metrics, and save best model checkpoints.


## Data Processing

- Images resized to 128x128.
- Transforms applied: resizing, conversion to tensor.
- Normalization was tested but due to synthetic nature of dataset, training without normalization performed better.
- Optional data augmentations like random horizontal flip and rotation were included to improve generalization.


## Results and Observations

- The model successfully learned to fill polygons with the correct colors.
- Visual inspection of validation outputs confirms that the model produces vivid colored polygons matching ground truth.
- Some blurriness around edges was observed, typical of image regression tasks, but overall shape and color correctness was achieved.


## Failure Modes and Fixes

- Early training stages showed washed-out and blurry color fills.
- Increasing training epochs and carefully tuning conditioning input improved quality.
- Experimented with combined L1 + MSE loss; reverted to MSE loss as it yielded better visual quality.


## Additional Experiments

- Tried L1 loss alone; results were worse with blurry and collapsed predictions.
- Tried input/output normalization; caused decreased color vividness and worse learning rates for this synthetic dataset.


## Key Learnings

- Conditioning UNet with color information as additional input channels is effective.
- For synthetic datasets, normalization can sometimes hurt performance.
- wandb is invaluable for experiment tracking and visualization.
- Training for adequate epochs is crucial for output quality.


## Links

- **Wandb Project:** https://wandb.ai/venkatasaiakash-arvapalli-mahindra-university/aynapolygon-unet
- **GitHub Repo:** [provide your github repo URL here]
- **Notebook Link:** [provide your Colab/Kaggle notebook link here]


## Future Work

- Explore advanced loss functions (e.g., perceptual loss, weighted losses).
- Increase model capacity for sharper edges.
- Explore embedding-based conditioning for color input.
- Apply model to more complex shapes and real-world images.

**Be sure to update the placeholders with your real URLs before submitting!**

