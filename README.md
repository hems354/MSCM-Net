# MSCM-Net for pCR Prediction

The code for training, testing, and visualization of the MSCM-Net model for pathological complete response (pCR) prediction from 3D CT images.

---

## Requirements

monai  
numpy  
torch  
scipy  
matplotlib  
pytorch-grad-cam  


## Getting Started

### Installation

Create and activate a new conda environment

---

## Training

Run the training script: 

```python train.py```
## Testing

Run the testing script: 

```python test.py```
## Grad-CAM Visualization

Grad-CAM visualization can be implemented based on the `pytorch_grad_cam` package.  
Specifically, users can refer to the `GradCAM` function provided in the `pytorch_grad_cam` library and adapt it for the MSCM-Net architecture to generate 3D activation maps for model interpretation. A typical workflow includes loading a trained MSCM-Net checkpoint, selecting a target convolutional layer, resizing and normalizing the input 3D image, and then applying Grad-CAM to obtain the 3D attention map. The resulting heatmap can be visualized by extracting a representative slice and overlaying it on the corresponding original CT image.
