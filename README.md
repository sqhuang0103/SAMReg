# SAMReg
This document provides a quick guide to setting up and using SAMReg, a demonstration implementation that's currently under active development.
### Requirements
Before you start, make sure your environment meets these dependencies:

```
cuda==11.4
monai==1.3.0
nibabel==5.2.0
torch==1.12.0
torchvision==0.13.0
segment-anything==1.0
```

### Getting Started with Data
We've included a sample dataset in the `example` folder to help you get started quickly. This dataset contains pairs of images from different categories, some with labels and some without:
```commandline
.example/
|--- pathology/
| |--- image1.png
| |--- image2.ong
|--- prostate_2d/
| |--- image1.png
| |--- image2.png
| |--- label1.png
| |--- label2.png
|--- cardiac_2d/
| |--- ...
|--- abdomen_2d/
| |--- ...
|--- cell/
| |--- ...
```
To use your own data, simply add it to the `example` folder and specify the image paths when running the demo:
```commandline
--fix_image ./example/cardiac_2d/image1.png --mov_image ./example/cardiac_2d/image2.png
```

### Using SAM Pre-trained Models
Download the SAM pre-trained models and specify the checkpoint path for inference:
```commandline
--sam_checkpoint path/to/snapshot/
```
We recommend using the SAM-H model for this demo. You can find the models here:

- SAM-B: [Download SAM-B Model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
- SAM-L: [Download SAM-L Model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- SAM-H: [Download SAM-H Model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

### Quick Demo Instructions
To run a quick demo, use the following command:
```commandline
python demo.py --sam_checkpoint path/to/snapshot/
```
### Optional: Warping the Image
For traditional image warping, enable interpolation and specify the type of ROI:
```commandline
python demo.py --sam_checkpoint path/to/snapshot/ --interpolate --ROI_type pseudo_ROI 
```
Choose between `label_ROI` or `pseudo_ROI` for the metric object based on your preference.
