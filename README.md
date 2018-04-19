# CellDetector
https://www.kaggle.com/c/data-science-bowl-2018

### Table of Content
- [Introduction](README.md#introduction)
- [Run instruction](README.md#run-instruction)
- [Dependencies](README.md#dependencies)
- [Data preprocessing](README.md#data-preprocessing)
  - [Input browse](README.md#input-browse) 
  - [Statistics](README.md#statistics)
  - [Label preparation](README.md#label-preparation)
- [Model training](README.md#data-training)
  - [Data Feed](README.md#data-feed)
  - [Architecture](README.md#architecture)
  - [Parameters](README.md#parameters)
- [Model prediction and evaluation](README.md#model-prediction-and-evaluation)
- [Results](README.md#parameters)
- [Authors](README.md#authors)


### Introduction

The implementation of 2018 Data Science Bowl competition.

Identify single cell in the input microscope images,
export the predicted cells with explicitly labeled boundaries,
and a `csv` file with individual cells.

The implementation is based on [deep contextual network](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/11789).

Any suggestion is welcome.


### Run instruction

To pre-process the input data based on competition stage1 training dataset (not provided in the repo)
, visualize the input samples and generate `y_label` numpy array:

`python preprocessing/image_overlay.py`

To generate model based on competition stage1 training dataset (not provided in the repo): 

`python DCN/deepcn.py`

To run TensorBoard:

`tensorboard --logdir DCN/logs/run1`

To predict:

Change the input and output file path in [`prediction.py`](https://github.com/OXPHOS/CellDetector/blob/master/DCN/prediction.py)

`python DCN/predction.py`

The `prediction.py` will generate the predicted cells with explicitly labeled boundaries,
and a `csv` file labeling the masks for individual cells.


### Dependencies

- Tensorflow

- TensorBoard

- Numpy

- Opencv

- skimage

- [imgaug](https://github.com/aleju/imgaug)

- [tqdm](https://github.com/noamraph/tqdm)

### Data preprocessing

#### Input browse

In the training samples, each input image comes with a handful of independent masks labeling each individual cell nucleus. 
To get a general idea about how nuclei interact with each other and how they distribute in each image, 
we [overlayed](https://github.com/OXPHOS/CellDetector/blob/master/preprocessing/image_overlay.py#L153) the masks to the original input images. 

There are in total **670** images in stage1 training samples. After browsing all overlayed images, we identify one bad input data:

![baddata](https://github.com/OXPHOS/CellDetector/blob/master/readme_images/stage1/7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80_tri.png)

With `id=7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80`, 
despite numerous detectable nuclei in the original image, the image is severely underlabeled, and thus is removed from the training set.

Other images can be intuitively divided into three groups, based on their color/morphology:

- Fluoresence images, likely to be DAPI staining. Made up of most of the input samples. Size varies but all are dense dots.
![ex1](https://github.com/OXPHOS/CellDetector/blob/master/readme_images/stage1/1f6b7cead15344593b32d5f2345fc26713dc74d9b31306c824209d67da401fd8_tri.png)

- Brightfield images. Nuclei can be seen in the center of the cells.
![ex2](https://github.com/OXPHOS/CellDetector/blob/master/readme_images/stage1/4e07a653352b30bb95b60ebc6c57afbc7215716224af731c51ff8d430788cd40_tri.png)

- Purple brightfield images. Morpholgy varies a lot, but the nuclei are dense than the background. (Dot sizes stand for counts.)
![ex3](https://github.com/OXPHOS/CellDetector/blob/master/readme_images/stage1/5ef4442e5b8b0b4cf824b61be4050dfd793d846e0a6800afa4425a2f66e91456_tri.png)

#### Statistics

To better understand the training dataset, we looked into the following 3 attributes of the data:

- [Image size distribution](https://github.com/OXPHOS/CellDetector/blob/master/preprocessing/imagesize_distribution.py)
  
  We plotted the height and weight of each input image, and found that 49.9% of the input are `256*256`,
  while the dimensions of input spread over 9 different groups, with the largest image being `1040*1388`. (Dot sizes stand for counts.)
  
  ![image_size_distribution_plot](https://github.com/OXPHOS/CellDetector/blob/master/readme_images/stage1/imagesize_summary.png)
  
  Considering the image to be predicted will not share fixed dimensions as well, and to make the maximum use of the training dataset,
  we decided to construct the network without fully connected layer. 
  
- [Cell number distribution](https://github.com/OXPHOS/CellDetector/blob/master/preprocessing/cellnumber_distribution.py)
  
  While most of the images have 1-50 nulcei, the number of cells in each image spread from 1 to 375. 
  
  ![Cell number distribution plot](https://github.com/OXPHOS/CellDetector/blob/master/readme_images/stage1/cellnumber_distribution.png)
  
  We identified another subgroup of image based on their nuclei assembly struture (*D. melanogaster* embryos) which may 
  require special attention during training.
  
  ![dm embryos2](https://github.com/OXPHOS/CellDetector/blob/master/readme_images/stage1/a102535b0e88374bea4a1cfd9ee7cb3822ff54f4ab2a9845d428ec22f9ee2288_tri.png)

- [Cell size distribution](https://github.com/OXPHOS/CellDetector/blob/master/preprocessing/cellsize_distribution.py)
  
  Most of the cells are below `80*80`, while the largest one is `139*114`.
  ![Cell size distribution plot](https://github.com/OXPHOS/CellDetector/blob/master/readme_images/stage1/cellsize_summary.png)
  
#### Label preparation

  The key to predict nuclei is to predict the margin of the nuclei. Thus, besides **nuclei** and **background**, 
  we defined a pixel as **margin** if `pixel(i,j)`'s adjacent 8 neighbors - `pixels(i-1:i+1, j-1:j+1)` does not fall into the
  same (**nuclei** or **background**) categories.
  
  | Labels        | Categories    |
  | ------------- |:-------------:| 
  | 0       | background | 
  | 1      | nuclei   | 
  | 2 | margin   | 
  
  Each input image is converted to grayscale numpy arrays based on their labels. 
  Images reconstructed from the labels are like below (Last panel):
  
  ![labels](https://github.com/OXPHOS/CellDetector/blob/master/readme_images/stage1/1f6b7cead15344593b32d5f2345fc26713dc74d9b31306c824209d67da401fd8_margin.png)

### Model training

We implemented the training model with [deep contextual network](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/11789).
 
 
#### [Data feed](imageparser.py)

As the input images have different dimensions, they were padded by border mirror reflection and then 
divided into chunks for data feeding.

Images are then augmented with library [`imgaug`](https://github.com/aleju/imgaug). Following augmentation operations were performed on the training dataset:

- Color inversion
- GaussianBlur
- Sharpen
- GaussianNoise

When training, images are fed via [`BatchReader`](https://github.com/OXPHOS/CellDetector/blob/master/DCN/BatchReader.py) 
class according to the specified input path and batch size.


#### Architecture

The tensorboard visualized DCN is below:

![tensorboard_graph](https://github.com/OXPHOS/CellDetector/blob/master/readme_images/tensorboard)
 
Features from inputs were extracted from two `convolution` operations.

Following the 2-step `convolution`, inputs were down-sampled via `max pool`.

After the `max-pool`, 2-step `convolution` was re-performed, and the output was up-sampled to the original input size for prediction.

Down-sampling was performed twice and the inputs used for up-sampling were 1/4 and 1/16 of the original images.
 
The prediction layers are then summed up and returned with the sum layer.

The cross entropy with stochastic descent gradient were used to optimize paramters
 

#### Parameters

All convolution layers have kernel of `3*3` and stride of `1`.

All down-sampling has a stride of `2`.

The weight used for background, cell and boundary in loss function are `1`, `2`, `2`.

The input block size is `256*256`, with a stride of `256` and a batch size of `10`.

Data samples were iterate 12 times to obtain better prediction.



### Model prediction and evaluation

Saved model can be used for [prediction](https://github.com/OXPHOS/CellDetector/blob/master/DCN/prediction.py) 
via specifying the model path, input and output path.

Images to be predicted are resized and padding to `256*256`, predicted, and resized to the original size.

**Evaluation was not implemented properly. mean IoU would be preferred.**


### Result

The [loss](https://github.com/OXPHOS/CellDetector/blob/master/DCN/deepcn.py#L241) 
associated with the aforementioned parameters is
 
 ![loss](https://github.com/OXPHOS/CellDetector/blob/master/readme_images/loss)
 
 with the cyan being the loss of training samples (every 400 training steps) and orange being
 the loss of validation samples (every 1000 training steps, the validation samples were then fed 
 to the training machine as well.)
 
 The model is then used to predict the stage2 test dataset of the competition (not provided in the repo).
 
 The drawbacks of the predictions are:
 
 - Boundaries too thick
 
 - Unable to separate connected cells very well
 
 - Performed very bad on some data samples (listed below)
 
 ![pred1](https://github.com/OXPHOS/CellDetector/blob/master/readme_images/stage2/f9f27d4797e4d6eeca25d42b09c8ba8063394a7512eb954ad7b1ed884f58219d.png)
 ![pred2](https://github.com/OXPHOS/CellDetector/blob/master/readme_images/stage2/f26da2a67183aad1850a157153450a130b81fca4d8760c3d0b8b8e91a02cf340.png)
 ![pred3](https://github.com/OXPHOS/CellDetector/blob/master/readme_images/stage2/fae7766ca49d917e00d7967686ac86bdbf2d92b343914560ff428cd376614770.png)
 ![pred4](https://github.com/OXPHOS/CellDetector/blob/master/readme_images/stage2/fb9b0b2daa50af5e0eb219c08f1f2b8926efbb4827706311435b6c3d2aff8a20.png)
 

### Authors

