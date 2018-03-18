# CellDetector
https://www.kaggle.com/c/data-science-bowl-2018

### Table of Content
- [Introduction](README.md#introduction)
- [Run instruction](README.md#run-instruction)
- [Data preprocessing](README.md#data-preprocessing)
  - [Input browse](README.md#input-browse) 
  - [Statistics](README.md#statistics)
  - [Label preparation](README.md#label-preparation)
- [Model training](README.md#data-training)
  - [Architecture](README.md#architecture)
  - [Kernels](README.md#kernels)
- [Model evaluation](README.md#model-evaluation)
- [Authors](README.md#authors)


### Introduction


### Run instruction

(How to apply the machine to new data)
(single and batch)

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

#### Architecture

#### Kernel

### Model evaluation

### Authors
