# Recurrent Pixel Embedding for Instance Grouping

For paper, slides and poster, please refer to our [project page](http://www.ics.uci.edu/~skong2/SMMMSG.html "pixel-grouping")

![alt text](http://www.ics.uci.edu/~skong2/image/icon_pixelEmbedding.png "visualization")

![alt text](https://raw.githubusercontent.com/aimerykong/Recurrent-Pixel-Embedding-for-Instance-Grouping/master/figure_to_show/demo_combo_v2.png "visualization")


We introduce a differentiable, end-to-end trainable framework for solving pixel-level grouping problems such as instance segmentation consisting of two novel components. First, we regress pixels into a hyper-spherical embedding space so that pixels from the same group have high cosine similarity while those from different groups have similarity below a specified margin. We analyze the choice of embedding dimension and margin, relating them to theoretical results on the problem of distributing points uniformly on the sphere. Second, to group instances, we utilize a variant of mean-shift clustering, implemented as a recurrent neural network parameterized by kernel bandwidth. This recurrent grouping module is differentiable, enjoys convergent dynamics and probabilistic interpretability. Backpropagating the group-weighted loss through this module allows learning to focus on only correcting embedding errors that won't be resolved during subsequent clustering. Our framework, while conceptually simple and theoretically abundant, is also practically effective and computationally efficient. We demonstrate substantial improvements over state-of-the-art instance segmentation for object proposal generation, as well as demonstrating the benefits of grouping loss for classification tasks such as boundary detection and semantic segmentation.


Several demos are included as below. 
As for details on the training, demo and code, please go into each demo folder.

1. demo 1: a tutorial for learning the embedding hypersphere and mean shift grouping. 
	We use instance segmentation as example, and include useful visualization functions. [TODO];
2. demo 2: boundary detection on BSDS500 dataset (also including code, model, visualization) [TODO];
3. demo 3: objectness proposal detection on PASCAL VOC2012 dataset [TODO];
4. demo 4: instance-level segmentation on PASCAL VOC2012 dataset [TODO].

Please download those models from the [google drive](http://www.ics.uci.edu/~skong2/SMMMSG.html)

MatConvNet is used in our project, and some functions are changed/added. Please compile accordingly by adjusting the path --

```python
LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:local matlab 

path_to_matconvnet = './libs/matconvnet-1.0-beta23_modifiedDagnn/';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(fullfile(path_to_matconvnet, 'matlab'));
vl_compilenn('enableGpu', true, ...
               'cudaRoot', '/usr/local/cuda-7.5', ...
               'cudaMethod', 'nvcc', ...
               'enableCudnn', true, ...
               'cudnnRoot', '/usr/local/cuda-7.5/cudnn-v5') ;

```


If you find our model/method/dataset useful, please cite our work:

    @inproceedings{kong2017grouppixels,
      title={Recurrent Pixel Embedding for Instance Grouping},
      author={Kong, Shu and Fowlkes, Charless},
      booktitle={arxiv},
      year={2017}
    }




last update: 11/17/2017

Shu Kong

aimerykong At g-m-a-i-l dot com

