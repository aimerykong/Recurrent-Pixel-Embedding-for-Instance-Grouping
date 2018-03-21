# Recurrent Pixel Embedding for Instance Grouping

For paper, slides and poster, please refer to our [project page](http://www.ics.uci.edu/~skong2/SMMMSG.html "pixel-grouping")


![alt text](http://www.ics.uci.edu/~skong2/image/icon_pixelEmbedding.png "visualization")


We introduce a differentiable, end-to-end trainable framework for solving pixel-level grouping problems such as instance segmentation consisting of two novel components. First, we regress pixels into a hyper-spherical embedding space so that pixels from the same group have high cosine similarity while those from different groups have similarity below a specified margin. We analyze the choice of embedding dimension and margin, relating them to theoretical results on the problem of distributing points uniformly on the sphere. Second, to group instances, we utilize a variant of mean-shift clustering, implemented as a recurrent neural network parameterized by kernel bandwidth. This recurrent grouping module is differentiable, enjoys convergent dynamics and probabilistic interpretability. Backpropagating the group-weighted loss through this module allows learning to focus on only correcting embedding errors that won't be resolved during subsequent clustering. Our framework, while conceptually simple and theoretically abundant, is also practically effective and computationally efficient. We demonstrate substantial improvements over state-of-the-art instance segmentation for object proposal generation, as well as demonstrating the benefits of grouping loss for classification tasks such as boundary detection and semantic segmentation.

**keywords**: Pixel Embedding, Recurrent Grouping, Boundary Detection, Object Proposal Detection, Instance Segmentation, Semantic Segmentation, Maximum Margin, Metric Learning, Hard Pixel Pair Mining, Distributing Many Points on a (Hyper-) Sphere, Mean Shift Clustering, Recurrent Networks, Mode Seeking, von Mises Fisher Distribution, Robust Loss, Instance-aware Pixel Weighting, non-parametric model, etc.


Several demos are included as below. 
As for details on the training, demo and code, please go into each demo folder.

1. [demo 1](https://github.com/aimerykong/Recurrent-Pixel-Embedding-for-Instance-Grouping/tree/master/demo1_tutorial_instance_segmentation): a tutorial for learning the embedding hypersphere and mean shift grouping. 
	We use instance segmentation as example, and include useful visualization functions. [**Ready**];
2. [demo 2](https://github.com/aimerykong/Recurrent-Pixel-Embedding-for-Instance-Grouping/tree/master/demo2_boundary_detection): boundary detection on BSDS500 dataset (also including code, model, visualization) [**Ready**];
3. [demo 3](https://github.com/aimerykong/Recurrent-Pixel-Embedding-for-Instance-Grouping/tree/master/demo3_objectness_proposal_detection): objectness proposal detection on PASCAL VOC2012 dataset [**Ready**];
4. [demo 4](https://github.com/aimerykong/Recurrent-Pixel-Embedding-for-Instance-Grouping/tree/master/demo4_semantic_instance_segmentation): instance-level segmentation on PASCAL VOC2012 dataset [TODO].
5. [demo 5](https://github.com/aimerykong/Recurrent-Pixel-Embedding-for-Instance-Grouping/tree/master/demo5_analysis_MShift_gradient): analysis of Mean Shift gradient. [**Ready**] 

Please download those models from the [google drive](https://drive.google.com/drive/folders/1K2bCmz_mldIhV1e3hCbtBrARZR_0bylm?usp=sharing). 

MatConvNet is used in our project, and some functions are changed/added. Please compile accordingly by adjusting the path --

```python
LD_LIBRARY_PATH=/usr/local/cuda/lib64:local matlab 

path_to_matconvnet = './libs/matconvnet-1.0-beta23_modifiedDagnn/';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(fullfile(path_to_matconvnet, 'matlab'));
vl_compilenn('enableGpu', true, ...
               'cudaRoot', '/usr/local/cuda', ...
               'cudaMethod', 'nvcc', ...
               'enableCudnn', true, ...
               'cudnnRoot', '/usr/local/cuda/cudnn/lib64') ;

```


If you find our model/method/dataset useful, please cite our work ([draft at arxiv](https://arxiv.org/abs/1712.08273)):

    @inproceedings{kong2018grouppixels,
      title={Recurrent Pixel Embedding for Instance Grouping},
      author={Kong, Shu and Fowlkes, Charless},
      booktitle={2018 Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2018}
    }



![alt text](https://raw.githubusercontent.com/aimerykong/Recurrent-Pixel-Embedding-for-Instance-Grouping/master/figure_to_show/demo_combo_v2.png "visualization")


last update: 03/08/2018

Shu Kong

aimerykong At g-m-a-i-l dot com

