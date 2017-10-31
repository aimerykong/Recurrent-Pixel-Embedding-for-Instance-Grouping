# Learning to Group Pixels into Boundaries, Objectness, Segments, and Instances
### Code, demo and model for our project of learning to group pixels

![alt text](http://www.ics.uci.edu/~skong2/image/SMMMSG_small.png "visualization")

For papers, slides and posters, please refer to our [project page](www.ics.uci.edu/~skong2/SMMMSG.html "pixel-grouping")

An end-to-end trainable framework is introduced for solving pixel-labeling vision problems. The framework consists of two novel modules, pixel-pair spherical max-margin embedding regression and recurrent mean shift grouping. While architecture-wise agnostic, conceptually simple, computationally efficient, practically effective, and theoretically abundant, the framework can be purposed for boundary detection, object proposal detection, generic and instance-level segmentation, spanning low-, mid- and high-level vision tasks. Thorough experiments demonstrate that the new framework achieves state-of-the-art performance on all these tasks.



Several demos are/will be included as below -- 

1. demo 1: tutorial for learning the embedding hyperspherical space and mean shift grouping, instance segmentation as example [TODO];
2. demo 2: boundary detection on BSDS500 dataset (code, model, visualization) [TODO];
3. demo 3: objectness proposal detection on PASCAL VOC2012 dataset [TODO];
4. demo 4: instance-level segmentation on PASCAL VOC2012 dataset [TODO].

Please download those models from the [google drive](www.ics.uci.edu/~skong2/SMMMSG.html)

MatConvNet is used in our project, some functions are changed. So it might be required to re-compile. Useful commands are --

```python
LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:local matlab 

path_to_matconvnet = '../matconvnet';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(fullfile(path_to_matconvnet, 'matlab'));
vl_compilenn('enableGpu', true, ...
               'cudaRoot', '/usr/local/cuda-7.5', ...
               'cudaMethod', 'nvcc', ...
               'enableCudnn', true, ...
               'cudnnRoot', '/usr/local/cuda-7.5/cudnn-v5') ;

```

As for details on the training, demo and code, please go into each demo folder.


If you find our model/method/dataset useful, please cite our work:

    @inproceedings{kong2017lowrankbilinear,
      title={Learning to Group Pixels into Boundaries, Objectness, Segments and Instances},
      author={Kong, Shu and Fowlkes, Charless},
      booktitle={arxiv},
      year={2017}
    }




last update: 10/31/2017

Shu Kong

aimerykong At g-m-a-i-l dot com

