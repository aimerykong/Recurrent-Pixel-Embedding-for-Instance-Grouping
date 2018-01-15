# Recurrent Pixel Embedding for Instance Grouping

For papers, slides and posters, please refer to our [project page](http://www.ics.uci.edu/~skong2/SMMMSG.html "pixel-grouping")

<img src="http://www.ics.uci.edu/~skong2/image/fig00_visualization.jpg" alt="" data-canonical-src="http://www.ics.uci.edu/~skong2/image/fig00_visualization.jpg " width="500" height="390" />

<img src="http://www.ics.uci.edu/~skong2/image/fig01_visualization_looping.jpg" alt="" data-canonical-src="http://www.ics.uci.edu/~skong2/image/fig01_visualization_looping.jpg " width="545" height="350" />


This folder is self-contained that provides an analysis on the gradient through Mean Shift loop(s). Please run script "simulation07_1D_GBMS_1step_trajectory" to visualize the trajectories of 1D points. 
Modifying "meanShiftNumber = 7;" in Line-13 will back-propagate different loops of Mean Shift.


MatConvNet is used in our project, and some functions are changed/added. The opensource toolbox is [../libs/matconvnet-1.0-beta23_modifiedDagnn](https://github.com/aimerykong/Recurrent-Pixel-Embedding-for-Instance-Grouping/tree/master/libs/matconvnet-1.0-beta23_modifiedDagnn). Please compile accordingly by adjusting the path --

```python
LD_LIBRARY_PATH=/usr/local/cuda/lib64:local matlab 

path_to_matconvnet = '../libs/matconvnet-1.0-beta23_modifiedDagnn';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(fullfile(path_to_matconvnet, 'matlab'));
vl_compilenn('enableGpu', true, ...
               'cudaRoot', '/usr/local/cuda', ...
               'cudaMethod', 'nvcc', ...
               'enableCudnn', true, ...
               'cudnnRoot', '/usr/local/cuda/cudnn') ;

```


If you find our model/method/dataset useful, please cite our work:

    @inproceedings{kong2017grouppixels,
      title={Learning to Group Pixels into Boundaries, Objectness, Segments and Instances},
      author={Kong, Shu and Fowlkes, Charless},
      booktitle={arXiv:1712.08273},
      year={2017}
    }




last update: 01/15/2018

Shu Kong

aimerykong At g-m-a-i-l dot com


