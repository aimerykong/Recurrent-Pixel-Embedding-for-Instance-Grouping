# Recurrent Pixel Embedding for Instance Grouping

For papers, slides and posters, please refer to our [project page](http://www.ics.uci.edu/~skong2/SMMMSG.html "pixel-grouping")

<img src="http://www.ics.uci.edu/~skong2/image/fig00_visualization.jpg" alt="" data-canonical-src="http://www.ics.uci.edu/~skong2/image/fig00_visualization.jpg " width="500" height="390" />

<img src="http://www.ics.uci.edu/~skong2/image/fig01_visualization_looping.jpg" alt="" data-canonical-src="http://www.ics.uci.edu/~skong2/image/fig01_visualization_looping.jpg " width="545" height="350" />


This folder provides the first demo on how to train the model for instance segmentation/grouping. Please run the following scripts sequentially--

1. step001...m: download MNISt dataset and generate the training&testing dataset for this demo.
2. step002...m: generate the imdb structure which provides training and testing splits with index and meta information (for mean subtraction, etc.).
3. step003...m: train for instance grouping based on a semantic segmentation network. Five loops of Mean Shift are added with the loss.
4. step004...m: fine-tune the model at step003. This step is optional, and for demonstrating how to read and fine-tune with customized configurations.
5. step005...m: visualize the results with multiple Mean Shift loops during training.

Some snapshots are shared in the [google drive](https://drive.google.com/open?id=1QxnjCPEXekxYPhNHz5-pB-lW346xhuNU). Please download this folder, rename it to "exp" and put it here, so that all the scripts can know where to find the models for visualization and resuming the training.


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


