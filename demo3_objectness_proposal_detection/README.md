# Learning to Group Pixels into Boundaries, Objectness, Segments, and Instances

For papers, slides and posters, please refer to our [project page](http://www.ics.uci.edu/~skong2/SMMMSG.html "pixel-grouping")


![alt text](https://raw.githubusercontent.com/aimerykong/Recurrent-Pixel-Embedding-for-Instance-Grouping/master/demo3_objectness_proposal_detection/results/id1_summary.jpg "visualization")

To run the demo, please download the model from the [google drive](https://drive.google.com/drive/u/1/folders/1Ii1RPiwB-SvQchnmRvSVEcGSCCrxvpHc), and put it under ./basemodel








MatConvNet is used in our project, and some functions are changed/added. Please compile accordingly by adjusting the path --

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


If you find our model/method/dataset useful, please cite our work:

    @inproceedings{kong2017grouppixels,
      title={Learning to Group Pixels into Boundaries, Objectness, Segments and Instances},
      author={Kong, Shu and Fowlkes, Charless},
      booktitle={arxiv},
      year={2017}
    }




last update: 03/20/2018

Shu Kong

aimerykong At g-m-a-i-l dot com

