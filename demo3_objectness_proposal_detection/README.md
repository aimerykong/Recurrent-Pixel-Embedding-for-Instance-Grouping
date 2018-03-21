# Recurrent Pixel Embedding for Instance Grouping

For papers, slides and posters, please refer to our [project page](http://www.ics.uci.edu/~skong2/SMMMSG.html "pixel-grouping")


![alt text](https://raw.githubusercontent.com/aimerykong/Recurrent-Pixel-Embedding-for-Instance-Grouping/master/demo3_objectness_proposal_detection/results/id1_summary.jpg "visualization")

To run the demo, please download the model from the [google drive](https://drive.google.com/drive/u/1/folders/1Ii1RPiwB-SvQchnmRvSVEcGSCCrxvpHc), and put it under ./basemodel

Simply running script ["demo.m"](https://github.com/aimerykong/Recurrent-Pixel-Embedding-for-Instance-Grouping/blob/master/demo3_objectness_proposal_detection/demo.m) will help visualize embedding features. The demo will traverse images under folder ["images"](https://github.com/aimerykong/Recurrent-Pixel-Embedding-for-Instance-Grouping/tree/master/demo3_objectness_proposal_detection/images), and save all visual results under folder ["results"](https://github.com/aimerykong/Recurrent-Pixel-Embedding-for-Instance-Grouping/tree/master/demo3_objectness_proposal_detection/results) (*find more in this folder*).






As MatConvNet is used in our project, please compile accordingly by adjusting the following path.

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

    @inproceedings{kong2018grouppixels,
      title={Recurrent Pixel Embedding for Instance Grouping},
      author={Kong, Shu and Fowlkes, Charless},
      booktitle={2018 Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2018}
    }




last update: 03/20/2018

Shu Kong

aimerykong At g-m-a-i-l dot com

