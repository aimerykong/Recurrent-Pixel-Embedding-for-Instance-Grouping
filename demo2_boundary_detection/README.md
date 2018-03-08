# Learning to Group Pixels into Boundaries, Objectness, Segments, and Instances

For papers, slides and posters, please refer to our [project page](http://www.ics.uci.edu/~skong2/SMMMSG.html "pixel-grouping")

<img src="http://www.ics.uci.edu/~skong2/image/demo_boundaryDet.png" alt="" data-canonical-src="http://www.ics.uci.edu/~skong2/image/demo_boundaryDet.png " width="600" height="350" />


This demo is for boundary detection. When downloading our trained models from the [google drive](https://drive.google.com/drive/u/1/folders/1MfWWToezy9E6Sv6jY7JfxoUo2igX42Wg), please copy the whole folder(s) inside the link to "models" directory.

Running script [main001_visualize.m](https://github.com/aimerykong/Recurrent-Pixel-Embedding-for-Instance-Grouping/blob/master/demo2_boundary_detection/main001_visualize.m) will show you the embedding visualization as well as intermediate&final results. Note that you might need to compile several scripts which are used for boundary thinning (NMS) from Piotr Dollar's [edge toolbox](https://github.com/pdollar/edges). If issues about this happens, commenting out the related lines will help executing the script.



If you find our model/method/dataset useful, please cite our work ([draft at arxiv](https://arxiv.org/abs/1712.08273)):

    @inproceedings{kong2018grouppixels,
      title={Recurrent Pixel Embedding for Instance Grouping},
      author={Kong, Shu and Fowlkes, Charless},
      booktitle={2018 Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2018}
    }





last update: 03/08/2018

Shu Kong

aimerykong At g-m-a-i-l dot com

