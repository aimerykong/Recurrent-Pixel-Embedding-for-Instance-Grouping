# Recurrent Pixel Embedding for Instance Grouping

For project information, please refer to our [project page](https://mscvprojects.ri.cmu.edu/2020teamc/team/ "RGBT-detection")


![alt text](https://www.youtube.com/watch?v=vRJTlpsGvTs "video demo")


Object detection with multimodal inputs can improve many safety-critical perception systems such as autonomous vehicles (AVs). Motivated by AVs that operate in both day and night, we study multimodal object detection with RGB and thermal cameras, since the latter can provide much stronger object signatures under poor illumination. We explore strategies for fusing information from different modalities. Our key contribution is a non-learned late-fusion method that fuses together bounding box detections from different modalities via a simple probabilistic model derived from first principles. Our simple approach, which we call Bayesian Fusion, is readily derived from conditional independence assumptions across different modalities. We apply our approach to benchmarks containing both aligned (KAIST) and unaligned (FLIR) multimodal sensor data. Our Bayesian Fusion outperforms prior work by more than {\bf 13\%} in relative performance.


**keywords**: Object Detection, Thermal, infrared camera, RGB-thermal detection, multimodality, multispectral, autonomous driving, sensor fusion, non-maximal suppression, probablistic modeling.



If you find our model/method/dataset useful, please cite our work ([arxiv manuscript](https://arxiv.org/abs/1712.08273)):

    @inproceedings{RGBT-detection,
      title={Multimodal Object Detection via Bayesian Fusion},
      author={Chen, Yi-Ting and Shi, Jinghao and Mertz, Christoph and Kong, Shu and Ramanan, Deva},
      booktitle={preprint},
      year={2021}
    }


last update: April, 2021

Shu Kong

aimerykong At g-m-a-i-l dot com

