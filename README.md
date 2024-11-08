<p align="center">
  <h1 align="center">LSSInst: Improving Geometric Modeling in LSS-Based BEV Perception with Instance Representation
  </h1>
  <p align="center">
    3DV, 2025
    <br />
    <a href="https://WeijieMax.github.io/"><strong>Weijie Ma</strong></a>,
    <a href="https://github.com/Jingwei-Jiang/"><strong>Jingwei Jiang</strong></a>,
    <a href="https://github.com/Young98CN/"><strong>Yang Yang</strong></a>,
    <a href="https://lovesnowbest.site/"><strong>Zehui Chen</strong></a>,
    <a href="https://scholar.google.com/citations?user=FaOqRpcAAAAJ"><strong>Hao Chen</strong></a>
  </p>

<div align="center">

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2405.17427-red)](https://arxiv.org/abs/)

</div>

### TLDR

A novel and effective exploration of geometric modeling with intance representation for the modern LSS-based BEV detection.

### 

### Introduction

With the attention gained by camera-only 3D object detection in autonomous driving, methods based on Bird-Eye-View (BEV) representation especially derived from the forward view transformation paradigm, i.e., lift-splat-shoot (LSS), have recently seen significant progress. The BEV representation formulated by the frustum based on depth distribution prediction is ideal for learning the road structure and scene layout from multi-view images. However, to retain computational efficiency, the compressed BEV representation such as in resolution and axis is inevitably weak in retaining the individual geometric details, undermining the methodological generality and applicability. With this in mind, to compensate for the missing details and utilize multi-view geometry constraints, we propose LSSInst, a two-stage object detector incorporating BEV and instance representations in tandem. The proposed detector exploits fine-grained pixel-level features that can be flexibly integrated into existing LSS-based BEV networks. Having said that, due to the inherent gap between two representation spaces, we design the instance adaptor for the BEV-to-instance semantic coherence rather than pass the proposal naively. Extensive experiments demonstrated that our proposed framework is of excellent generalization ability and performance, which boosts the performances of modern LSS-based BEV perception methods without bells and whistles and outperforms current LSS-based state-of-the-art works on the large-scale nuScenes benchmark.

### 

## Citation

If you find our work useful for your project, please consider citing our paper:


```bibtex
@article{lssinst,
  title={LSSInst: Improving Geometric Modeling in LSS-Based BEV Perception with Instance Representation},
  author={Weijie Ma and Jingwei Jiang and Yang Yang and Zehui Chen and Hao Chen},
  journal={International Conference on 3D Vision (3DV)},
  year={2025}
}
```
