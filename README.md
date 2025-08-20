<p align="center">
  <h1 align="center">LSSInst: Improving Geometric Modeling in LSS-Based BEV Perception with Instance Representation
  </h1>
  <p align="center">
    <a href="https://openreview.net/forum?id=MaN2x3O2Rk">Paper link</a> in
    International Conference on 3D Vision (3DV), 2025
    <br />
    <a href="https://WeijieMax.github.io/"><strong>Weijie Ma</strong></a>,
    <a href="https://jingwei-jiang.github.io/"><strong>Jingwei Jiang</strong></a>,
    <a href="https://young98cn.github.io/"><strong>Yang Yang</strong></a>,
    <a href="https://lovesnowbest.site/"><strong>Zehui Chen</strong></a>,
    <a href="https://scholar.google.com/citations?user=FaOqRpcAAAAJ"><strong>Hao Chen</strong></a>
  </p>

<div align="center">
  <a href="https://arxiv.org/abs/2411.06173">
    <img src="https://img.shields.io/badge/arXiv-2411.06173-red" alt="arXiv">
  </a>
</div>

### TL; DR

A novel and effective exploration of geometric modeling with intance representation for the modern LSS-based BEV detection.

### 

### Brief Introduction

This codebase is built upon [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) and [SOLOFusion](https://github.com/Divadi/SOLOFusion). 

**Note**: This repository serves as a research reference implementation. Due to the rapid evolution of dependencies and the time elapsed since the original development, the code may require adjustments to work with current environments. We recommend referring to the original MMDetection3D and SOLOFusion documentation for the most up-to-date **installation** procedures and **environment** setup.

#### Quick Start

The main configuration file is located at: `configs/lssinst/lssinst.py`

**Stage-1 Initialization**: The configuration automatically loads [stage-1 pre-trained weights](https://github.com/Divadi/SOLOFusion/releases/download/v0.1.0/r50-fp16_phase2_ema.pth) for initialization.

**Usage:**
```bash
# Training
bash tools/dist_train.sh configs/lssinst/lssinst.py 8
```


### 

## Citation

If this work is helpful for your research, please consider citing our paper:

```bibtex
@article{lssinst,
  title={LSSInst: Improving Geometric Modeling in LSS-Based BEV Perception with Instance Representation},
  author={Weijie Ma and Jingwei Jiang and Yang Yang and Zehui Chen and Hao Chen},
  journal={International Conference on 3D Vision (3DV)},
  year={2025}
}
```
