<br />
<p align="center">
  <h1 align="center">3DMIT: 3D Multi-modal Instruction Tuning for Scene Understanding </h1>
  <p align="center">
    <a href="https://staymylove.github.io">Zeju Li</a>, Chao Zhang, Xiaoyan Wang, Ruilong Ren, Yifan Xu, Ruifei Ma, Xiangde Liu
  </p>
  <p align="center">
    <a href='https://arxiv.org/abs/2401.03201'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
    </a>
   
  </p>
  <p align="center">
    <img src="figs/overview.png" alt="Logo" width="80%">
  </p>
</p>

# Description

Official implementation of the paper: 3DMIT: 3D Multi-modal Instruction Tuning for Scene Understanding

# Setup

To set up the environment, run the following commands:

```bash
conda create -n 3dmit python==3.10.13
conda activate 3dmit
```

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```
# Citation
If you find our work useful, please consider citing:
```
@misc{li20243dmit,
      title={3DMIT: 3D Multi-modal Instruction Tuning for Scene Understanding}, 
      author={Zeju Li and Chao Zhang and Xiaoyan Wang and Ruilong Ren and Yifan Xu and Ruifei Ma and Xiangde Liu},
      year={2024},
      eprint={2401.03201},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


# Acknowledge
Our based code: https://github.com/OpenGVLab/LAMM
https://github.com/Chat-3D/Chat-3D-v2
https://github.com/Chat-3D/Chat-3D
https://github.com/baaivision/Uni3D
