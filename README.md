# PASD: 3D Segment Anything Model with Visual Mamba for Diagnosing Placenta Accreta Spectrum

This repository hosts the official PyTorch implementation of **3DSAMba**, a deep learning framework
for MRI-based diagnosis of Placenta Accreta Spectrum (PAS). The project is named **PASD**
(Placenta Accreta Spectrum Diagnosis).

> **Paper**: *3D Segment Anything Model with Visual Mamba for Diagnosing Placenta Accreta Spectrum*
> (IEEE Transactions on Image Processing)

## Highlights

- **First MRI-based PAS dataset** with both segmentation and classification annotations.
- **3D SAM backbone** with an efficient adapter mechanism for the medical domain.
- **Multi-Level Fusion Mamba (MLFM)** to merge feature maps across hierarchical levels.
- **Fusion State Space Model (FSSM)** to integrate multi-scale encoder/decoder features.
- A two-stage pipeline that uses predicted lesion masks to refine PAS classification.

## Pipeline Overview

```
       MRI Volume
           |
           v
   +-------+--------+              +-------------------------+
   |  3D LoRA-SAM   | ----------> |  MLFM + FSSM Decoder    |
   |  (image enc.)  |              |  (segmentation head)    |
   +----------------+              +-----------+-------------+
                                                |
                              segmentation mask v
                                                |
              MRI x mask  ----> +---------------+---------------+
                                |    Conv3D Classifier (PAS)    |
                                +-------------------------------+
```

## Repository Structure

```
PASD/
├── segment_anything/             # Modified SAM (forward signature adapted to 3D + adapter)
├── networks/
│   └── unetr.py                  # UNETR backbone used as utility
├── decoder.py                    # PASD decoder with MLFM / FSSM modules
├── class_net.py                  # 3D classifier head (Conv3DNet)
├── sam_lora_image_encoder.py     # LoRA wrapper around SAM image encoder
├── vmamba.py                     # VSSBlock (forward / backward SSM)
├── vmamba2.py                    # VSSBlock2 / VSSBlock3 (multi-scale fusion)
├── vmamba_class.py               # Mamba-based classifier backbone (optional)
├── selective_scan.py             # Pure-PyTorch selective scan fallback
├── selective_scan_cuda_core.py   # CUDA selective scan shim
├── dataset.py                    # Segmentation dataset loader
├── dataset_class.py              # Classification dataset loader (uses masks)
├── test_score.py                 # Metric helpers (Dice, IoU, HD95, etc.)
├── train_seg.py                  # Train segmentation model
├── train_class.py                # Train classifier on masked MRI
├── test_seg.py                   # Evaluate segmentation
└── test_class.py                 # Evaluate classification
```

## Environment

Recommended setup:

- Python 3.10+
- CUDA 11.8 / 12.x
- PyTorch 2.0+

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

If you have an Ampere or newer GPU, you may additionally install the official
`mamba_ssm` / `causal_conv1d` packages for faster selective scans. The repo
falls back to a pure-PyTorch implementation when those kernels are unavailable.

## Data Preparation

The default loaders expect the PAS dataset to be organised as:

```
train/                            # training cases
├── 001-<case-name>-tra-<label>/  # the trailing digit encodes the class label
│   ├── <case>.nii.gz             # MRI volume (>= 18-char filename)
│   └── <case>-mask.nii.gz        # GT segmentation mask
├── 002-...
test/                             # test cases (same layout)
test_other/                       # predicted masks used by dataset_class.py
```

Volumes are resampled to `48 x 256 x 256` and intensity-clipped to `[0, 1650]`
before normalisation. See `dataset.py` / `dataset_class.py` for details.

## Pre-trained Weights

The pre-trained checkpoints (SAM ViT-B and PASD segmentation / classification
heads) are **not** bundled in this repository because of size constraints.
You will need:

| File                          | Purpose                                  |
| ----------------------------- | ---------------------------------------- |
| `sam_vit_b_01ec64.pth`        | SAM ViT-B weights (~375 MB)              |
| `samba_2d_fusion_hyper.pth`   | PASD segmentation weights                |
| `class_unetr.pth`             | PASD classification head weights         |

Download `sam_vit_b_01ec64.pth` from the
[official SAM release](https://github.com/facebookresearch/segment-anything#model-checkpoints)
and place it at the repository root. The PASD weights will be released separately;
please contact the authors if you need them in advance.

## Training

Train the segmentation model:

```bash
python train_seg.py
```

Train the classification head on masked MRI:

```bash
python train_class.py
```

## Evaluation

Segmentation evaluation reports Dice, IoU, accuracy, specificity and sensitivity:

```bash
python test_seg.py
```

Classification evaluation reports overall accuracy:

```bash
python test_class.py
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhang2025pasd,
  title   = {3D Segment Anything Model with Visual Mamba for Diagnosing Placenta Accreta Spectrum},
  author  = {Zhang, Yuliang and He, Fang and Peng, Lulu and Guo, Qing and Yu, Lin and
             Wang, Zhijian and Shun, Wei and Liu, Jue and Chen, Yonglu and Huang, Jianwei and
             Bao, Zeye and Cai, Zhishan and Chen, Yanhong and Hu, Miao and Gu, Zhongjia and
             Shi, Yiyu and Yan, Tianyu and Zhang, Pingping and Ting, Song and Du, Lili and Chen, Dunjin},
  journal = {IEEE Transactions on Image Processing},
  year    = {2025}
}
```

## Acknowledgements

This repository builds on top of:

- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [VMamba](https://github.com/MzeroMiko/VMamba)
- [MONAI](https://github.com/Project-MONAI/MONAI)

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
