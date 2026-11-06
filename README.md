# Open-Set Recognition with CLIP and Vision Transformers

Research internship project, NIT Calicut (May–July 2026), under the guidance of Dr. Sumesh T A, Department of Computer Science & Engineering.

## Overview

Standard image classifiers assume every test image belongs to a class seen during training. Open-set recognition (OSR) removes this assumption: the model must classify known classes correctly and reject images from unknown classes.

This project studies OSR with frozen foundation models:

- A ViT-B/16 baseline with a linear probe, scored with MSP and energy
- Zero-shot CLIP (ViT-B/32), scored with maximum similarity and entropy
- Prompt-calibrated CLIP: one learnable residual vector per class text embedding, with both CLIP encoders frozen
- A fused open-set score combining energy, cosine-similarity margin and distance to the nearest class centroid, with weights tuned on a validation set that never uses the test unknowns

## Results (AUROC)

| Method | CIFAR-10 | CIFAR-100 | TinyImageNet |
|---|---|---|---|
| ViT + MSP | 0.9172 | 0.8852 | 0.9527 |
| ViT + Energy | **0.9575** | **0.9075** | **0.9687** |
| CLIP + Similarity | 0.9153 | 0.8232 | 0.9357 |
| CLIP + Entropy | 0.9205 | 0.7560 | 0.8038 |
| Prompt CLIP + Fused | 0.9493 | 0.8767 | 0.9339 |

- The fused score outperforms zero-shot CLIP scoring on every dataset.
- The ViT linear probe with energy scoring is the strongest method overall.
- Prompt calibration improves CIFAR-10 known-class accuracy from 92.38% to 93.54%.

FPR@95TPR and OSCR results are in the Week 6 notebook.

## Splits

- CIFAR-10: known = airplane, automobile, cat, dog, horse; unknown = bird, deer, frog, ship, truck
- CIFAR-100: first 20 classes known, remaining 80 unknown
- TinyImageNet: first 20 classes known, remaining 180 unknown

## Notebooks

| Week | Notebook | Contents |
|---|---|---|
| 1 | [Week1_nit.ipynb](Week1_nit.ipynb) | Dataset setup and known/unknown split |
| 2 | [Week2_nit.ipynb](Week2_nit.ipynb) | Initial ViT baseline (MSP, energy) |
| 3 | [Week3_nit.ipynb](Week3_nit.ipynb) | CLIP zero-shot OSR |
| 4 | [Week4_nit.ipynb](Week4_nit.ipynb) | Prompt calibration on known classes |
| 5 | [Week5_nit.ipynb](Week5_nit.ipynb) | Fused score, weight tuning, ablation |
| 6 | [Week6_nit.ipynb](Week6_nit.ipynb) | Full evaluation on three datasets with AUROC, FPR@95TPR, OSCR |
| 7 | [Week7_nit.ipynb](Week7_nit.ipynb) | t-SNE, score distributions, failure-case analysis |

Note: the Week 2 baseline used the pretrained ImageNet head directly and gave below-chance AUROC. Week 6 replaces it with a linear probe on frozen features.

## Running

All notebooks run on Google Colab with a T4 GPU. Main libraries: PyTorch, torchvision, OpenCLIP, timm, scikit-learn.

## Paper

The full write-up is available here: [OSR_CLIP_ViT_Paper.pdf](OSR_CLIP_ViT_Paper.pdf)

Written in IEEE conference format as part of the NIT Calicut research internship (not published).
