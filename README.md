# Deep Learning Projects

![Python](https://img.shields.io/badge/Python-3.x-blue) ![Framework](https://img.shields.io/badge/Framework-PyTorch-orange) ![Environment](https://img.shields.io/badge/Environment-Google%20Colab-yellow) ![Status](https://img.shields.io/badge/Status-Portfolio%20organization-informational)

A curated collection of seven hands-on deep learning course projects, covering computer vision, medical image segmentation, object detection, image captioning, and image super-resolution. Many experiments were developed on **Google Colab with GPU acceleration**.

## Project gallery

| Project | Area | Highlights |
|:--|:--|:--|
| **[01. CIFAR-10 Image Classification](01-cifar10-image-classification/README.md)** | Image classification and foundational model experiments | 2 notebooks, 0 Python source files |
| **[02. Building a Multilayer Perceptron](02-mlp-from-scratch/README.md)** | Neural network components and backpropagation | 1 notebooks, 0 Python source files |
| **[03. PyTorch in Practice](03-pytorch-practice/README.md)** | Applied PyTorch experiments | 1 notebooks, 0 Python source files |
| **[04. Brain MRI Image Segmentation](04-brain-mri-segmentation/README.md)** | Medical image segmentation and U-Net experiments | 7 notebooks, 11 Python source files |
| **[05. Traffic Scene Object Detection](05-traffic-object-detection/README.md)** | YOLO and RT-DETR experiments on traffic scenes | 9 notebooks, 3 Python source files |
| **[06. Image Caption Generation](06-image-captioning/README.md)** | Visual-language modeling on Flickr8k | 7 notebooks, 11 Python source files |
| **[07. Image Super-Resolution](07-image-super-resolution/README.md)** | SRResNet and SRGAN image reconstruction | 2 notebooks, 14 Python source files |

## Repository layout

```text
deep-learning-projects/
├── README.md
├── 01-cifar10-image-classification/
├── 02-mlp-from-scratch/
├── 03-pytorch-practice/
├── 04-brain-mri-segmentation/
├── 05-traffic-object-detection/
├── 06-image-captioning/
└── 07-image-super-resolution/
```

Each project contains README.md, notebooks/, src/, assets/, and (where applicable) archive/ supporting text files. Original notebooks and PDF reports are provided separately.
```

## How to explore

1. Select a project and read its `README.md`.
2. Browse the prepared `notebooks/` for the workflow and saved outputs.
3. Inspect `src/` for standalone modules, including code created using Colab's `%%writefile`.
4. Open a notebook in Google Colab if you want to reproduce it; update paths, dependencies, and runtime settings first.

## Engineering choices

- **Keep notebooks** for exploratory experiments, narratives, visualizations, and Colab execution.
- **Expose `.py` modules** for reusable training, datasets, model definitions, and evaluation code.
- **Preserve original notebooks** in the separate original backup ZIP; project source and conclusions are not silently rewritten.
- **Avoid large datasets and model weights in Git**; retrieve data separately and consider Git LFS or external storage if checkpoints must be shared.

## Reproducibility and documentation status

This repository is an organized portfolio of submitted experiments, **not a verified fully reproducible software release**. Some notebook cells rely on Google Drive mounts, hard-coded paths, install commands, or specific GPU runtimes. The project pages describe the available source. Metrics and visuals, where present, come from saved notebook outputs and have not been rerun.

### Translation policy

The README files are written in English. The prepared notebooks provide English high-level replacements for Chinese markdown cells and translate some safe, standalone Chinese code comments. A full manual line-by-line translation and verification of figures, string literals, and explanatory derivations remains a future refinement; the untouched originals are provided in the separate backup ZIP.

## Attribution

These projects were completed as coursework. Dataset and third-party model licenses belong to their respective owners; course origins are not represented as independently commissioned research.

### Original backup

The original `.ipynb` notebooks and Chinese PDF reports are supplied in a **separate backup ZIP** rather than duplicating very large embedded outputs inside the public Git repository. Some large notebook outputs in the prepared versions are stripped to keep uploads manageable; selected figures are retained in each project's `assets/` directory.

**Source cleanup:** Leading Colab `%%writefile` directives were removed from exported `.py` files; standalone modules passed Python syntax parsing. Execution and full English translation remain unverified.
