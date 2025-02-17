# HyperMoE: Mixture of Spatial-Spectral Experts Network for Multispectral and Hyperspectral Image Fusion

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)]()
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

## 📖 Abstract
Fusing high-resolution multispectral images (HRMSI) with low-resolution hyperspectral images (LRHSI) to generate high-resolution hyperspectral images (HRHSI) is an efficient technique. However, existing networks often face modal conflicts when fusing local spatial features and global spectral features, which affects performance. To address this, we propose a Mixture of Experts (MoE) model for image fusion, called HyperMoE. HyperMoE effectively captures features at different modalities and levels through global sparse mixture of experts, generating higher quality images while significantly reducing computational and storage costs. Extensive experiments demonstrate that HyperMoE outperforms state-of-the-art methods in both quantitative metrics and visual quality. 

## 📦 Requirements
- Python 3.8+
- PyTorch 1.6+
- CUDA 11.1+

## 📂Dataset
- [Pavia University](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)
- [Chikusei](https://naotoyokoya.com/Download.html)
- [Houston](https://hyperspectral.ee.uh.edu/?page_id=459)

## 🛠️Usage
Place the dataset in the dataset directory, and run the following command:
```bash
python main.py 
```

## 🔍 Contact

If you have any questions or suggestions, please submit an Issue.
