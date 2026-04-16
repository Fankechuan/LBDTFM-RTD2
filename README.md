# Lightweight Bimodal Visual Detection Framework for Transmission Line Defects

This repository provides the official PyTorch implementation of the paper:

**"Lightweight Bimodal Visual Detection Framework for Transmission Line Defects in Complex UAV Inspection Environments"**
Submitted to *The Visual Computer*.

---

## 1. Overview

This work proposes a lightweight bimodal visual detection framework for UAV-based transmission line inspection. The framework addresses challenges in small object detection, multimodal discrepancy, and edge deployment.

The core idea is to **decouple modality-specific detection and fuse results at the decision level**, ensuring both efficiency and robustness.

---

## 2. Method Description (Key Algorithms)

The proposed framework consists of three main components:

### 2.1 Modality Classification (EfficientNetV2-S)

* Input UAV images are first classified into:

  * Infrared thermal images
  * Visible-light insulator images
  * Visible-light foreign object images
* This avoids unnecessary computation and routes data to the optimal detection branch.

---

### 2.2 Dual-Branch Detection

#### Visible Branch

* Backbone: GhostNetV2
* Detector: RT-DETRv2
* Enhancements:

  * SE Attention
  * Improved feature extraction for small targets and cluttered backgrounds

#### Infrared Branch

* Backbone: ResNet-18
* Detector: RT-DETRv2
* Focus:

  * High-sensitivity detection of thermal anomalies (overheating)

---

### 2.3 Decision-Level Fusion

* Activated only when paired bimodal data are available
* Combines:

  * Structural features (visible)
  * Thermal features (infrared)
* Reduces missed detections compared to single-modality approaches

---

## 3. Requirements

Tested environment:

* Python >= 3.8
* PyTorch >= 2.0.1
* torchvision >= 0.15.2

Additional dependencies:

```bash
torch>=2.0.1
torchvision>=0.15.2
faster-coco-eval>=1.6.5
PyYAML
tensorboard
scipy
pycocotools
onnx
onnxruntime-gpu
tensorrt==8.5.2.2
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 4. Usage

### 4.1 Training

Train the LBDTFM-RTD2 from scratch or using pre-trained weights:

```bash
python train.py --config configs/train.yaml
```

---

### 4.2 Evaluation

```bash
python eval.py --weights weights/best.pth
```

---

### 4.3 Inference

```bash
python infer.py --source demo.jpg
```

---

## 5. Reproducibility

In accordance with the reproducible research principles required by *The Visual Computer*, we provide the complete source code, configuration files, and dataset references.

All experiments reported in the paper can be reproduced by:

* Using the provided configuration files in `/configs`
* Following the training and evaluation commands above
* Accessing the public datasets listed in the repository

This repository is directly associated with the manuscript currently under review at *The Visual Computer*.
Researchers are encouraged to cite the corresponding paper when using this code.

---

## 6. Dataset

We use:

* Self-constructed UAV bimodal dataset (visible + infrared)
* Public datasets:

  * https://aistudio.baidu.com/datasetdetail/185520
  * https://aistudio.baidu.com/datasetdetail/310490

---



## 7. Citation

If you find this work useful, please cite:

```bibtex
@article{yourname2026bimodal,
  title={Lightweight Bimodal Visual Detection Framework for Transmission Line Defects},
  journal={The Visual Computer},
  year={2026}
}
```

---

## ⚠️ Note

This repository is directly related to a manuscript currently submitted to *The Visual Computer*.
Please cite the corresponding paper when using this code.
