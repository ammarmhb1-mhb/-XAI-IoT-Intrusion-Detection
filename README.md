# XAI for IoT Intrusion Detection on Edge Devices

Code and datasets for the paper:

**"Quantifying the Cost of Interpretability: XAI for IoT Intrusion Detection on Edge Devices"**

## Overview

This repository contains the complete experimental code for evaluating the computational cost of XAI methods (TreeSHAP, KernelSHAP, GradientSHAP, LIME) on IoT intrusion detection systems using the CICIoT2023 dataset.

## Contents

| File | Description |
|---|---|
| `code.py` | Complete experimental code (3 platforms: GPU, CPU, CPU-Limited) |
| `results_GPU.json` | Measurements on GPU platform (if available) |
| `results_CPU.json` | Measurements on CPU platform (if available) |
| `results_CPU_LIMITED.json` | Measurements on CPU-Limited platform (if available) |
| `explanation_quality.json` | Explanation quality metrics (if available) |

## Requirements

- Python 3.13
- XGBoost 2.0.0
- SHAP 0.44.0
- TensorFlow 2.13.0
- scikit-learn 1.3.0
- NumPy, pandas, scipy

## Dataset

CICIoT2023: [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/iotdataset-2023.html)

The dataset is loaded automatically from Hugging Face in the code.

## How to Run

1. Open `code.py` in Google Colab
2. Change `EXPERIMENT_MODE` to one of:
   - `"GPU"` — for GPU platform
   - `"CPU"` — for CPU platform
   - `"CPU_LIMITED"` — for CPU-Limited (edge emulation)
3. Run the code
4. Results are saved as `results_ciciot2023_{MODE}.json`

## Experimental Setup

- **Models**: XGBoost, CNN
- **XAI Methods**: TreeSHAP, KernelSHAP, GradientSHAP, LIME
- **Platforms**: GPU (T4), CPU, CPU-Limited (25% speed, 2 cores)
- **Measurements**: Detection accuracy, latency, memory, energy, explanation quality

## Citation

If you use this code, please cite the paper.

## Contact

Ammar Majeed Hameed Bedear — ammarmhb1@gmail.com
