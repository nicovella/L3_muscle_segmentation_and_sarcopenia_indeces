# L3_muscle_segmentation_and_sarcopenia_indeces
Deep learning pipeline for automatic clinical sarcopenia assessment from L3 CTs. A custom U-Net is used to segment abdominal and psoas muscles. The independent inference tool calculates physical cross-sectional area (cm2) and mean radiodensity (HU) from NIfTI meta-data and provides ready to use diagnostic text reports.


# Automated Sarcopenia Assessment via Custom U-Net Segmentation

## Project Overview
This repository contains a complete, end-to-end Deep Learning pipeline for the automated assessment of Sarcopenia from Computed Tomography (CT) scans. 

By leveraging a custom deep, fully-convolutional U-Net architecture, this tool autonomously segments the **Abdominal Muscle Wall (Class 1)** and **Psoas Muscles (Class 2 & 3)** at the L3/L4 vertebral levels. It bypasses manual, time-consuming radiological tracing to instantly compute cross-sectional areas (cm²) and mean radiodensities (Hounsfield Units - HU) required for the Skeletal Muscle Index (SMI).

## Key Features
* **Clinical-Grade Processing:** Native integration with `.nii` (NIfTI) files, preserving critical spatial metadata, affine matrices, and physical pixel spacing.
* **Radiological Windowing:** Automated Hounsfield Unit (HU) clipping to [-30, +150] to specifically isolate striated skeletal muscle tissue prior to neural network inference.
* **Physical Validation Metrics:** Evaluates model boundary accuracy using Balanced Average Hausdorff Distance (bAHD) and Maximum Hausdorff Distance (Max HD) calculated strictly in **millimeters (mm)**, rather than pixels.
* **Standalone Deployment:** Includes an isolated inference engine that requires no training dependencies, generating both a reconstructed NIfTI mask and an automated clinical `.txt` report.

---

## Repository Architecture

The project is structured into three distinct execution modules to strictly decouple data engineering, model training, and clinical deployment:

### 1. `01_patient_engineering_sanity_check.ipynb`
**Purpose:** Data Ingestion, Privacy, and Quality Assurance.
* Decompresses raw `.nii.gz` files.
* Anonymizes patient identifiers to comply with medical data privacy standards, creating a secure mapping registry.
* Audits all segmentation masks to ensure categorical integrity (strictly `[0, 1, 2]` labels) and prevents data corruption prior to training.

### 2. `02_model_training_and_evaluation.ipynb`
**Purpose:** Architecture, Training, and Physical Validation.
* Implements a custom Keras `Sequence` generator for memory-efficient, on-the-fly batching and HU windowing.
* Utilizes a Combined Objective Function (Sparse Categorical Crossentropy + custom Dice Loss) to combat extreme class imbalance.
* Performs quantitative validation in physical space (mm) and automatically searches the validation set for the "Worst-Case Scenario" to audit morphological failures.

### 3. `03_clinical_inference_and_reporting.ipynb` (Deployment)
**Purpose:** Lightweight inference for end-users and clinicians.
* Bypasses training configurations by loading pre-trained `.keras` weights directly.
* Takes an unseen `.nii` CT scan and predicts the muscle map.
* Resamples the prediction to native resolution, restores original NIfTI headers, and calculates the exact cross-sectional Area (cm²) and Density (HU).
* Exports the final mask and an automated Sarcopenia Assessment Text Report.

---

## Installation & Requirements

To run this pipeline locally, you must install the required Python packages. It is highly recommended to use a virtual environment (e.g., Anaconda or `venv`).

**Prerequisites:**
* Python 3.8+
* GPU support highly recommended for Module 02 (NVIDIA CUDA toolkit).

**Setup:**
```bash
# Clone the repository
git clone [https://github.com/YourUsername/sarcopenia-unet-assessment.git](https://github.com/YourUsername/sarcopenia-unet-assessment.git)
cd sarcopenia-unet-assessment

# Install required packages
pip install -r requirements.txt
