# Tactile-EMI-Monitor 
**Hardware Signal Integrity and Vulnerability Analysis Framework for Optical Tactile Sensors**

## Project Overview
Optical tactile sensors (e.g., GelSight Mini) convert physical deformations into visual data for Convolutional Neural Network (CNN) inference. While CNNs demonstrate robustness against certain spatial noise, the underlying mixed-signal hardware (CMOS pixel arrays, Row-ADCs, and USB UVC protocols) remains physically susceptible to Electromagnetic Interference (EMI) and thermal drift.

This repository provides an analytical framework to evaluate these physical-layer vulnerabilities. By integrating Digital Signal Processing (DSP) metrics with CNN feature analysis, this project offers an end-to-end pipeline—from signal-aware data collection to real-time degradation logging—to quantitatively assess how hardware-level RF interference impacts predictions.

---

## System Architecture & Pipeline

This repository consists of four core modules designed for signal analysis:

### 1. Physics-Grounded Data Collection (`src/data_collector.py`)
A data collection tool that utilizes standard image processing (Gaussian blur and absolute difference) to implement a threshold-based contact mask. It strictly enforces lossless `.png` storage to preserve raw analog noise characteristics and provides real-time RGB channel monitoring during data acquisition. *(See `docs/DATA_COLLECTOR.md`)*

### 2. Hardware Fidelity Evaluator (`src/dataset_evaluator.py`)
A statistical evaluation script to quantify the raw hardware data quality prior to CNN training. By calculating the **Hardware Signal-to-Noise Ratio (SNR)** and **ANOVA F-Statistic** across individual RGB channels, researchers can mathematically identify hardware-level noise distribution to inform dynamic channel masking strategies. *(See `docs/EVALUATOR.md`)*

### 3. Dynamic Channel Ablation Training (`src/train_models.py`)
A PyTorch training pipeline that supports a 7-channel ablation study (`RGB`, `RG`, `GB`, `RB`, `R`, `G`, `B`). It integrates a custom `ChannelMasker` that strictly zeroes out specific channels *after* ImageNet normalization. This ensures the CNN receives a mathematically neutral zero without injecting artificial negative activation surges.

### 4. Inference & EMI Degradation Monitor (`src/detector_v1` to `v4`)
The real-time inference system is divided into sequential versions to isolate software defenses and observe hardware failures:
*   **V1 (Baseline):** Implements transient state filtering (debouncing) and RG-channel masking for stable semantic inference.
*   **V2 (Vulnerability Logger):** Disables debouncing to expose unfiltered analog anomalies and observe instantaneous voltage transients.
*   **V3 (DSP Monitor):** Introduces statistical metrics (Global Variance and 2D-FFT High-Frequency Ratio) to analyze spatial noise distributions.
*   **V4 (Deep Feature & CMOS Monitor):** An optimized, single-pass inference monitor. It simultaneously logs AI feature space shifts (Cosine Distance) and image-level degradation (SSIM Loss, Total Variation, Shannon Entropy) to record the exact trajectory of a hardware crash. *(See `docs/EVOLUTION.md`)*

---

## Key Experimental Findings

Our empirical EMI injection tests—specifically conducted within the **115 MHz to 131 MHz** frequency band, with **125 MHz** demonstrating the highest reproducibility for these vulnerabilities—revealed the following factual hardware limitations:

1.  **USB PHY Failure (Hardware DoS):** 
    The most dominant failure mode under radiated EMI is the disruption of the USB physical layer. High-power RF injection corrupts the signaling, causing a CRC failure and triggering an immediate hardware disconnect (Denial of Service) before targeted semantic spoofing can occur.
2.  **CMOS Saturation (Sensor Blinding / White-Out):** 
    In specific high-power EMI scenarios prior to USB failure, the induced RF energy severely disrupts the sensor's internal circuitry (e.g., corrupting analog gain/exposure registers or data bus logic). This results in the pixel array outputting completely white, overexposed frames. This total destruction of physical tactile features forces the CNN to output chaotic, random misclassifications.
3.  **Exploratory Logging Architecture:** 
    Consistently recording the exact mathematical metrics of the white-out transition proved highly challenging due to physical setup constraints (e.g., critical antenna positioning angles and power amplifier energy limitations triggering premature USB crashes). To enable future precision measurements, we established the **V4 framework (`detector_v4.py`)**. This architecture is designed to capture these catastrophic events mathematically—aiming to log the severe drop in Shannon Entropy (total loss of structural information) and the massive spike in Cosine Distance within the ResNet-18 feature space when sensor blinding occurs.

---

## Installation & Setup

We provide a cross-platform Python script to automatically create an isolated virtual environment and install the required dependencies (PyTorch, OpenCV, SciPy, etc.).

### 1. Run the Auto-Setup Script
Open your terminal or command prompt in the project root directory and execute:
```bash
python setup_env.py
```
*(This creates a virtual environment named `tactile_env` and installs packages from `requirements.txt`.)*

### 2. Activate the Virtual Environment
You must activate the virtual environment before running any scripts:
*   **Linux / macOS:** 
    ```bash
    source tactile_env/bin/activate
    ```
*   **Windows (Command Prompt):**
    ```cmd
    tactile_env\Scripts\activate
    ```

### 3. Hardware SDK Requirement
**[CRITICAL]:** The standard GelSight driver folder (`gs_sdk/`) must be present in the root directory of this project for the scripts to establish a USB connection with the hardware.

---

## 📂 Experimental Data & Media

To reproduce our results or observe the physical EMI phenomena, please access our comprehensive **https://drive.google.com/drive/folders/1iBuSMDTVZGinFxqeh6bHqhKnltIhUgqo?usp=drive_link**. 

This cloud directory contains the following supplementary materials:
*   **`dataset/`**: The raw, uncompressed tactile image dataset (`.zip`) used for training and hardware signal integrity (SNR/ANOVA) evaluation.
*   **`emi_failure_modes/`**: Direct visual evidence of the hardware vulnerabilities, including the "Sensor Blinding" (white-out) frames and a video recording of the USB PHY crash (Denial of Service).
*   **`antenna_setup/`**: Reference images of the physical RF injection environment, detailing antenna positioning and distance relative to the sensor.

---

## Quick Start

Once the environment is active and the sensor is connected, you can run the following modules:

**1. Evaluate dataset hardware signal integrity:**
```bash
python src/dataset_evaluator.py
```

**2. Run the baseline robust inference (V1):**
```bash
python src/detector_v1.py
```

**3. Run the Deep Feature & DSP Degradation Monitor (V4):**
```bash
python src/detector_v4.py
```
*UI Instructions for V4: Press `[l]` to lock onto a baseline target, then `[e]` to begin logging EMI degradation metrics to a CSV file.*
