# Tactile Sensor Dataset Evaluator & Multi-Channel Training Pipeline

## Abstract
This repository provides a flexible, hardware-aware evaluation and training framework for vision-based tactile sensors (e.g., GelSight). 

Most computer vision pipelines blindly consume raw `RGB` images. However, in optical tactile sensors, RGB channels represent **3D lighting geometry** (Photometric Stereo), not just color. Depending on LED calibration, elastomer wear, manufacturing variations, or environmental Electromagnetic Interference (EMI), specific color channels may contribute more thermal noise than meaningful structural data. 

Instead of assuming `RGB` is optimal, this pipeline allows researchers to evaluate their dataset's hardware fidelity and run a **7-Channel Ablation Study** (`RGB`, `RG`, `GB`, `RB`, `R`, `G`, `B`) to identify the most robust configuration for *their specific sensor prototype*.

---

## 1. The Evaluator's Perspective: Why Channel Selection Matters

Before training, we strongly recommend evaluating the dataset's physical Signal-to-Noise Ratio (SNR) and Machine Learning Feature Separability (ANOVA F-Statistic). 

### Sample Evaluation Report (Case Study)
Below is an example output from a dataset evaluation run on our specific hardware prototype:

```text
[INFO] Scanning dataset directory: data/dataset ...
[INFO] Successfully processed 1921 images across 6 classes.

============================================================
1. Hardware Integrity: Dataset SNR (dB)
============================================================
 - R    Combination SNR :   1.36 dB
 - G    Combination SNR :   1.88 dB
 - B    Combination SNR :   0.87 dB
 - RG   Combination SNR :   1.59 dB
 - RB   Combination SNR :   1.09 dB
 - GB   Combination SNR :   1.28 dB
 - RGB  Combination SNR :   1.31 dB

============================================================
2. ML Feature Separability: ANOVA F-Statistic
============================================================
 - R    Combination F-Score :   202.75
 - G    Combination F-Score :   245.78
 - B    Combination F-Score :   157.46
 - RG   Combination F-Score :   224.27
 - RB   Combination F-Score :   180.10
 - GB   Combination F-Score :   201.62
 - RGB  Combination F-Score :   202.00

============================================================
Executive Summary
============================================================
 -> Highest SNR (Hardware Fidelity) : G Configuration
 -> Highest ML Separability (ANOVA) : G Configuration
 -> Lowest Performance (Noise Source) : B Configuration
 -> Recommendation: Mask out channels that heavily degrade the combined F-Score.
 ```

### Interpretation & Flexibility
In our specific case study, the **Blue (B)** channel exhibited severe noise degradation (0.87 dB SNR) and dragged down the overall `RGB` performance. Consequently, training the model exclusively on `G` or `RG` configurations yielded significantly higher physical robustness. 

**However, hardware varies.** Your sensor might have an impeccably calibrated Blue LED or a noisy Red channel due to ambient lighting conditions. This pipeline empowers you to test all combinations and select the optimal weights (`.pth`) that maximize the signal integrity of *your* unique setup.

---

## 2. Mathematical Foundation: The `ChannelMasker` Trap

If your evaluation suggests masking a specific channel (e.g., forcing Blue to zero), you cannot simply set the raw image pixels to `0` before passing it to a standard Convolutional Neural Network (CNN). 

Standard deep learning pipelines utilize ImageNet Normalization: Z = (X - μ) / σ. If we aggressively zero out the raw Blue channel (X_B = 0) *before* normalization, the CNN receives: Z_B = (0 - 0.406) / 0.225 ≈ -1.804.

To a CNN, -1.804 is not "empty space"; it is a **massive negative activation surge**. By trying to remove the noise, we accidentally inject artificial high-frequency artifacts. 

**Our Implementation:** The `ChannelMasker` in our training script (`src/train_models.py`) is strategically applied at the *very end* of the preprocessing pipeline. We apply normalization first, and then explicitly force the normalized tensor to 0.0. This mathematically guarantees that the CNN filters receive an absolute zero, perfectly simulating a physical optical filter.

---

## 3. Quick Start & Execution Guide (Google Colab)

To efficiently train all 7 configurations sequentially, we recommend executing the training script on a cloud GPU (e.g., Google Colab T4).

### Step 1: Prepare Your Data
Compress your tactile dataset into a file named `dataset.zip` and upload it to the root of your Google Drive (`/MyDrive/dataset.zip`).

### Step 2: Colab Execution
1. Open a new Google Colab notebook and set the runtime to **GPU (T4)**.
2. Upload and execute the `src/train_models.py` script in your Colab environment.
3. The script will automatically:
   * Mount your Google Drive and verify the path.
   * Extract the dataset to the local NVMe SSD (`/content/dataset`) for maximum I/O speed.
   * Auto-correct nested folder structures if they exist.
   * Train 7 independent ResNet-18 models.
   * Generate Confusion Matrices and a Comparative Bar Chart (`accuracy_comparison_bar.png`).
4. Review the generated bar chart and select the best `.pth` weight for your deployment.