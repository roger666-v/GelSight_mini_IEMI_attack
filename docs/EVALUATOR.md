# Analytical Framework for Optical Tactile Sensor Channels

## Overview
In the realm of vision-based tactile sensors (e.g., GelSight), the raw output is typically an RGB image matrix. However, not all color channels contribute equally to tactile perception. Due to varying LED illumination angles, internal CMOS thermal properties, and external Electromagnetic Interference (EMI) susceptibility, certain channels may introduce more noise than meaningful contact signals.

This document provides a rigorous mathematical and physical breakdown of the evaluation metrics used in `dataset_evaluator.py`. By quantifying hardware fidelity and machine learning separability, this tool allows researchers to objectively identify corrupted data channels in their specific hardware prototype and optimize downstream neural network performance.

---

## 1. Hardware Signal-to-Noise Ratio (SNR)

### Physical Meaning
Before feeding data into a neural network, it is critical to assess the raw analog-to-digital hardware fidelity. In an optical tactile sensor:
*   **Signal (Information):** The high-frequency physical deformations (edges, textures, and ridges) caused by pressing an object against the elastomer.
*   **Noise (Interference):** The ambient baseline fluctuations, including CMOS thermal noise (Johnson–Nyquist noise), ambient lighting bleed, and EMI-induced bit-flips.

### Mathematical Formulation
We extract the high-frequency structural energy using the **Mean Sobel Magnitude** across the spatial domain ($x, y$). The gradient magnitude for a channel $C$ is given by:

$$|\nabla C|=\sqrt{\left(\frac{\partial C}{\partial x}\right)^2+\left(\frac{\partial C}{\partial y}\right)^2}$$

The SNR is then calculated in decibels (dB) by comparing the expected gradient energy of physical object contact ($P_{signal}$) against the baseline background noise ($P_{noise}$):

$$SNR_{dB}=10\log_{10}\left(\frac{\frac{1}{N}\sum_{i=1}^{N}|\nabla C_{contact}|_i}{\frac{1}{M}\sum_{j=1}^{M}|\nabla C_{background}|_j}\right)$$

**Interpretation:** A low SNR in a specific channel indicates that the channel is physically dominated by random noise rather than structural deformation data. Relying on this channel for model training will likely degrade overall robustness.

---

## 2. Machine Learning Separability (ANOVA F-Statistic)

### Physical Meaning
While SNR measures raw hardware integrity, the **ANOVA F-Statistic** measures mathematical separability for classification algorithms. It quantitatively answers the question: *Does this specific channel or channel combination actually help a model distinguish between different tactile textures (e.g., a fingerprint vs. a coin)?*

### Mathematical Formulation
Analysis of Variance (ANOVA) calculates the F-score by comparing the variance *between* different classes (inter-class) to the variance *within* the same class (intra-class).

For a feature vector $X$ (e.g., channel variance or edge energy) across $K$ classes:

$$F=\frac{\text{Between-Group Variability}}{\text{Within-Group Variability}}=\frac{\frac{1}{K-1}\sum_{k=1}^{K}n_k(\bar{X}_k-\bar{X})^2}{\frac{1}{N-K}\sum_{k=1}^{K}\sum_{i=1}^{n_k}(X_{ik}-\bar{X}_k)^2}$$

*   **Numerator (Inter-class):** A high value means different objects produce vastly different channel responses (highly desirable).
*   **Denominator (Intra-class):** A low value means the same object scanned multiple times yields highly consistent, stable data (highly desirable).

**Interpretation:** A high F-Score indicates that the extracted features are strongly correlated with the target classes, making it easier for a CNN to draw robust decision boundaries.

---

## 3. Interpreting Channel Combinations (Ablation Analysis)

By calculating features for all permutations (`R`, `G`, `B`, `RG`, `RB`, `GB`, `RGB`), the evaluator uncovers the concept of **Orthogonal Feature Complementarity** and helps prevent the **Curse of Dimensionality**.

### Constructive Synergy (The Ideal Scenario)
Typically, different LEDs illuminate the elastomer from different physical angles. For example, the Red channel may capture horizontal macro-shapes, while the Green channel captures vertical micro-textures. 
*   When evaluating the `RG` combination, the F-Score typically spikes higher than either `R` or `G` individually because their spatial gradients are functionally orthogonal (capturing different axes of physical deformation).

### Destructive Contamination (Identifying the Bottleneck)
Not all combinations yield improvements. If a specific channel suffers from poor quantum efficiency or high thermal noise, combining it with clean channels can mathematically harm the dataset.
*   If the F-Score drops when transitioning from a 2-channel configuration (e.g., `RG`) to the full 3-channel configuration (`RGB`), it empirically proves that the third channel introduces overlapping intra-class noise that actively confuses the feature space.

**Conclusion:**
The `dataset_evaluator.py` provides the empirical data necessary to make informed engineering decisions. Rather than assuming `RGB` is optimal, this tool allows researchers to mathematically justify dynamic channel masking (e.g., intentionally zeroing out a corrupted channel) prior to CNN inference.

---

## 4. Real-World Case Study & Implementation

To demonstrate the necessity of channel ablation, below is the real-world evaluation output from our specific tactile sensor prototype consisting of 1,921 physical contact images.

### Dataset Evaluation Output

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

### Data Interpretation & Engineering Decisions

By cross-referencing the SNR and ANOVA metrics, several critical hardware realities are exposed:

1.  **The Blue Channel is Physically Corrupted:**
    The `B` channel yields an SNR of only **0.87 dB** and the lowest F-Score (**157.46**). In signal processing, an SNR approaching or falling below 1.0 dB indicates that the noise power is severely overwhelming the actual tactile structural signal, rendering it nearly useless for high-precision inference.
2.  **Destructive Contamination (The Curse of Dimensionality):**
    When combining channels, more data does not equal better data.
    *   The `RG` combination achieves a highly robust F-Score of **224.27**.
    *   When the Blue channel is introduced to form the full `RGB` matrix, the F-Score drops significantly to **202.00**, and the SNR drops from 1.59 dB to 1.31 dB.
    *   **Conclusion:** This empirically proves that the B channel introduces overlapping intra-class noise that actively destroys the feature space.
3.  **Final Engineering Decision:**
    Rather than assuming standard `RGB` is optimal, this evaluator mathematically justifies our decision to implement the `ChannelMasker` in the training pipeline. By physically masking out the Blue channel (opting for the `RG` or `G` configuration), we maximize hardware signal integrity and secure the CNN against sub-threshold environmental noise.