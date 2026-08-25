# Evolution of the Tactile Inference System: From Robustness to Vulnerability Detection

## 1. Overview
This document outlines the four-phase evolutionary framework (V1 → V2 → V3 → V4) designed to investigate the **Analog Signal Integrity and Hardware Vulnerabilities** of optical tactile sensors under Electromagnetic Interference (EMI).

Empirical testing revealed a harsh hardware reality: the sensor does not gracefully degrade under EMI. Across all versions, the dominant failure mode is a catastrophic hardware shutdown (USB PHY crash / Denial of Service). However, in several instances, severe RF injection triggered a **"Sensor Blinding"** effect—saturating the CMOS array to pure white and forcing the AI into random misclassifications right before the crash.

This progression documents our transition from building a robust AI baseline, to observing these catastrophic analog failures, and finally establishing a high-dimensional mathematical logging framework (V4) aimed at future research into targeted semantic spoofing. 

To bridge the gap between deep learning and hardware physics, we include explicit explanations of the physical and mathematical mechanisms underlying our metrics.

---

## 2. Phase 1: The Industrial Baseline (`detector_v1.py`)

**Goal:** To establish a highly robust, real-time semantic inference engine immune to transient motion and environmental thermal noise.

### Architectural Features & The Math:
*   **The Normalization Trap & Channel Masking:**
    Optical tactile sensors extract physical shapes using colored LEDs, but the Blue (B) channel is notoriously noisy. We must mute it, but *when* we apply this mask mathematically dictates the outcome.
    Standard AI preprocessing uses ImageNet normalization. Given an input matrix $X$, mean $\mu$, and standard deviation $\sigma$:
    $$Z = \frac{X - \mu}{\sigma}$$
    >  **Mathematical Mechanism:** If we mathematically force the Blue channel to zero ($X_B = 0$) *before* normalization, the formula subtracts the dataset's mean ($\mu$) from 0, resulting in a large negative value ($-\mu / \sigma$). The Convolutional Neural Network (CNN) will treat this large negative number as a strong, artificial signal (an anomaly). By applying the mask *after* normalization, we ensure the matrix value is absolute zero, which the CNN correctly processes as "no information."
*   **Transient State Filtering (Debounce):**
    V1 uses `STABLE_FRAME_THRESHOLD = 3` to discard unstable frames, triggering inference only when physical motion stops.
*   **The Catastrophic Blindspot:**
    While mathematically robust, V1 is completely defenseless against EMI. When RF power hits a threshold, the underlying USB controller crashes. The software's debounce logic becomes irrelevant because the hardware suffers a sudden Denial of Service (DoS).

---

## 3. Phase 2: Exposing Macroscopic Vulnerability (`detector_v2.py`)

**Goal:** To strip away software defenses (debouncing) to observe the unfiltered analog effects of EMI on the sensor before it crashes.

### Real-World Observations & Physical Physics:
*   **The "Digital Cliff" (Hardware Shutdown):**
    Unlike analog systems that gradually degrade with noise, digital communication systems (like USB) fall off a "Digital Cliff." The sensor ignores low-power RF, but once a critical voltage threshold is crossed, the differential signaling (D+/D-) is corrupted. The CRC (Cyclic Redundancy Check) fails repeatedly, forcing the hardware to instantly disconnect and shut down.
*   **The "Sensor Blinding" Phenomenon (White-Out Effect):**
    Before the USB communication crashes, intense EMI can couple directly into the sensor's Analog-to-Digital Converters (ADCs). 
    > **Physical Mechanism:** EMI injects excess electromagnetic energy into the sensor's internal circuits. This induced voltage raises the analog signal level beyond the ADC's maximum readable limit (saturation). When the ADC is saturated, it outputs the maximum digital value (e.g., 255 in an 8-bit system) across the entire pixel array, resulting in a completely white image. Because the AI's final classification layer (Softmax) mathematically *must* output probabilities that sum to 100%, it is forced to calculate probabilities from this featureless data, causing chaotic misclassification.

---

## 4. Phase 3: Statistical Signal Processing (`detector_v3.py`)

**Goal:** To use Digital Signal Processing (DSP) mathematics to detect "sub-threshold" RF injection (early warnings) before the system falls off the Digital Cliff.

### The Math & Mechanics:
V3 uses statistical metrics to flag corrupted frames independently of the AI:

*   **Metric 1: Difference Variance ($\sigma^2$)**
    $$\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2$$
    > **Mathematical Mechanism:** Variance measures how far a set of numbers spreads out from their average. In a stable physical state, pixel intensities fluctuate very little, keeping variance near zero. When EMI couples into the circuit, it induces random voltage fluctuations across thousands of pixels simultaneously. Even if each pixel only fluctuates by 1 or 2 digital units, squaring and summing these deviations across the entire sensor array mathematically amplifies this distributed noise, making it highly detectable.
*   **Metric 2: 2D FFT High-Frequency Energy Ratio ($HF_{ratio}$)**
    $$HF_{ratio} = \left( \frac{\sum_{(u,v) \in \Omega_{HF}} |F(u,v)|^2}{\sum_{\text{all } u,v} |F(u,v)|^2} \right) \times 100\%$$
    > **Physical Mechanism:** The Fast Fourier Transform (FFT) converts spatial images into a frequency domain. Physical objects pressed against the sensor (like a finger or coin) create gradual color gradients, which mathematically correspond to *low-frequency* signals. In contrast, EMI noise causes sharp, pixel-to-pixel voltage jumps, representing *high-frequency* signals. By calculating the ratio of high-frequency energy to total energy, we can mathematically separate genuine physical deformation from electronic interference.

**The Empirical Reality:** In practice, V3 proved that this specific commercial hardware lacks an early warning margin. The sensor reacts in a strict binary fashion: it is either perfectly stable, or it instantly white-outs/crashes. V3 successfully logs the catastrophic white-out event, but exposes that "early detection" is practically impossible due to the hardware's strict digital cliff.

---

## 5. Phase 4: The Exploratory Framework & Future Outlook (`detector_v4.py`)

**Goal:** Acknowledging the strict "shutdown or white-out" nature of the hardware, V4 is a **High-Dimensional Exploratory Logging Architecture** designed for future advanced spoofing research.

### 5.1 The Ultimate Objective: Targeted Semantic Spoofing
Currently, EMI only achieves brute-force blinding (white-out) or crashing (DoS). The future goal of hardware security is **Targeted Spoofing**—manipulating the RF noise frequencies so precisely that we trick the AI into predicting a specific incorrect class (e.g., morphing a "Sponge" into a "Coin" in the feature space) without crashing the USB.

### 5.2 The Logging Architecture (The Math of AI Confusion)
V4 uses a highly optimized Single-Pass Inference design to simultaneously log metrics at 5 FPS:

*   **Deep Feature Space Shift (Cosine Distance):**
    $$D_C = 1 - \frac{A \cdot B}{||A|| \times ||B||}$$
    > **Mathematical Mechanism:** A neural network processes an image into a high-dimensional vector (e.g., 512 dimensions) representing abstract features. Cosine distance measures the angle between two vectors in this multi-dimensional space. A distance of 0 means the vectors point in the exact same direction (identical features). When EMI corrupts the image, it alters the CNN's internal math, causing the resulting vector to rotate away from the baseline vector. This metric quantifies exactly how much the AI's internal representation has been skewed.
*   **Shannon Entropy (Information Loss):**
    $$H = -\sum_{i} p_i \log_2(p_i)$$
    > **Mathematical Mechanism:** In Information Theory, Shannon Entropy quantifies the amount of uncertainty or structural information in a signal. A clear image has a distinct distribution of pixel values, yielding moderate entropy. When EMI forces the sensor into a white-out state, almost all pixels share the exact same value (255). The probability distribution collapses into a single spike, causing the mathematical entropy to drop to near zero. This definitively proves the total loss of structural information.
*   **CMOS DSP Metrics:** V4 also logs the **Banding Index** and **Total Variation** to track exactly how the row-by-row pixel scanners (Row-ADCs) behave during the RF injection.

### Conclusion
The evolution from V1 to V4 documents the extreme fragility of commercial optical tactile sensors. While current EMI attacks predominantly result in system crashes or white-out blinding, V4 establishes the necessary analytical foundation. It bridges the gap between hardware physics and deep learning, providing the mathematical telemetry required to study whether these raw vulnerabilities can eventually be weaponized into precise, targeted adversarial attacks on edge-AI hardware.