# Open-Source Tactile Ecosystem & Research Positioning

## 1. Introduction and Taxonomy
The open-source ecosystem for vision-based tactile sensors (e.g., GelSight, DIGIT) has expanded rapidly. However, repositories are heavily fragmented based on downstream tasks—ranging from raw driver initialization to complex robotic kinematics. 

To help future researchers navigate this landscape, we present a comprehensive taxonomy of the current ecosystem. We objectively evaluate these repositories across four distinct functional categories and explicitly define the boundary of our own research: **Hardware Security and Signal Integrity.**

---

## 2. Landscape Comparison

| Category | Primary Focus | Prominent Repositories | Core Mathematical / Engineering Approach | Constraints Regarding Hardware Security |
| :--- | :--- | :--- | :--- | :--- |
| **A. Base SDKs** | Raw hardware streaming & initialization | `gelsightinc/gsrobotics`, `joehjhuang/gs_sdk` | Direct FFMpeg / OpenCV UVC decoding | Lacks runtime diagnostic metrics; blindly trusts the USB payload. |
| **B. ROS Wrappers** | Multi-node robotic communication | `RVSATHU/gelsight_mini_ros`, `ai4ce/gelsight_ROS2_interface` | Topic serialization (`rosbag`) | Network packetization masks bit-level ADC errors and introduces latency. |
| **C. Kinematics** | Slip detection & shear force | `GelSight/tracking`, `personalrobotics/gelslight_tracking` | Lucas-Kanade Optical Flow, Affine transformations | Requires external Force/Torque sensors for quantitative ground-truth validation. |
| **D. 3D Topology** | Depth mapping & FEA | `rpl-cmu/gelslam`, `feats-ai/feats` | Poisson integration, Convex Optimization | Requires a calibrated robotic manipulator for physical ground-truth validation. |

---

## 3. Comprehensive Category Analysis & Engineering Decisions

### Category A: Base Hardware SDKs & Drivers
These repositories form the bedrock of the ecosystem, providing the essential logic to fetch uncompressed RGB frames via standard USB Video Class (UVC) drivers. 
*   **Engineering Reality:** Most downstream Machine Learning researchers treat these SDKs as "black boxes," assuming the digital output perfectly reflects the analog reality.
*   **Our Positioning:** We selected `joehjhuang/gs_sdk` as our architectural foundation. Its lightweight, transparent FFMpeg implementation allows us to intercept the raw tensor data exactly as it leaves the physical sensor interface. This zero-abstraction access is critical for our research, as we aim to detect sub-threshold physical anomalies (e.g., thermal noise, RF coupling) before they are smoothed out by higher-level software APIs.

### Category B: Robotic Arm Integration (ROS / ROS2)
Designed for closed-loop robotic control, these wrappers package tactile image streams into standardized ROS topics, enabling synchronized data collection with robotic manipulators (e.g., UR5, Franka Emika).
*   **Academic Considerations:** While excellent for distributed robotics, ROS message serialization introduces non-deterministic network latency. Furthermore, ROS often drops corrupted packets silently.
*   **Our Positioning:** Hardware security analysis requires strict, deterministic frame-timing to capture instantaneous RF voltage transients. By intentionally bypassing ROS/ROS2 middleware, our pipeline avoids the data-masking effects of packetization and maintains the deterministic low-latency execution required for real-time DSP monitoring.

### Category C: Slip Detection via Optical Flow
These repositories focus on tracking the physical displacement of the internal marker array (the black dots on the elastomer) to calculate planar shear forces.
*   **Mathematical Approach:** They fundamentally rely on the optical flow brightness constancy constraint:
    $$\nabla I \cdot \vec{v} + \frac{\partial I}{\partial t} = 0$$
*   **The Scope Boundary (Resource Constraints):** Quantitatively validating physical slip and calculating true shear forces strictly requires a physical baseline—typically achieved using a highly calibrated multi-axis Force/Torque sensor (e.g., ATI Nano43). Without this external equipment to establish ground truth, any force measurements or slip predictions remain purely qualitative and cannot be objectively benchmarked.
*   **Our Positioning:** Acknowledging the lack of ground-truth force measurement equipment in our current setup, we explicitly scope out kinematic tracking. Instead, we pivot to investigating the underlying 2D signal integrity, evaluating how electrical noise (such as EMI) inherently corrupts the raw spatial gradients ($\nabla I$) that these optical flow algorithms rely upon.

### Category D: 3D Reconstruction & Topology (FEA)
This is the most mathematically rigorous branch of tactile research. By observing the shading of the RGB LEDs, these repositories reconstruct the continuous 3D surface $Z(x,y)$.
*   **Mathematical Approach:** This typically involves solving the Poisson partial differential equation based on surface normals ($G_x, G_y$):
    $$\nabla^2 Z = \rho(G_x, G_y)$$
    Advanced iterations utilize Convex Optimization and factor-graph frameworks for spatial mapping.
*   **The Scope Boundary (Resource Constraints):** Similar to Category C, validating 3D depth gradients requires precise external ground truth, specifically a precision robotic manipulator to apply exact millimeter-level compressions. Without this hardware baseline, 3D measurements are visual approximations rather than verified metric data.
*   **Our Positioning:** Due to these stringent hardware constraints, we do not pursue 3D topologies. We focus entirely on **2D Semantic Inference and Sub-threshold Signal Integrity**, mathematically proving how analog vulnerabilities corrupt the fundamental 2D spatial features extracted by Convolutional Neural Networks (CNNs).

---

## 4. Executive Summary of Our Contribution

By analyzing the open-source ecosystem, a common assumption emerges: **Current tactile projects often implicitly trust the hardware**, operating under the premise that the USB stream is a flawless digital representation of the physical environment.

Our repository is designed to highlight and quantify the vulnerabilities within this assumption. Operating at the intersection of **Analog Signal Processing** and **Machine Learning**, we provide an investigative toolset (including dataset evaluators and DSP-based monitors) to assess hardware signal integrity. Rather than ensuring absolute data perfection, our framework serves as an analytical benchmark for researchers to detect, quantify, and mitigate sub-threshold hardware degradation (such as thermal noise and EMI) before deploying these sensors in complex environments.