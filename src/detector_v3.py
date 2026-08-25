import os
import sys
import time
import cv2
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from skimage.metrics import structural_similarity as ssim

# ==============================================================================
# Resolve module path for external SDK
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from gs_sdk import gs_device

# ==============================================================================
# System Configuration & Thermal Management
# ==============================================================================
MODEL_FILENAME = 'resnet18_rg_best.pth'  
DISPLAY_SCALE = 2.3  

# FPS Lock for Thermal Stability (Mitigates Resonant Frequency Drift)
TARGET_FPS = 5  
FRAME_DELAY = 1.0 / TARGET_FPS

# EMI Quantification Thresholds
ADC_NOISE_FLOOR = 5  # Absolute voltage deviation threshold for PCR

def apply_rg_channel_mask(tensor):
    """
    Zeros out the Blue (B) channel tensor to suppress thermal noise.
    Forces the CNN to learn strictly from macro-shape (R) and micro-texture (G).
    """
    masked_tensor = tensor.clone()
    masked_tensor[0, 2, :, :] = 0.0  
    return masked_tensor

def nothing(x): 
    """Dummy callback for OpenCV trackbars."""
    pass

def main():
    print("[INFO] Starting V3 Statistical EMI Attack Monitoring System...")

    # ------------------------------------------
    # 1. Model Initialization
    # ------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names = ['background', 'breadboard', 'coin', 'fingerprint', 'sponge', 'usb']
    
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model_path = os.path.join(BASE_DIR, 'models', MODEL_FILENAME)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[SUCCESS] Model weights loaded: {MODEL_FILENAME}")
    except FileNotFoundError:
        print(f"[ERROR] Weights not found at {model_path}.")
        return

    model = model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ------------------------------------------
    # 2. Sensor & UI Setup
    # ------------------------------------------
    dev = gs_device.Camera("GelSight Mini", 240, 320)
    dev.connect()
    
    window_name = "Semantic Tactile Interface"
    cv2.namedWindow(window_name)
    cv2.createTrackbar("Area_Thresh", window_name, 30, 5000, nothing)
    cv2.createTrackbar("Diff_Sens", window_name, 3, 100, nothing)

    print("[INFO] Warming up and acquiring stable baseline. Do not touch the sensor...")
    for _ in range(15): 
        baseline = dev.get_image()
        cv2.waitKey(100)
        
    gray_baseline = cv2.GaussianBlur(cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    # ------------------------------------------
    # 3. Scientific Data Logger Initialization
    # ------------------------------------------
    # Updated CSV header for V3 Statistical Metrics
    log_filename = f"iemi_statistical_log_{int(time.time())}.csv"
    csv_file = open(log_filename, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Timestamp", "SSIM", "PCR_Percent", "Diff_Var", "HF_Ratio", "Max_Surge", "Predicted_Label", "Confidence"])

    fft_history = []
    frame_count = 0
    last_frame_time = time.time()
    
    print("\n[READY] System running.")

    try:
        while True:
            # --- Image Acquisition ---
            img = dev.get_image()
            if img is None: 
                continue
                
            frame_count += 1
            current_time = time.time()

            gray_img = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            
            # Absolute difference matrix (Voltage deviation model)
            diff = cv2.absdiff(gray_img, gray_baseline)
            noise_max = np.max(diff)
            
            # ==========================================
            # [V3 UPGRADE] Microscopic Statistical Metrics 
            # ==========================================
            # 1. Structural Similarity Index (Macro structural degradation)
            current_ssim = ssim(gray_baseline, gray_img, data_range=255)
            
            # 2. PCR (Pixel Corruption Ratio) - Hard Threshold Bit-flips
            corrupted_pixels = np.sum(diff > ADC_NOISE_FLOOR)
            pcr_percent = (corrupted_pixels / diff.size) * 100.0
            
            # 3. Difference Variance - Sub-threshold energy detection
            diff_variance = np.var(diff)

            # ==========================================
            # Frequency Domain Analysis (FFT)
            # ==========================================
            f_transform = np.fft.fft2(gray_img)
            f_shift = np.fft.fftshift(f_transform)
            
            # 4. High-Frequency (HF) Energy Ratio Calculation
            h_fft, w_fft = f_shift.shape
            center_y, center_x = h_fft // 2, w_fft // 2
            
            # Create a spatial mask to block out the DC component (Low frequencies)
            y, x = np.ogrid[:h_fft, :w_fft]
            hf_mask = (x - center_x)**2 + (y - center_y)**2 > 30**2
            
            # Calculate energy distribution
            high_freq_energy = np.sum(np.abs(f_shift[hf_mask])**2)
            total_energy = np.sum(np.abs(f_shift)**2)
            hf_ratio = (high_freq_energy / (total_energy + 1e-8)) * 100.0

            # Real-time FFT visualization
            magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
            fft_history.append(magnitude_spectrum)
            if len(fft_history) > 5:
                fft_history.pop(0)
                
            avg_magnitude = np.mean(fft_history, axis=0)
            avg_magnitude = cv2.normalize(avg_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # High-Pass Filter visual mask (The Black Sphere)
            cv2.circle(avg_magnitude, (center_x, center_y), 30, (0, 0, 0), -1) 
            
            cv2.putText(avg_magnitude, f"Diff Var: {diff_variance:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(avg_magnitude, f"HF Energy: {hf_ratio:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("FFT Spectrum (RF Spikes Detector)", avg_magnitude)

            # --- EMI Heatmap ---
            deadzone_diff = diff.copy()
            deadzone_diff[deadzone_diff <= 2] = 0 
            amplified_diff = cv2.convertScaleAbs(deadzone_diff, alpha=15.0, beta=0)
            emi_heatmap = cv2.applyColorMap(amplified_diff, cv2.COLORMAP_JET)
            cv2.putText(emi_heatmap, f"PCR: {pcr_percent:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("EMI Heatmap", emi_heatmap)

            # ==========================================
            # AI Inference Engine & Spoofing Trigger
            # ==========================================
            CONTACT_THRESHOLD = max(1, cv2.getTrackbarPos("Area_Thresh", window_name))
            _, thresh = cv2.threshold(diff, max(1, cv2.getTrackbarPos("Diff_Sens", window_name)), 255, cv2.THRESH_BINARY)
            
            # Trigger AI if physical contact is made OR strong RF interference is detected (PCR > 1.0)
            is_contact = (np.sum(thresh == 255) > CONTACT_THRESHOLD) or (pcr_percent > 1.0)

            if is_contact:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                input_tensor = preprocess(Image.fromarray(rgb_img)).unsqueeze(0)
                input_tensor = apply_rg_channel_mask(input_tensor).to(device)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = F.softmax(outputs, dim=1)[0].cpu().numpy()

                bg_index = class_names.index('background')
                probabilities[bg_index] = 0.0
                prob_sum = np.sum(probabilities)
                if prob_sum > 0: probabilities = probabilities / prob_sum

                predicted_idx = np.argmax(probabilities)
                predicted_label = class_names[predicted_idx].upper()
                confidence = probabilities[predicted_idx] * 100
            else:
                predicted_label = "IDLE"
                confidence = 0.0

            # Log Sub-threshold metrics
            csv_writer.writerow([f"{current_time:.3f}", f"{current_ssim:.4f}", f"{pcr_percent:.3f}", f"{diff_variance:.2f}", f"{hf_ratio:.2f}", noise_max, predicted_label, f"{confidence:.2f}"])
            
            # Terminal Throttled Output
            if frame_count % 5 == 0:
                print(f"[Metrics] Var: {diff_variance:5.2f} | HF%: {hf_ratio:5.2f} | PCR: {pcr_percent:5.2f}% | DETECTOR: {predicted_label}")

            # ==========================================
            # Main UI Rendering
            # ==========================================
            new_width = int(img.shape[1] * DISPLAY_SCALE)
            new_height = int(img.shape[0] * DISPLAY_SCALE)
            display_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            h, w = display_img.shape[:2]

            overlay = display_img.copy()
            bar_height = 120
            cv2.rectangle(overlay, (0, h - bar_height), (w, h), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.85, display_img, 0.15, 0, display_img)
            
            if is_contact:
                color = (0, 0, 255) if pcr_percent > 1.0 else (0, 255, 100)
                warning_text = "EMI SPOOFING DETECTED" if pcr_percent > 1.0 else "NORMAL CONTACT"
                
                cv2.putText(display_img, f"{predicted_label}", (40, h - 40), cv2.FONT_HERSHEY_DUPLEX, 1.8, color, 2)
                cv2.putText(display_img, f"CONFIDENCE: {confidence:.1f}%", (w - 250, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(display_img, warning_text, (40, h - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            else:
                cv2.putText(display_img, "READY FOR EMI INJECTION", (40, h - 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (150, 150, 150), 2)

            cv2.imshow(window_name, display_img)
            
            # ==========================================
            # Thermal Management & Event Loop
            # ==========================================
            elapsed = time.time() - last_frame_time
            sleep_time = FRAME_DELAY - elapsed
            
            delay_ms = max(1, int(sleep_time * 1000)) if sleep_time > 0 else 1

            key = cv2.waitKey(delay_ms) & 0xFF
            last_frame_time = time.time()

            if key == ord('c'):
                gray_baseline = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
                print("[INFO] Baseline manually recalibrated.")
            elif key == ord('q'):
                print("[INFO] Shutting down system...")
                break

    finally:
        csv_file.close()
        cv2.destroyAllWindows()
        if hasattr(dev, 'disconnect'): 
            dev.disconnect()
        print(f"[INFO] Log safely saved to {log_filename}")

if __name__ == '__main__':
    main()