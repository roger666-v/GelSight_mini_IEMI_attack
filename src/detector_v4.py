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
from scipy.spatial.distance import cosine

# ==============================================================================
# Resolve module path for external SDK
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from gs_sdk import gs_device

# ==============================================================================
# System Configuration
# ==============================================================================
MODEL_FILENAME = 'resnet18_rg_best.pth'  # Ensure this matches your trained weights
DISPLAY_SCALE = 2.3  
TARGET_FPS = 5  
FRAME_DELAY = 1.0 / TARGET_FPS

class ChannelMasker(object):
    """
    Dynamically zeros out specific RGB channels to validate hardware signal integrity.
    [CRITICAL]: Applied AFTER Normalization to match the standard training pipeline exactly.
    """
    def __init__(self, mode='RG'):
        self.mode = mode.upper()
    def __call__(self, tensor):
        masked_tensor = tensor.clone()
        if 'R' not in self.mode: masked_tensor[0, :, :] = 0.0
        if 'G' not in self.mode: masked_tensor[1, :, :] = 0.0
        if 'B' not in self.mode: masked_tensor[2, :, :] = 0.0
        return masked_tensor

def nothing(x): 
    pass

# ==============================================================================
# Main V4 Pipeline (Deep Feature Shift & DSP Monitor)
# ==============================================================================
def main():
    print("[INFO] Starting V4 EMI Degradation Monitor (Optimized Version)...")

    # ------------------------------------------
    # 1. Model Initialization & Feature Extractor
    # ------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names = ['background', 'breadboard', 'coin', 'fingerprint', 'sponge', 'usb']
    
    full_model = models.resnet18(weights=None)
    full_model.fc = nn.Linear(full_model.fc.in_features, len(class_names))
    model_path = os.path.join(BASE_DIR, 'models', MODEL_FILENAME)
    
    try:
        full_model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[SUCCESS] Model weights loaded: {MODEL_FILENAME}")
    except FileNotFoundError:
        print(f"[ERROR] Weights not found at {model_path}.")
        return

    full_model = full_model.to(device)
    full_model.eval()

    # Extract the CNN backbone (everything before the final FC layer)
    feature_extractor = torch.nn.Sequential(*(list(full_model.children())[:-1])).to(device)
    feature_extractor.eval()
    
    # Store the FC layer separately for the single-pass optimization
    fc_layer = full_model.fc

    # CRITICAL: Proper Preprocessing Pipeline (Normalize -> Mask)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ChannelMasker('RG')
    ])

    # ------------------------------------------
    # 2. Sensor Setup
    # ------------------------------------------
    dev = gs_device.Camera("GelSight Mini", 240, 320)
    dev.connect()
    
    window_name = "V4 Semantic & Physical Monitor"
    cv2.namedWindow(window_name)

    print("[INFO] Warming up. Do not touch the sensor...")
    baseline = None
    for _ in range(15): 
        try:
            baseline = dev.get_image()
            cv2.waitKey(100)
        except Exception:
            pass
            
    if baseline is not None:
        gray_baseline = cv2.GaussianBlur(cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    else:
        print("[ERROR] Failed to capture baseline. Please check USB.")
        return

    # ------------------------------------------
    # 3. Experiment States & Loggers
    # ------------------------------------------
    log_filename = f"v4_emi_experiment_{int(time.time())}.csv"
    csv_file = open(log_filename, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "Timestamp", "Phase", "Predicted", "Confidence", 
        "SSIM_Loss", "Feature_Distance", "Banding_Idx", "TV_Score", "Entropy"
    ])

    experiment_phase = "WAITING" 
    locked_label = None
    locked_confidence = 0.0
    locked_features = None
    locked_gray_img = None
    locked_gray_small = None # For fast SSIM
    last_frame_time = time.time()
    
    print("\n[INSTRUCTION] Press [l] to LOCK-IN the current object.")
    print("[INSTRUCTION] Press [e] to begin EMI INJECTION phase.")
    print("[INSTRUCTION] Press [r] to RESET. Press [q] to QUIT.\n")

    try:
        while True:
            # --- Image Acquisition ---
            try:
                img = dev.get_image()
            except Exception:
                img = None

            if img is None:
                print("[WARNING] USB Disconnected. Waiting for hardware recovery...")
                time.sleep(1)
                continue

            current_time = time.time()
            gray_img = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)

            # --- AI Inference (OPTIMIZED: Single Pass) ---
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(Image.fromarray(rgb_img)).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # 1. Extract Deep Features (Shape: [1, 512, 1, 1])
                features_tensor = feature_extractor(input_tensor)
                # Flatten to 1D vector (Shape: [1, 512])
                features_flat = torch.flatten(features_tensor, 1) 
                
                # 2. Pass directly to FC layer for classification (Avoids running CNN twice)
                outputs = fc_layer(features_flat)
                
                probabilities = F.softmax(outputs, dim=1)[0].cpu().numpy()
                predicted_idx = np.argmax(probabilities)
                current_label = class_names[predicted_idx].upper()
                current_confidence = probabilities[predicted_idx] * 100

                # Save features for Cosine Distance calculation
                features = features_flat.squeeze().cpu().numpy()

            # --- Calculate Metrics (If Locked) ---
            feature_distance = 0.0
            ssim_loss = 0.0
            banding_index = 0.0
            tv_score = 0.0
            entropy = 0.0
            
            if experiment_phase in ["LOCKED", "EMI_INJECTION"]:
                # [AI Metric 1] Feature Space Shift (Cosine Distance)
                feature_distance = cosine(locked_features, features) * 100.0
                
                # [AI Metric 2] Structural Similarity Loss (OPTIMIZED: Downscaled by 50%)
                gray_small = cv2.resize(gray_img, (160, 120))
                current_ssim = ssim(locked_gray_small, gray_small, data_range=255)
                ssim_loss = (1.0 - current_ssim) * 100.0

                # ==========================================
                # [CMOS Physical Metrics] Base DSP Analysis 
                # ==========================================
                noise_map = cv2.absdiff(gray_img, locked_gray_img)
                noise_map_f = noise_map.astype(np.float32) 

                # 1. Banding Index (Row-ADC Power Coupling)
                row_noise_means = np.mean(noise_map_f, axis=1)
                banding_index = np.var(row_noise_means)

                # 2. Total Variation (High-Frequency Bit-flip / Salt & Pepper)
                diff_x = np.mean(np.abs(noise_map_f[:, 1:] - noise_map_f[:, :-1]))
                diff_y = np.mean(np.abs(noise_map_f[1:, :] - noise_map_f[:-1, :]))
                tv_score = (diff_x + diff_y) * 100.0 

                # 3. Shannon Entropy (Random Information Injection)
                hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256]).flatten()
                hist_prob = hist / (hist.sum() + 1e-7)
                hist_prob = hist_prob[hist_prob > 0] 
                entropy = -np.sum(hist_prob * np.log2(hist_prob))

            # --- Logging ---
            csv_writer.writerow([
                f"{current_time:.3f}", experiment_phase, current_label, 
                f"{current_confidence:.2f}", f"{ssim_loss:.4f}", f"{feature_distance:.4f}", 
                f"{banding_index:.4f}", f"{tv_score:.4f}", f"{entropy:.4f}"
            ])

            # ==========================================
            # UI Rendering
            # ==========================================
            new_width = int(img.shape[1] * DISPLAY_SCALE)
            new_height = int(img.shape[0] * DISPLAY_SCALE)
            display_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            h, w = display_img.shape[:2]

            overlay = display_img.copy()
            bar_height = 200 
            cv2.rectangle(overlay, (0, h - bar_height), (w, h), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.85, display_img, 0.15, 0, display_img)
            
            # Left Panel: AI Prediction
            cv2.putText(display_img, f"PRED: {current_label}", (30, h - 140), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
            cv2.putText(display_img, f"CONF: {current_confidence:.1f}%", (30, h - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

            # Right Panel: Invisible Degradation Metrics
            if experiment_phase in ["LOCKED", "EMI_INJECTION"]:
                # AI Level Metrics
                cv2.putText(display_img, f"FEAT SHIFT: {feature_distance:.2f}", (w - 320, h - 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                cv2.putText(display_img, f"SSIM LOSS : {ssim_loss:.2f}%", (w - 320, h - 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Physical DSP Metrics
                cv2.putText(display_img, f"BANDING   : {banding_index:.3f}", (w - 320, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
                cv2.putText(display_img, f"TV SCORE  : {tv_score:.2f}", (w - 320, h - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
                cv2.putText(display_img, f"ENTROPY   : {entropy:.3f}", (w - 320, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

            # Bottom Status Bar
            if experiment_phase == "WAITING":
                cv2.rectangle(display_img, (0, h - 30), (w, h), (100, 100, 100), -1)
                cv2.putText(display_img, "Press [l] to LOCK baseline object", (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            elif experiment_phase == "LOCKED":
                cv2.rectangle(display_img, (0, h - 30), (w, h), (0, 150, 0), -1)
                cv2.putText(display_img, f"LOCKED on {locked_label}. Press [e] to start EMI.", (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            elif experiment_phase == "EMI_INJECTION":
                cv2.rectangle(display_img, (0, h - 30), (w, h), (0, 0, 255), -1)
                cv2.putText(display_img, "EMI INJECTION ACTIVE. Logging data...", (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow(window_name, display_img)
            
            # ==========================================
            # Keyboard Events & State Machine
            # ==========================================
            elapsed = time.time() - last_frame_time
            sleep_time = FRAME_DELAY - elapsed
            delay_ms = max(1, int(sleep_time * 1000)) if sleep_time > 0 else 1

            key = cv2.waitKey(delay_ms) & 0xFF
            last_frame_time = time.time()

            if key == ord('l'):
                experiment_phase = "LOCKED"
                locked_label = current_label
                locked_confidence = current_confidence
                locked_features = features.copy()
                locked_gray_img = gray_img.copy()
                locked_gray_small = cv2.resize(locked_gray_img, (160, 120)) # Cache small img for fast SSIM
                print(f"\n[LOCKED] Target: {locked_label} | Conf: {locked_confidence:.1f}%")
                
            elif key == ord('e') and experiment_phase == "LOCKED":
                experiment_phase = "EMI_INJECTION"
                print("\n[WARNING] EMI Injection Phase Started. Recording DSP degradation metrics...")
                
            elif key == ord('r'):
                experiment_phase = "WAITING"
                print("\n[INFO] Experiment reset.")
                
            elif key == ord('q'):
                break

    finally:
        csv_file.close()
        cv2.destroyAllWindows()
        if hasattr(dev, 'disconnect'): 
            dev.disconnect()
        print(f"[INFO] Experiment log saved to {log_filename}")

if __name__ == '__main__':
    main()