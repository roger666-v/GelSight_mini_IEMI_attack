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
# System Configuration (EMI EXPERIMENT EDITION)
# ==============================================================================
MODEL_FILENAME = 'resnet18_rg_best.pth'  
DISPLAY_SCALE = 2.3  
STABLE_FRAME_THRESHOLD = 0  # Set to 0: Makes the system hyper-sensitive to instantaneous EMI pulses

def apply_rg_channel_mask(tensor):
    """
    Zeros out the Blue (B) channel tensor to physically suppress thermal/EMI noise.
    """
    masked_tensor = tensor.clone()
    masked_tensor[0, 2, :, :] = 0.0  
    return masked_tensor

def nothing(x): 
    pass

# ==============================================================================
# Main Inference Pipeline
# ==============================================================================
def main():
    print("[INFO] Starting EMI-Vulnerable Tactile System with Scientific Metrics...")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names = ['background', 'breadboard', 'coin', 'fingerprint', 'sponge', 'usb']
    
    # ------------------------------------------
    # 1. Model Initialization
    # ------------------------------------------
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
    cv2.createTrackbar("Area_Thresh", window_name, 10, 5000, nothing)
    cv2.createTrackbar("Diff_Sens", window_name, 3, 100, nothing)

    print("[INFO] Initializing STATIC background baseline...")
    baseline = None
    for _ in range(10): 
        baseline = dev.get_image()
        cv2.waitKey(50)
        
    gray_baseline = cv2.GaussianBlur(cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    # ------------------------------------------
    # 3. CSV Logger & History Variables
    # ------------------------------------------
    log_filename = f"emi_experiment_log_{int(time.time())}.csv"
    csv_file = open(log_filename, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Timestamp", "SSIM", "B_Var", "R_Var", "Noise_Max", "Contact_Area", "Predicted_Label", "Confidence"])
    print(f"[INFO] Experimental data will be logged to: {log_filename}")

    history_probs = []
    fft_history = []  # Queue for FFT temporal averaging
    stable_frames = 0
    frame_count = 0
    
    print("\n[READY] System operational! Waiting for Signal Generator Injection...")

    try:
        while True:
            img = dev.get_image()
            if img is None: continue
            frame_count += 1
            current_time = time.time()

            # --- Core Physics Engine ---
            CONTACT_THRESHOLD = max(1, cv2.getTrackbarPos("Area_Thresh", window_name))
            SENSITIVITY = max(1, cv2.getTrackbarPos("Diff_Sens", window_name))
            
            gray_img = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            
            # Calculate background absolute difference (Core physical variable)
            diff = cv2.absdiff(gray_img, gray_baseline)
            _, thresh = cv2.threshold(diff, SENSITIVITY, 255, cv2.THRESH_BINARY)
            
            contact_area = np.sum(thresh == 255)
            noise_max = np.max(diff)
            is_contact = contact_area > CONTACT_THRESHOLD

            # ==========================================
            # [Academic Metrics] SSIM & Channel Variance
            # ==========================================
            current_ssim = ssim(gray_baseline, gray_img, data_range=255)
            b_channel, g_channel, r_channel = cv2.split(img)
            b_var = np.var(b_channel)
            r_var = np.var(r_channel)

            # ==========================================
            # [Frequency Domain] Noise-Resistant FFT
            # ==========================================
            f_transform = np.fft.fft2(gray_img)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
            
            fft_history.append(magnitude_spectrum)
            if len(fft_history) > 10:
                fft_history.pop(0)
                
            avg_magnitude = np.mean(fft_history, axis=0)
            avg_magnitude = cv2.normalize(avg_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Mask the low-frequency DC center to highlight high-frequency RF spikes
            h_fft, w_fft = avg_magnitude.shape
            cv2.circle(avg_magnitude, (w_fft//2, h_fft//2), 30, (0, 0, 0), -1)
            cv2.imshow("FFT Spectrum (Look for bright spikes)", avg_magnitude)

            # ==========================================
            # [Visualization] Noise-Gated EMI Heatmap
            # ==========================================
            deadzone_diff = diff.copy()
            deadzone_diff[deadzone_diff <= 2] = 0  # Filter out shot noise (values <= 2)
            
            amplified_diff = cv2.convertScaleAbs(deadzone_diff, alpha=15.0, beta=0)
            emi_heatmap = cv2.applyColorMap(amplified_diff, cv2.COLORMAP_JET)
            cv2.putText(emi_heatmap, f"Surge: {noise_max}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("EMI Heatmap (Filtered)", emi_heatmap)

            # --- Inference Engine (Vulnerable to EMI) ---
            if is_contact:
                stable_frames += 1
                if stable_frames > STABLE_FRAME_THRESHOLD:
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

                    history_probs.append(probabilities)
                    if len(history_probs) > 3: history_probs.pop(0)
                    smooth_probs = np.mean(history_probs, axis=0)

                    predicted_idx = np.argmax(smooth_probs)
                    predicted_label = class_names[predicted_idx].upper()
                    confidence = smooth_probs[predicted_idx] * 100
                else:
                    predicted_label = "STABILIZING..."
                    confidence = 0.0
            else:
                stable_frames = 0
                history_probs.clear()
                predicted_label = "IDLE"
                confidence = 0.0

            # --- CSV & Terminal Log ---
            csv_writer.writerow([f"{current_time:.3f}", f"{current_ssim:.4f}", f"{b_var:.1f}", f"{r_var:.1f}", noise_max, contact_area, predicted_label, f"{confidence:.2f}"])
            
            if frame_count % 30 == 0:
                print(f"[Metrics] SSIM: {current_ssim:.4f} | B-Var: {b_var:.1f} | R-Var: {r_var:.1f} | Max Surge: {noise_max}")

            # ------------------------------------------
            # UI Rendering (Main Dashboard)
            # ------------------------------------------
            new_width = int(img.shape[1] * DISPLAY_SCALE)
            new_height = int(img.shape[0] * DISPLAY_SCALE)
            display_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            h, w = display_img.shape[:2]

            overlay = display_img.copy()
            bar_height = 120
            cv2.rectangle(overlay, (0, h - bar_height), (w, h), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.85, display_img, 0.15, 0, display_img)
            
            if is_contact:
                color = (0, 0, 255) # Bright red warning
                cv2.putText(display_img, f"{predicted_label}", (40, h - 40), cv2.FONT_HERSHEY_DUPLEX, 1.8, color, 2)
                cv2.putText(display_img, f"CONFIDENCE: {confidence:.1f}%", (w - 250, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(display_img, "SIGNAL DETECTED", (40, h - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            else:
                color = (150, 150, 150) # Gray idle
                cv2.putText(display_img, "READY FOR EMI", (40, h - 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
                cv2.putText(display_img, "Waiting for RF Interference...", (40, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)

            cv2.putText(display_img, f"Model: {MODEL_FILENAME}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 1)
            cv2.putText(display_img, "[C] Calibrate   [Q] Quit", (w - 280, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

            cv2.imshow(window_name, display_img)
            
            # Frame rate lock to prevent CPU overloading (~30 FPS)
            key = cv2.waitKey(33) & 0xFF
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