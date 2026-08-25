import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# ==============================================================================
# Resolve module path for external SDK
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from gs_sdk import gs_device

# ==============================================================================
# System Configuration
# ==============================================================================
MODEL_FILENAME = 'resnet18_rg_best.pth'  
DISPLAY_SCALE = 2.3  
STABLE_FRAME_THRESHOLD = 3  # Frames required to stabilize before inference

def apply_rg_channel_mask(tensor):
    """
    Zeros out the Blue (B) channel tensor to physically suppress thermal/EMI noise.
    [CRITICAL]: This MUST be applied AFTER Normalization to match the training pipeline exactly.
    """
    masked_tensor = tensor.clone()
    masked_tensor[0, 2, :, :] = 0.0  
    return masked_tensor

def nothing(x):
    """Dummy callback for OpenCV trackbars."""
    pass

# ==============================================================================
# Main Inference Pipeline (Version 1: Robust Baseline)
# ==============================================================================
def main():
    print("[INFO] Initializing V1 Baseline Tactile Inference System...")

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

    # ------------------------------------------
    # 2. Data Preprocessing Pipeline
    # ------------------------------------------
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ------------------------------------------
    # 3. Sensor & UI Setup
    # ------------------------------------------
    dev = gs_device.Camera("GelSight Mini", 240, 320)
    dev.connect()
    
    window_name = "Semantic Tactile Interface"
    cv2.namedWindow(window_name)
    
    # Trackbars for real-time physics sensitivity tuning
    cv2.createTrackbar("Area_Thresh", window_name, 50, 5000, nothing)
    cv2.createTrackbar("Diff_Sens", window_name, 4, 100, nothing)

    print("[INFO] Capturing initial background baseline. Do not touch sensor...")
    baseline = None
    for _ in range(10): 
        baseline = dev.get_image()
        cv2.waitKey(50)
        
    gray_baseline = cv2.GaussianBlur(cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    
    history_probs = []
    stable_frames = 0  # Counter for debounce logic
    print("\n[READY] System operational. Ready for semantic inference.")

    # ------------------------------------------
    # 4. Real-Time Inference Loop
    # ------------------------------------------
    try:
        while True:
            img = dev.get_image()
            if img is None: 
                continue

            # --- Core Physics Engine (Contact Detection) ---
            CONTACT_THRESHOLD = max(1, cv2.getTrackbarPos("Area_Thresh", window_name))
            SENSITIVITY = max(1, cv2.getTrackbarPos("Diff_Sens", window_name))
            
            gray_img = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            diff = cv2.absdiff(gray_img, gray_baseline)
            _, thresh = cv2.threshold(diff, SENSITIVITY, 255, cv2.THRESH_BINARY)
            
            contact_area = np.sum(thresh == 255)
            is_contact = contact_area > CONTACT_THRESHOLD

            # --- Inference Engine (Debounced + RG Masked) ---
            if is_contact:
                stable_frames += 1
                
                # Require N stable frames to bypass transient motion blur
                if stable_frames > STABLE_FRAME_THRESHOLD:
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # 1. Normalize First
                    input_tensor = preprocess(Image.fromarray(rgb_img)).unsqueeze(0)
                    # 2. Mask the Blue Channel Second (Matches Training exactly)
                    input_tensor = apply_rg_channel_mask(input_tensor).to(device)
                    
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = F.softmax(outputs, dim=1)[0].cpu().numpy()

                    # Force background probability to zero when physical contact is confirmed
                    bg_index = class_names.index('background')
                    probabilities[bg_index] = 0.0
                    prob_sum = np.sum(probabilities)
                    if prob_sum > 0:
                        probabilities = probabilities / prob_sum

                    # Temporal smoothing for prediction stability
                    history_probs.append(probabilities)
                    if len(history_probs) > 5: 
                        history_probs.pop(0)
                    smooth_probs = np.mean(history_probs, axis=0)

                    predicted_idx = np.argmax(smooth_probs)
                    predicted_label = class_names[predicted_idx].upper()
                    confidence = smooth_probs[predicted_idx] * 100
                else:
                    predicted_label = "STABILIZING..."
                    confidence = 0.0
            else:
                # Reset states instantly upon release
                stable_frames = 0
                history_probs.clear()
                predicted_label = "IDLE"
                confidence = 0.0

            # ------------------------------------------
            # 5. UI Rendering (Main Dashboard)
            # ------------------------------------------
            new_width = int(img.shape[1] * DISPLAY_SCALE)
            new_height = int(img.shape[0] * DISPLAY_SCALE)
            display_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            h, w = display_img.shape[:2]

            overlay = display_img.copy()
            bar_height = 120
            cv2.rectangle(overlay, (0, h - bar_height), (w, h), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.85, display_img, 0.15, 0, display_img)
            
            if is_contact and stable_frames > STABLE_FRAME_THRESHOLD:
                color = (0, 255, 100) # Green
                cv2.putText(display_img, f"{predicted_label}", (40, h - 40), cv2.FONT_HERSHEY_DUPLEX, 1.8, color, 2)
                cv2.putText(display_img, f"CONFIDENCE: {confidence:.1f}%", (w - 250, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(display_img, "MATCH FOUND", (40, h - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            elif is_contact:
                color = (0, 165, 255) # Orange
                cv2.putText(display_img, f"{predicted_label}", (40, h - 40), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
                cv2.putText(display_img, "Analyzing surface topography...", (40, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            else:
                color = (150, 150, 150) # Gray
                cv2.putText(display_img, "READY TO SCAN", (40, h - 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
                cv2.putText(display_img, "Apply pressure to trigger inference", (40, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)

            cv2.putText(display_img, f"Model: {MODEL_FILENAME} (RG-Mode)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_img, "[C] Calibrate Baseline   [Q] Quit", (w - 280, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

            # Show Main UI
            cv2.imshow(window_name, display_img)
            
            # Show Contact Physics Mask (Crucial for tuning Diff_Sens & Area_Thresh)
            cv2.imshow("Contact Physics Mask", thresh)
            
            # ------------------------------------------
            # 6. Keyboard Events
            # ------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                gray_baseline = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
                print("[INFO] Baseline manually recalibrated.")
            elif key == ord('q'):
                print("[INFO] Shutting down system...")
                break

    finally:
        cv2.destroyAllWindows()
        if hasattr(dev, 'disconnect'): 
            dev.disconnect()

if __name__ == '__main__':
    main()