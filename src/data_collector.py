import os
import sys
import cv2
import numpy as np

# ==============================================================================
# Resolve module path for external SDK
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from gs_sdk import gs_device

# ==============================================================================
# Directory Management
# ==============================================================================
def setup_directories(categories):
    """
    Creates standard dataset directories. 
    Adopts the 'Pure Raw' philosophy: Saves uncompressed RGB, leaving channel 
    manipulation for the ML pipeline.
    """
    base_dir = os.path.join(BASE_DIR, "data", "dataset")
    
    for cat in categories.values():
        os.makedirs(os.path.join(base_dir, cat), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "background"), exist_ok=True)
        
    return base_dir

def get_file_count(folder_path):
    """Counts existing PNG files to prevent overwriting the dataset."""
    if not os.path.exists(folder_path): 
        return 0
    return len([f for f in os.listdir(folder_path) if f.endswith('.png')])

def nothing(x):
    """Dummy callback for OpenCV trackbars."""
    pass

# ==============================================================================
# Main Data Collection Pipeline
# ==============================================================================
def main():
    print("[INFO] Launching Data Collector...")
    
    # ------------------------------------------
    # 1. Class Configuration
    # ------------------------------------------
    CLASS_MAP = {
        '1': "fingerprint",
        '2': "sponge",
        '3': "breadboard",
        '4': "coin",
        '5': "usb"
    }
    
    base_dir = setup_directories(CLASS_MAP)
    dev = gs_device.Camera("GelSight Mini", 240, 320)
    dev.connect()
    
    # ------------------------------------------
    # 2. UI & Trackbar Initialization
    # ------------------------------------------
    window_name = "Collector"
    cv2.namedWindow(window_name)
    
    # Trackbars for real-time physics sensitivity tuning
    cv2.createTrackbar("Area_Thresh", window_name, 50, 5000, nothing)
    cv2.createTrackbar("Diff_Sens", window_name, 4, 100, nothing)

    print("[INFO] Initializing sensor baseline. Do NOT touch the sensor...")
    baseline = None
    for _ in range(10):  
        baseline = dev.get_image()
        cv2.waitKey(50)
    
    gray_baseline = cv2.GaussianBlur(cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    print("\n[READY] System online! Target: 300-500 high-quality images per class.")
    print("-" * 50)
    for key, name in CLASS_MAP.items():
        print(f" Press [{key}] to save -> {name.upper()}")
    print(" Press [0] to save -> BACKGROUND")
    print(" Press [C] to recalibrate Baseline")
    print(" Press [Q] to quit")
    print("-" * 50)

    try:
        while True:
            img = dev.get_image()
            if img is None: 
                continue
            
            clean_raw_img = img.copy()
            
            # ---------------------------------------------------------
            # 3. Physics Engine (Contact Trigger Mask)
            # ---------------------------------------------------------
            CONTACT_THRESHOLD = max(1, cv2.getTrackbarPos("Area_Thresh", window_name))
            SENSITIVITY = max(1, cv2.getTrackbarPos("Diff_Sens", window_name))
            
            gray_img = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            diff = cv2.absdiff(gray_img, gray_baseline)
            _, thresh = cv2.threshold(diff, SENSITIVITY, 255, cv2.THRESH_BINARY)
            
            contact_area = np.sum(thresh == 255)
            is_contact = contact_area > CONTACT_THRESHOLD
            
            # ---------------------------------------------------------
            # 4. RGB Channel Analysis Window (Live Diagnostic)
            # ---------------------------------------------------------
            b, g, r = cv2.split(clean_raw_img)
            zeros = np.zeros_like(b)
            r_vis = cv2.merge([zeros, zeros, r])
            g_vis = cv2.merge([zeros, g, zeros])
            b_vis = cv2.merge([b, zeros, zeros])
            
            cv2.putText(r_vis, "R-Channel", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(g_vis, "G-Channel", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(b_vis, "B-Channel", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            rgb_analysis = cv2.hconcat([r_vis, g_vis, b_vis])
            cv2.imshow("Live RGB Spectrum Diagnostic", rgb_analysis)

            # ---------------------------------------------------------
            # 5. Main UI Dashboard Rendering
            # ---------------------------------------------------------
            ui_panel = np.zeros((140, 320, 3), dtype=np.uint8)
            cv2.putText(ui_panel, "RAW MODE (.png)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Status Indication
            if is_contact:
                status_color = (0, 255, 100) # Green
                cv2.putText(ui_panel, "CONTACT -> READY", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)
            else:
                status_color = (0, 100, 255) # Orange
                cv2.putText(ui_panel, "IDLE -> APPLY PRESS", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)
            
            # Dynamic Pressure Indicator Bar
            bar_width = int(min(contact_area / (CONTACT_THRESHOLD * 2.0 + 1e-5), 1.0) * 140)
            cv2.rectangle(ui_panel, (170, 45), (170 + bar_width, 55), status_color, -1)
            cv2.rectangle(ui_panel, (170, 45), (310, 55), (200, 200, 200), 1)

            # Class Count Display
            counts = {k: get_file_count(os.path.join(base_dir, v)) for k, v in CLASS_MAP.items()}
            bg_count = get_file_count(os.path.join(base_dir, "background"))
            cv2.putText(ui_panel, f"1:{counts['1']} 2:{counts['2']} 3:{counts['3']}", 
                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(ui_panel, f"4:{counts['4']} 5:{counts['5']} 0(Bg):{bg_count}", 
                        (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(ui_panel, "[C]Calibrate   [Q]Quit", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

            display_combined = cv2.vconcat([clean_raw_img, ui_panel])
            cv2.imshow(window_name, display_combined)
            cv2.imshow("Contact Physics Mask", thresh)

            # ---------------------------------------------------------
            # 6. Keyboard Event Handling (Data Persistence)
            # ---------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            
            if chr(key) in CLASS_MAP.keys():
                if is_contact:
                    cat_name = CLASS_MAP[chr(key)]
                    count = get_file_count(os.path.join(base_dir, cat_name))
                    filename = f"{cat_name}_{count:04d}.png"
                    
                    save_path = os.path.join(base_dir, cat_name, filename)
                    cv2.imwrite(save_path, clean_raw_img) # Lossless save
                    
                    print(f"[MANUAL-SAVE] {cat_name.upper()} | Saved: {filename}")
                else:
                    print("[WARNING] Insufficient pressure. Adjust 'Area_Thresh' slider or apply more force.")
            
            if key == ord('0'):  
                count = get_file_count(os.path.join(base_dir, "background"))
                filename = f"bg_{count:04d}.png"
                
                save_path = os.path.join(base_dir, "background", filename)
                cv2.imwrite(save_path, clean_raw_img)
                print(f"[MANUAL-SAVE] BACKGROUND | Saved: {filename}")
            
            if key == ord('c'):
                gray_baseline = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
                print("[INFO] Baseline recalibrated successfully!")
                
            if key == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        if hasattr(dev, 'disconnect'): 
            dev.disconnect()

if __name__ == '__main__':
    main()