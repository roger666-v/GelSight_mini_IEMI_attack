import os
import cv2
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import f_oneway
from sklearn.feature_selection import f_classif

# ==============================================================================
# Dataset-Level RGB Channel & Combination Evaluator
# 
# Description:
# This script performs a dataset-wide macro-analysis to determine the validity 
# of individual RGB channels and their combinations (RG, RB, GB, RGB) for 
# optical tactile sensor data. It utilizes:
# 1. Hardware Signal-to-Noise Ratio (SNR) 
# 2. ANOVA F-Statistic (Machine Learning Feature Separability)
# ==============================================================================

def get_channel_features(channel):
    """
    Extracts core features from a single channel[cite: 6]:
    - Variance: Represents global contrast and illumination distribution[cite: 6].
    - Mean Sobel Magnitude: Represents high-frequency physical edge energy[cite: 6].
    """
    variance = np.var(channel)
    sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
    edge_energy = np.mean(cv2.magnitude(sobel_x, sobel_y))
    
    return variance, edge_energy

def evaluate_dataset(dataset_path):
    print(f"[INFO] Scanning dataset directory: {dataset_path} ...")
    
    data_records = []
    valid_extensions = ('.png', '.jpg', '.jpeg')
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    if 'background' not in classes:
        print("[ERROR] 'background' directory not found. Required for SNR baseline.")
        return

    # ==========================================
    # 1. Feature Extraction Loop
    # ==========================================
    for class_name in classes:
        class_dir = os.path.join(dataset_path, class_name)
        for file in os.listdir(class_dir):
            if file.lower().endswith(valid_extensions):
                img_path = os.path.join(class_dir, file)
                img = cv2.imread(img_path)
                if img is None: continue
                
                B, G, R = cv2.split(img)
                
                r_var, r_edge = get_channel_features(R)
                g_var, g_edge = get_channel_features(G)
                b_var, b_edge = get_channel_features(B)
                
                data_records.append({
                    'Class': class_name,
                    'Is_Background': 1 if class_name == 'background' else 0,
                    'R_Var': r_var, 'R_Edge': r_edge,
                    'G_Var': g_var, 'G_Edge': g_edge,
                    'B_Var': b_var, 'B_Edge': b_edge
                })

    df = pd.DataFrame(data_records)
    print(f"[INFO] Successfully processed {len(df)} images across {len(classes)} classes.\n")

    # Define all channel combinations to evaluate
    base_channels = ['R', 'G', 'B']
    combinations_to_test = base_channels + ['RG', 'RB', 'GB', 'RGB']

    # ==========================================
    # 2. Hardware Signal-to-Noise Ratio (SNR)
    # ==========================================
    print("=" * 60)
    print("1. Hardware Integrity: Dataset SNR (dB)")
    print("=" * 60)
    
    bg_df = df[df['Is_Background'] == 1]
    obj_df = df[df['Is_Background'] == 0]
    
    snr_results = {}
    for combo in combinations_to_test:
        # Calculate combined signal and noise power across selected channels
        noise_power = sum([bg_df[f'{ch}_Edge'].mean() for ch in combo]) / len(combo) + 1e-6
        signal_power = sum([obj_df[f'{ch}_Edge'].mean() for ch in combo]) / len(combo)
        
        snr_ratio = signal_power / noise_power
        snr_db = 10 * np.log10(snr_ratio)
        snr_results[combo] = snr_db
        
        print(f" - {combo:<4} Combination SNR : {snr_db:>6.2f} dB")

    # ==========================================
    # 3. Machine Learning Separability (ANOVA F-Score)
    # ==========================================
    print("\n" + "=" * 60)
    print("2. ML Feature Separability: ANOVA F-Statistic")
    print("=" * 60)
    
    ml_df = df[df['Is_Background'] == 0]
    y = ml_df['Class']
    f_results = {}
    
    for combo in combinations_to_test:
        # Aggregate feature columns for the specific combination
        feature_cols = []
        for ch in combo:
            feature_cols.extend([f'{ch}_Var', f'{ch}_Edge'])
            
        X = ml_df[feature_cols]
        f_scores, _ = f_classif(X, y)
        
        # Average F-score across all features in this combination
        avg_f_score = np.mean(f_scores)
        f_results[combo] = avg_f_score
        
        print(f" - {combo:<4} Combination F-Score : {avg_f_score:>8.2f}")

    # ==========================================
    # 4. Executive Summary
    # ==========================================
    print("\n" + "=" * 60)
    print("Executive Summary")
    print("=" * 60)
    
    best_snr_combo = max(snr_results, key=snr_results.get)
    best_f_combo = max(f_results, key=f_results.get)
    worst_f_combo = min(f_results, key=f_results.get)
    
    print(f" -> Highest SNR (Hardware Fidelity) : {best_snr_combo} Configuration")
    print(f" -> Highest ML Separability (ANOVA) : {best_f_combo} Configuration")
    print(f" -> Lowest Performance (Noise Source) : {worst_f_combo} Configuration")
    print(" -> Recommendation: Mask out channels that heavily degrade the combined F-Score.\n")

if __name__ == "__main__":
    dataset_directory = "data/dataset" 
    evaluate_dataset(dataset_directory)