import os
import sys
import subprocess
import platform

ENV_NAME = "tactile_env"

def main():
    print("======================================================================")
    print("[INFO] Starting environment setup for Tactile EMI Monitor...")
    print("======================================================================\n")

    # 1. Check Python version
    if sys.version_info < (3, 8):
        print(f"[ERROR] Python 3.8 or higher is required. You are using {sys.version_info.major}.{sys.version_info.minor}")
        sys.exit(1)
        
    print(f"[INFO] Detected Python version: {sys.version_info.major}.{sys.version_info.minor}")

    # 2. Create virtual environment
    print(f"[INFO] Creating Python virtual environment '{ENV_NAME}'...")
    try:
        subprocess.run([sys.executable, "-m", "venv", ENV_NAME], check=True)
    except subprocess.CalledProcessError:
        print(f"[ERROR] Failed to create virtual environment '{ENV_NAME}'.")
        sys.exit(1)

    # 3. Determine OS-specific paths for the virtual environment
    if platform.system() == "Windows":
        venv_python = os.path.join(ENV_NAME, "Scripts", "python.exe")
        activate_cmd = f"{ENV_NAME}\\Scripts\\activate"
    else:
        # macOS and Linux
        venv_python = os.path.join(ENV_NAME, "bin", "python")
        activate_cmd = f"source {ENV_NAME}/bin/activate"

    if not os.path.exists(venv_python):
        print(f"[ERROR] Virtual environment Python executable not found at: {venv_python}")
        sys.exit(1)

    # 4. Upgrade pip using the virtual environment's python
    print("[INFO] Upgrading pip...")
    subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], check=True)

    # 5. Install standard requirements
    print("\n[INFO] Installing dependencies from requirements.txt...")
    if not os.path.exists("requirements.txt"):
        print("[ERROR] 'requirements.txt' not found! Please ensure it is in the root directory.")
        sys.exit(1)
        
    try:
        subprocess.run([venv_python, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    except subprocess.CalledProcessError:
        print("[ERROR] Failed to install some dependencies. Please check the error log above.")
        sys.exit(1)

    # 6. Success Message & Activation Instructions
    print("\n======================================================================")
    print("[SUCCESS] Base environment setup complete!")
    print("\n[CRITICAL HARDWARE NOTE]:")
    print("The default PyTorch installed is the CPU version (or default CUDA).")
    print("For optimal V4 Inference performance (5 FPS), please ensure you install")
    print("the PyTorch version that matches your specific GPU's CUDA runtime.")
    print("Visit: https://pytorch.org/get-started/locally/")
    print("\n[NEXT STEP]:")
    print("To start using the system, you must activate the virtual environment manually.")
    print("Copy and run the following command in your terminal:")
    print(f"\n    {activate_cmd}\n")
    print("======================================================================")

if __name__ == "__main__":
    main()