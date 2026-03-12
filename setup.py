#!/usr/bin/env python3
"""
Setup script for the Local Image Upscaler
"""

import subprocess
import sys
import os

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")
    
    try:
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Basic dependencies installed")
        
        # Install realesrgan
        subprocess.check_call([sys.executable, "-m", "pip", "install", "realesrgan"])
        print("✅ RealESRGAN installed")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        sys.exit(1)

def test_installation():
    """Test if the installation works"""
    print("🧪 Testing installation...")
    
    try:
        # Test imports
        import torch
        import cv2
        from PIL import Image
        import numpy as np
        from realesrgan import RealESRGANer
        print("✅ All imports successful")
        
        # Check device
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        print(f"✅ Using {device} for processing")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    print("🚀 Setting up Local Image Upscaler...")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    install_dependencies()
    
    # Test installation
    if test_installation():
        print("\n🎉 Setup completed successfully!")
        print("\nYou can now use the upscaler:")
        print("  python upscale_image.py your_image.jpg")
        print("\nFor help:")
        print("  python upscale_image.py --help")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
