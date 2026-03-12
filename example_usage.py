#!/usr/bin/env python3
"""
Example usage of the Local Image Upscaler
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

def create_sample_image():
    """Create a sample low-resolution image for testing"""
    # Create a simple test image
    width, height = 64, 64
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw some patterns
    for i in range(0, width, 8):
        draw.line([(i, 0), (i, height)], fill='black', width=1)
    for i in range(0, height, 8):
        draw.line([(0, i), (width, i)], fill='black', width=1)
    
    # Add some text
    try:
        draw.text((10, 20), "TEST", fill='red')
    except:
        # If font is not available, skip text
        pass
    
    # Save the sample image
    sample_path = "sample_low_res.jpg"
    img.save(sample_path, quality=50)  # Low quality to simulate poor image
    print(f"Created sample image: {sample_path}")
    return sample_path

def example_single_image():
    """Example: Upscale a single image"""
    print("\n=== Example: Single Image Upscaling ===")
    
    # Create sample image if it doesn't exist
    if not Path("sample_low_res.jpg").exists():
        create_sample_image()
    
    # Import the upscaler
    from upscale_image import RealESRGANUpscaler
    
    # Initialize upscaler
    upscaler = RealESRGANUpscaler(model_name='RealESRGAN_x4plus', scale_factor=4)
    
    # Upscale the image
    try:
        output_path = upscaler.upscale_image("sample_low_res.jpg")
        print(f"✅ Single image upscaling completed: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def example_batch_processing():
    """Example: Batch process multiple images"""
    print("\n=== Example: Batch Processing ===")
    
    # Create a sample directory with images
    sample_dir = Path("sample_images")
    sample_dir.mkdir(exist_ok=True)
    
    # Create multiple sample images
    for i in range(3):
        img = Image.new('RGB', (32, 32), color=(i*80, i*80, i*80))
        img_path = sample_dir / f"sample_{i+1}.jpg"
        img.save(img_path, quality=40)
    
    print(f"Created sample images in: {sample_dir}")
    
    # Import the upscaler
    from upscale_image import RealESRGANUpscaler
    
    # Initialize upscaler
    upscaler = RealESRGANUpscaler(model_name='RealESRGAN_x2plus', scale_factor=2)
    
    # Batch upscale
    try:
        upscaler.upscale_batch(str(sample_dir))
        print("✅ Batch processing completed")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def example_different_models():
    """Example: Using different upscaling models"""
    print("\n=== Example: Different Models ===")
    
    if not Path("sample_low_res.jpg").exists():
        create_sample_image()
    
    models = [
        ('RealESRGAN_x2plus', 2),
        ('RealESRGAN_x4plus', 4),
    ]
    
    from upscale_image import RealESRGANUpscaler
    
    for model_name, scale in models:
        print(f"Testing {model_name} ({scale}x upscaling)...")
        try:
            upscaler = RealESRGANUpscaler(model_name=model_name, scale_factor=scale)
            output_path = upscaler.upscale_image(
                "sample_low_res.jpg", 
                f"sample_{model_name}_{scale}x.jpg"
            )
            print(f"✅ {model_name} completed: {output_path}")
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")

def main():
    """Run all examples"""
    print("🖼️  Local Image Upscaler - Examples")
    print("=" * 50)
    
    # Check if dependencies are installed
    try:
        import torch
        import cv2
        from PIL import Image
        print("✅ Dependencies detected")
    except ImportError:
        print("❌ Please run 'python setup.py' first to install dependencies")
        return
    
    # Run examples
    examples = [
        example_single_image,
        example_batch_processing,
        example_different_models,
    ]
    
    success_count = 0
    for example in examples:
        try:
            if example():
                success_count += 1
        except KeyboardInterrupt:
            print("\n⚠️  Example interrupted by user")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
    
    print(f"\n📊 Results: {success_count}/{len(examples)} examples completed successfully")
    
    # Show created files
    print("\n📁 Created files:")
    for file in Path(".").glob("sample*"):
        if file.is_file():
            size = file.stat().st_size
            print(f"  {file.name} ({size:,} bytes)")

if __name__ == "__main__":
    main()
