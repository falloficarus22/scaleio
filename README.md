# Local Image Upscaler

A completely offline image upscaling tool using RealESRGAN that runs locally on your machine. No cloud services, no data leaving your computer.

## Features

- 🖼️ **Local Processing**: All processing happens on your machine
- 🚀 **High Quality**: Uses RealESRGAN for superior upscaling results
- 🔧 **Easy Setup**: Simple Python installation
- 📁 **Batch Processing**: Upscale multiple images at once
- 💾 **Multiple Models**: Support for 2x, 4x, and 8x upscaling
- 🎯 **Free & Open Source**: No costs, no subscriptions

## Requirements

- Python 3.7 or higher
- A computer with GPU (recommended) or CPU

## Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd scaleio
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install realesrgan
   ```

3. **Verify installation**
   ```bash
   python upscale_image.py --help
   ```

## Usage

### Single Image Upscaling

```bash
# Basic usage (4x upscaling with default model)
python upscale_image.py input_image.jpg

# Specify output file
python upscale_image.py input_image.jpg -o output_image.jpg

# Use different model (2x upscaling)
python upscale_image.py input_image.jpg -m RealESRGAN_x2plus

# Use 8x upscaling
python upscale_image.py input_image.jpg -m RealESRGAN_x8plus
```

### Batch Processing

```bash
# Upscale all images in a directory
python upscale_image.py /path/to/images --batch

# Specify output directory
python upscale_image.py /path/to/images --batch -o /path/to/output
```

### Command Line Options

- `input`: Input image file or directory
- `-o, --output`: Output file or directory (optional)
- `-m, --model`: Model to use (RealESRGAN_x4plus, RealESRGAN_x2plus, RealESRGAN_x8plus)
- `-s, --scale`: Upscaling factor (2, 4, 8)
- `--batch`: Process all images in directory

## Models

| Model | Scale Factor | Best For |
|-------|-------------|----------|
| RealESRGAN_x2plus | 2x | Moderate enhancement, faster processing |
| RealESRGAN_x4plus | 4x | General purpose upscaling (default) |
| RealESRGAN_x8plus | 8x | Maximum upscaling, slower processing |

## Supported Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

## How It Works

1. **Model Download**: On first run, the tool automatically downloads the selected RealESRGAN model
2. **Local Processing**: Images are processed entirely on your machine using PyTorch
3. **Smart Upscaling**: RealESRGAN uses deep learning to intelligently enhance image details
4. **Privacy**: No data is sent to any external service

## Performance Tips

- **GPU Acceleration**: If you have an NVIDIA GPU, ensure CUDA is installed for faster processing
- **Memory Usage**: Large images may require significant RAM
- **Batch Processing**: Process multiple images efficiently with the `--batch` option

## Troubleshooting

### Common Issues

1. **"Missing dependencies" error**
   ```bash
   pip install realesrgan
   ```

2. **CUDA out of memory**
   - Use CPU processing: The tool automatically falls back to CPU if GPU memory is insufficient
   - Process smaller images

3. **Model download fails**
   - Check your internet connection for the first download
   - Models are cached locally after first use

### Getting Help

If you encounter issues:
1. Check that all dependencies are installed
2. Verify the input file exists and is a supported format
3. Ensure you have sufficient disk space for models and output

## Technical Details

- **Framework**: PyTorch
- **Model**: RealESRGAN (Real-World Super-Resolution via Synthetic Data)
- **Architecture**: RRDBNet (Residual-in-Residual Dense Block Network)
- **Processing**: Tile-based processing for large images

## License

This project uses RealESRGAN, which is licensed under the BSD 2-Clause License.

## Privacy Notice

- ✅ All processing happens locally on your machine
- ✅ No images are uploaded to any cloud service
- ✅ No data is collected or transmitted
- ✅ Models are downloaded once and cached locally
