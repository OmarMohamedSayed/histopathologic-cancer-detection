# Histopathologic Cancer Detection

A machine learning project for detecting metastatic cancer in histopathologic image patches using deep learning techniques.

## Project Overview

This project implements a binary classifier to identify metastatic cancer in small image patches (96x96 pixels) taken from larger digital pathology scans. The model uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify whether the center 32x32px region of an image contains tumor tissue.

## Dataset

The project uses the Histopathologic Cancer Detection dataset, which consists of:
- **Training images**: TIFF format images (96x96 pixels)
- **Labels**: CSV file with binary labels (0 = No Cancer, 1 = Cancer)
- **Test images**: Unlabeled images for prediction

### Data Structure Expected
```
/kaggle/input/histopathologic-cancer-detection/
├── train/
│   ├── image1.tif
│   ├── image2.tif
│   └── ...
├── test/
│   ├── test_image1.tif
│   ├── test_image2.tif
│   └── ...
└── train_labels.csv
```

## Features

- **Fast Training Mode**: Uses a subset of data (10% of full dataset) and limited epochs for quick experimentation
- **CNN Architecture**: Simple but effective convolutional neural network with:
  - 2 Convolutional layers with MaxPooling
  - Batch Normalization for training stability
  - Dropout for regularization
  - Binary classification output
- **Data Visualization**: Exploratory data analysis with sample images and label distribution
- **Performance Monitoring**: Training and validation accuracy/loss tracking

## Installation

### Prerequisites
- Python 3.7 or higher
- GPU support recommended for faster training (optional)

### Setup
1. Clone or download this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Alternative Installation (with specific versions)
```bash
pip install pandas>=1.5.0 numpy>=1.21.0 matplotlib>=3.5.0 seaborn>=0.11.0 scikit-learn>=1.0.0 tensorflow>=2.10.0 Pillow>=8.0.0
```

## Usage

### Running the Notebook
1. Ensure your dataset is organized in the expected structure
2. Update the `DATA_DIR` variable in the notebook to point to your dataset location
3. Open and run the Jupyter notebook:
```bash
jupyter notebook notebook3b11f2b77c.ipynb
```

### Key Configuration Parameters

You can modify these parameters in the notebook for different experiments:

```python
IMAGE_SIZE = 96        # Image dimensions (96x96 pixels)
BATCH_SIZE = 64        # Batch size for training
EPOCHS = 1             # Number of training epochs (increase for better performance)
STEPS_PER_EPOCH = 50   # Steps per epoch (remove limit for full training)
```

## Model Architecture

The CNN model consists of:

1. **Input Layer**: 96x96x3 RGB images
2. **Conv2D Layer**: 32 filters, 3x3 kernel, ReLU activation
3. **MaxPooling2D**: 2x2 pooling
4. **Conv2D Layer**: 64 filters, 3x3 kernel, ReLU activation
5. **MaxPooling2D**: 2x2 pooling
6. **Flatten Layer**: Convert to 1D
7. **Dense Layer**: 128 units, ReLU activation
8. **Batch Normalization**: Training stabilization
9. **Dropout**: 50% dropout rate
10. **Output Layer**: 1 unit, sigmoid activation (binary classification)

## Performance Optimization

The notebook is configured for fast execution with limited performance. To improve results:

### For Better Accuracy
1. **Use more data**: Increase `test_size` in `train_test_split` (or use full dataset)
2. **More epochs**: Increase `EPOCHS` to 10-20 or more
3. **Full training**: Remove `STEPS_PER_EPOCH` limitation
4. **Data augmentation**: Add rotation, flips, and zoom to `ImageDataGenerator`
5. **Advanced architecture**: Consider transfer learning with pre-trained models (ResNet50, VGG16)

### Example Full Training Configuration
```python
EPOCHS = 15
# Remove STEPS_PER_EPOCH limitation
# Use full dataset instead of subset
```

## Results

The current fast configuration provides:
- **Quick execution**: Completes in under 2 minutes
- **Basic performance**: Limited accuracy due to minimal training
- **Proof of concept**: Demonstrates the complete ML pipeline

## File Structure

```
histopathologic-cancer-detection/
├── notebook3b11f2b77c.ipynb    # Main Jupyter notebook
├── requirements.txt             # Python dependencies
└── README.md                   # Project documentation
```

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Test with different configurations
5. Submit a pull request

## License

This project is open source. Please check the original dataset license for data usage restrictions.

## Acknowledgments

- Dataset: Histopathologic Cancer Detection (Kaggle competition)
- Built with TensorFlow/Keras
- Inspired by medical imaging and deep learning research

## Support

For questions or issues:
1. Check the notebook comments for implementation details
2. Verify dataset structure and paths
3. Ensure all dependencies are properly installed
4. Consider GPU availability for larger experiments