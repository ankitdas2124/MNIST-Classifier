# MNIST Handwritten Digit Classifier

A deep learning project that trains a Convolutional Neural Network (CNN) on the MNIST dataset and provides a real-time interactive web interface for handwritten digit recognition.

## Overview

This project consists of two main components:

1. **Model Training** (`digit_predict_model.py`) - Trains a CNN on the MNIST dataset
2. **Web Interface** (`app.py`) - A Gradio-based application for real-time digit prediction

## Features

- **CNN-based Classification**: 3-layer convolutional neural network for accurate digit recognition
- **Real-time Prediction**: Interactive drawing canvas for live digit classification
- **Confidence Scores**: Displays top 3 predictions with confidence levels
- **Pre-trained Model**: Ready-to-use `mnist_model.keras` for immediate predictions

## Project Structure

```
MNIST-Classifier/
├── app.py                    # Gradio web interface
├── digit_predict_model.py    # CNN model training script
├── mnist_model.keras         # Pre-trained model file
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## Requirements

- Python 3.7+
- TensorFlow/Keras
- Gradio
- NumPy
- Pillow (PIL)
- Matplotlib

## Installation

1. Clone or navigate to the project directory:
```bash
cd MNIST-Classifier
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
# or
source .venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train a new CNN model on the MNIST dataset:

```bash
python digit_predict_model.py
```

This script will:
- Load the MNIST dataset
- Preprocess the data (normalization and reshaping)
- Build and train a 3-layer CNN
- Display sample training images
- Evaluate the model on test data
- Save the trained model as `mnist_model.keras`

**Training Details:**
- Epochs: 5
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Metrics: Accuracy

### Running the Web Interface

To launch the interactive digit recognition web app:

```bash
python app.py
```

The application will:
- Load the pre-trained model
- Open a web browser with the Gradio interface
- Allow you to draw digits on a canvas
- Display real-time predictions with confidence scores

**Features:**
- Draw or sketch a single digit (0-9)
- View top 3 predictions with confidence percentages
- Clear and redraw as needed

## Model Architecture

The CNN consists of:

```
Input Layer (28×28×1)
↓
Conv2D (32 filters, 3×3 kernel) + ReLU
↓
MaxPooling2D (2×2)
↓
Conv2D (64 filters, 3×3 kernel) + ReLU
↓
MaxPooling2D (2×2)
↓
Conv2D (64 filters, 3×3 kernel) + ReLU
↓
Flatten
↓
Dense (64 units) + ReLU
↓
Dense (10 units) + Softmax
↓
Output (10 classes: digits 0-9)
```

## Data Preprocessing

- **Normalization**: Pixel values scaled to [0, 1] range by dividing by 255
- **Reshaping**: Images reshaped from (28, 28) to (28, 28, 1) for CNN compatibility
- **Inversion**: User sketches are inverted before prediction to match MNIST training data

## Performance

The model achieves high accuracy on the MNIST test set after 5 epochs of training. Expected accuracy: ~98-99%

## Files Description

| File | Purpose |
|------|---------|
| `app.py` | Gradio web interface for digit prediction |
| `digit_predict_model.py` | CNN model training and evaluation script |
| `mnist_model.keras` | Pre-trained Keras model (binary file) |
| `requirements.txt` | Python package dependencies |

## Example Usage

1. **Train the model** (optional, model is pre-trained):
   ```bash
   python digit_predict_model.py
   ```

2. **Launch the application**:
   ```bash
   python app.py
   ```

3. **Use the web interface**:
   - Draw a digit in the provided canvas
   - View instant predictions and confidence scores
   - Modify drawings as needed

## Troubleshooting

- **Model file not found**: Ensure `mnist_model.keras` exists in the project directory, or train a new model using `digit_predict_model.py`
- **Dependency errors**: Run `pip install -r requirements.txt` again
- **Port conflicts**: If Gradio can't launch on the default port, check your firewall settings

## Future Improvements

- Add support for multiple digit recognition
- Implement data augmentation for better generalization
- Deploy to cloud platforms (Hugging Face Spaces, Streamlit Cloud)
- Add model accuracy metrics visualization
- Support for other handwriting datasets (SVHN, EMNIST)

## License

This project is provided as-is for educational purposes.

## Author

Ankit Das,
BCSE student at Jadavpur University,kolkata
