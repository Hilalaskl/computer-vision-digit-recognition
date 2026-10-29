# Handwritten Digit Recognition

A computer vision project that recognizes handwritten digits using Python, TensorFlow, and OpenCV.

The project uses a neural network trained on the MNIST dataset to classify handwritten digits from 0 to 9. It also processes a custom image containing handwritten digits and predicts each digit individually.

## Features

- Load and preprocess the MNIST dataset
- Build and train a neural network
- Evaluate model performance on test data
- Recognize handwritten digits from a custom image
- Display the top three predictions with confidence scores
- Save the trained model in Keras format

## Technologies

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

## Model Architecture

The neural network consists of the following layers:

1. Flatten layer — converts the 28 × 28 image into a one-dimensional vector
2. Dense layer — 128 neurons with ReLU activation
3. Dense layer — 64 neurons with ReLU activation
4. Dense output layer — 10 neurons with Softmax activation

The model is trained using the Adam optimizer and sparse categorical cross-entropy loss.

## Results

- **Test accuracy:** 97.47%
- **Input image:** `13.jpg`
- **Recognized number:** `13`

The model successfully recognized the handwritten number `13` in a custom image.

## Project Structure

```text
computer-vision-digit-recognition/
├── main.py
├── README.md
├── .gitignore
├── 13.jpg
└── digit_recognition.keras
```

## How to Run

### 1. Install dependencies

```bash
pip install tensorflow opencv-python matplotlib numpy
```

### 2. Prepare the input image

Place an image named `13.jpg` in the project directory. The current image-processing code is configured for this input image and its digit placement.

### 3. Run the project

```bash
python main.py
```

The program trains the model, evaluates its accuracy, displays predictions, and attempts to recognize the digits in the custom image.

## Future Improvements

- Improve recognition across different handwriting styles
- Support images containing numbers of different lengths
- Improve image preprocessing and digit segmentation
- Separate model training from image prediction

## Author

**Hilal Askl**

GitHub: [@Hilalaskl](https://github.com/Hilalaskl)
