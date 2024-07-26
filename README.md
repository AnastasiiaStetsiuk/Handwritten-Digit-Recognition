# Handwritten Digit Recognition with Neural Networks

This repository contains code for training a neural network to recognize handwritten digits using the MNIST dataset. The network is implemented using TensorFlow and Keras, and includes a simple graphical interface for drawing digits.

## Overview

The MNIST dataset is a classic dataset of handwritten digits, consisting of 60,000 training images and 10,000 test images. Each image is a 28x28 pixel grayscale image representing a digit from 0 to 9.

This project demonstrates how to build, train, and evaluate a neural network for digit recognition using TensorFlow and Keras, as well as a Tkinter-based GUI for drawing and recognizing digits in real-time.

## Installation

To run the code, you'll need to have Python installed along with the following libraries:

- TensorFlow
- NumPy
- Matplotlib
- Tkinter
- Pillow

You can install these dependencies using pip:

```bash
pip install tensorflow numpy matplotlib pillow
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/anastasiiastetsiuk/handwritten-digit-recognition.git
cd handwritten-digit-recognition
```

2. Run the training and GUI script:

```bash
python main.py
```

This script will load the MNIST dataset, build and train the neural network, evaluate it on the test data, and launch a Tkinter-based GUI for drawing and recognizing digits.

## Model Architecture

The neural network consists of the following layers:

1. Input layer: 28x28 pixels (flattened to a vector of size 784).
2. Reshape layer: Reshapes the input to (28, 28, 1) for the convolutional layers.
3. Convolutional layer: 32 filters, 3x3 kernel, ReLU activation.
4. MaxPooling layer: 2x2 pool size.
5. Convolutional layer: 64 filters, 3x3 kernel, ReLU activation.
6. MaxPooling layer: 2x2 pool size.
7. Flatten layer: Flattens the input before the dense layers.
8. Dense layer: 128 neurons, ReLU activation.
9. Dropout layer: 0.5 dropout rate for regularization.
10. Output layer: 10 neurons (one for each digit), softmax activation.

## GUI

The GUI is built using Tkinter and allows users to draw digits on a canvas. The drawn digit is then recognized by the trained model.

- **Draw**: Use the mouse to draw a digit on the canvas.
- **Recognize**: Click the "Recognize" button to identify the drawn digit.
- **Clear**: Click the "Clear" button to clear the canvas.

## Results

After training, the model achieves an accuracy of approximately XX% on the test set.

## References

- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
