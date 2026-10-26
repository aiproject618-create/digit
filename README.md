# Handwritten Digit Recognition Using Deep Learning

## 1. Project Overview

Handwritten Digit Recognition is an Artificial Intelligence project that identifies handwritten digits from 0 to 9 using a Convolutional Neural Network (CNN).

The application allows users to draw a digit or upload an image. The trained deep learning model analyzes the image and predicts which digit it represents.

This project demonstrates how Artificial Intelligence and Deep Learning can be used for image recognition.

## 2. Objectives

- To understand the basic concepts of Artificial Intelligence and Deep Learning.
- To train a CNN model to recognize handwritten digits.
- To predict digits from uploaded images and a drawing canvas.
- To display prediction probabilities in an interactive web application.
- To demonstrate a practical application of image classification.

## 3. Technologies Used

- **Python** — Programming language
- **TensorFlow/Keras** — Deep learning model
- **Streamlit** — Web application interface
- **NumPy** — Numerical operations
- **Pillow (PIL)** — Image processing
- **Matplotlib** — Probability visualization
- **MNIST Dataset** — Handwritten digit dataset

## 4. Project Structure

```text
handwritten-digit-recognition/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── digit_model.keras
```

**Note:** The `digit_model.keras` file is generated after training. It must be placed in the same directory as `app.py`.

## 5. Dataset

This project uses the MNIST handwritten digit dataset, which contains grayscale images of handwritten digits from 0 to 9.

- Training images: 60,000
- Testing images: 10,000
- Image dimensions: 28 × 28 pixels
- Number of classes: 10

The dataset is loaded automatically through TensorFlow/Keras during model training.

## 6. Installation and Setup

### Step 1: Install Python

Install a compatible Python version on your computer, or use Google Colab to train the model.

### Step 2: Download the Project

Download or clone this repository.

### Step 3: Install Dependencies

Open a terminal in the project directory and run:

```bash
pip install -r requirements.txt
```

### Step
