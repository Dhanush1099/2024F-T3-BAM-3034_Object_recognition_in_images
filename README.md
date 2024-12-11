# CIFAR-10 Image Classifier

This repository contains an image classification project that uses a deep learning model trained on the CIFAR-10 dataset. The project includes a pre-trained Convolutional Neural Network (CNN) for accurate classification and a web application for real-time predictions.

---

## 📋 Project Overview

The CIFAR-10 dataset is a collection of 60,000 32x32 color images divided into 10 classes, such as airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. This project demonstrates:
- Model training using TensorFlow/Keras.
- Real-time image classification via a Streamlit web app.
- Deployment of a pre-trained CNN model (`cnn_model.h5`).

---

## 🚀 Features

1. **Deep Learning Model**:
   - A Convolutional Neural Network (CNN) with multiple layers including convolution, pooling, flattening, and dense layers.
   - Trained to classify images into one of 10 mutually exclusive classes.

2. **Web Application**:
   - Built with Streamlit for real-time image uploads and predictions.
   - Simple interface for non-technical users.

3. **Pretrained Model**:
   - The trained CNN model (`cnn_model.h5`) is included for easy deployment.

4. **Evaluation Metrics**:
   - Accuracy, precision, recall, F1-score, and confusion matrix are used to evaluate performance.

---

## 📁 Project Structure

```plaintext
|-- app.py                 # Streamlit app for real-time predictions
|-- cnn_model.h5           # Pre-trained CNN model
|-- Code.ipynb             # Jupyter Notebook for model training
|-- README.md              # Project documentation

**Installation and Setup**
Clone the Repository:

```
git clone https://github.com/yourusername/cifar10-image-classifier.git
cd cifar10-image-classifier
```

Install Dependencies: Ensure you have Python 3.7+ installed. Install the required Python packages:

```
pip install -r requirements.txt

```
Run the Web Application: Start the Streamlit app:
```
streamlit run app.py

```
Open the app in your browser at http://localhost:8501.

**Usage**
## Web Application
Open the web application.
Upload an image in .png, .jpg, or .jpeg format.
View the uploaded image and its predicted class.

## Dataset
Training Dataset: 50,000 CIFAR-10 images.
Test Dataset: 10,000 CIFAR-10 images, split evenly across 10 classes.

**Evaluation Metrics**
Accuracy:
Overall classification accuracy: 70%.

Class-Wise Metrics:
High F1-scores for classes such as ship (80%) and automobile (83%).
Opportunities for improvement in classes like bird (57%) and cat (50%).

Classification Report:

            precision    recall  f1-score   support
 airplane       0.74      0.75      0.75      1000
automobile 0.86 0.80 0.83 1000 bird 0.66 0.50 0.57 1000 cat 0.45 0.55 0.50 1000 deer 0.62 0.67 0.65 1000 dog 0.61 0.58 0.59 1000 frog 0.75 0.78 0.77 1000 horse 0.70 0.78 0.74 1000 ship 0.80 0.81 0.80 1000 truck 0.83 0.76 0.79 1000 accuracy 0.70 10000 macro avg 0.70 0.70 0.70 10000 weighted avg 0.70 0.70 0.70 10000
