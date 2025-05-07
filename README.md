# Identification_of_Fake_Currency_Found_in_India

This project aims to develop machine learning models capable of detecting fake Indian currency notes. The models are built using the ResNet50 architecture, fine-tuned with custom layers for binary classification to distinguish between real and fake notes for denominations of ₹50, ₹100 and ₹500.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Data Augmentation](#data-augmentation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- results

## Introduction

Counterfeit currency is a significant problem in many countries, including India. This project focuses on detecting counterfeit ₹50, ₹100, ₹500 notes using deep learning techniques. The models are trained on datasets of real and fake currency images and use the ResNet50 architecture for robust feature extraction.

> **Reference:** As highlighted in the paper *"Detection of Fake Indian Currency Using Deep Convolutional Neural Network"* presented at MysuruCon 2023, CNNs combined with ResNet models have shown remarkable accuracy in this domain.

## Dataset

The dataset consists of high-quality scanned images of real and fake currency notes across multiple denominations. The structure includes:

- `Training/`: Real and fake notes used for model training
- `Validation/`: Real and fake notes used during model validation
- `Testing/`: Unseen real and fake notes used to evaluate final model performance

> **Note:** Fake note images were synthetically generated using image editing software by altering key security features, as described in the referenced paper.

## Model Architecture

The model is based on a pre-trained **ResNet50** architecture from Keras with the top layer removed. Custom layers are added for binary classification:

1. `GlobalAveragePooling2D`
2. `Dense(1024, activation='relu')`
3. `Dropout(0.5)`
4. `Dense(1, activation='sigmoid')`

This approach leverages transfer learning to speed up training and improve generalization.

## Data Augmentation

To improve model robustness, augmentation techniques are applied using Keras' `ImageDataGenerator`:

- Rotation: up to 90°
- Horizontal flip
- Vertical flip

These techniques help simulate real-world conditions such as orientation and lighting variations.

## Training

Models are trained with the following configuration:

- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam
- **Batch Size:** 8
- **Epochs:** Up to 100 (with early stopping)
- **Callbacks:** `ModelCheckpoint`, `EarlyStopping`

## Evaluation

The models are evaluated using:

- **Accuracy**
- **Confusion Matrix**
- **Mean Squared Error (MSE)**

The confusion matrix helps assess how well the model differentiates between real and fake notes, while MSE quantifies prediction error.

## Usage

To use a trained model:

1. Load the model using `load_model('best_model_500.keras')`
2. Preprocess an input image
3. Run the prediction function
results

![Screenshot 2025-05-07 225232](https://github.com/user-attachments/assets/9f987556-dd3a-4059-b10e-65fe473a2478)

```python
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input

def predict_image(model, img_path):
    img = image.load_img(img_path, target_size=(300, 300))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    prediction = model.predict(img_array)
    return "Real" if prediction[0][0] > 0.9 else "Fake"

# Example usage
test_img_path = "/path/to/image.jpg"
print("Prediction:", predict_image(finetune_model, test_img_path))

Results
    Accuracy
The model achieves high accuracy across all denominations. According to the referenced study, similar models achieved up to 98.3% accuracy.

Confusion Matrix


           Predicted
           Fake  Real
Actual
Fake       TP    FP
Real       FN    TN
 Mean Squared Error
MSE helps quantify prediction errors; lower values indicate better model performance.

Conclusion
This project demonstrates the power of CNNs and ResNet50 in detecting counterfeit Indian currency. With proper training and data augmentation, deep learning models can effectively distinguish between real and fake notes - a step toward practical fraud prevention systems.

As stated in the referenced IEEE paper, CNN-based systems like this can be deployed in banks, ATMs, and smartphones for real-time detection with high accuracy.
