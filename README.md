# Cat vs Dog Image Classification using Transfer Learning (VGG16)

## Project Overview
This project implements a binary image classification system to distinguish between cats and dogs using a convolutional neural network based on transfer learning with VGG16. The model is trained on labeled image data and deployed as an interactive web application using Gradio.

## Objective
- Classify images into Cat or Dog categories  
- Utilize pretrained VGG16 (ImageNet weights) for feature extraction  
- Apply data augmentation to improve generalization  
- Build a deployment-ready inference pipeline  

## Model Architecture
- Base Model: VGG16 (pretrained on ImageNet)  
- Input Shape: 224 × 224 × 3  
- Preprocessing inside the model:
  - Rescaling (1/255)
  - Random Rotation
  - Random Flip
  - Random Zoom
  - Random Translation  
- Feature Aggregation: Global Average Pooling  
- Classifier Head:
  - Dense layers with Batch Normalization, ReLU activation, and Dropout  
- Output Layer:
  - Sigmoid activation for binary classification  

## Training Details
- Loss Function: Binary Crossentropy  
- Optimizer: Adam  
- Batch Size: 32  
- Epochs: 20  
- Training Strategy:
  - VGG16 base model frozen (feature extraction)  

## Model Performance
Best observed validation performance:

Epoch 11/20  
Training Accuracy: 0.8590  
Validation Accuracy: 0.9246  
Validation Loss: 0.1795  

## Inference Pipeline
- Input images are resized to 224 × 224  
- Pixel normalization is handled inside the model  
- Sigmoid output threshold of 0.5 is used for classification  

Label logic:
DOG if prediction > 0.5  
CAT otherwise  

## Deployment
The trained model is deployed using Gradio as a simple web interface where users can upload an image and receive the predicted label.

## Gradio Interface
- Image upload input
- Single label output (Cat or Dog)
- Clean and minimal UI for demonstration

## Project Structure 
cat_dog_model/  
README.md  

## Key Highlights
- End-to-end image classification pipeline  
- Proper use of transfer learning with VGG16  
- Data augmentation applied inside the model  
- Deployment-ready architecture using Gradio  
- Clear separation between training and inference  

## Future Improvements
- Fine-tuning upper layers of VGG16  
- Threshold tuning based on validation metrics  
- Model explainability using Grad-CAM  
- Improved UI with confidence visualization  

## Author
Arshpreet Walia
Machine Learning Enthusiast
