# Emotion Classification Using Custom & ResNet-50 Models

## Overview
This project focuses on classifying human emotions using deep learning techniques. Initially, multiple custom CNN architectures were developed and tested before transitioning to a pre-trained ResNet-50 model for improved accuracy. The model is trained on a dataset of facial expressions categorized into seven classes.

## Dataset
The dataset consists of images categorized into the following seven emotions:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

Images are stored in the `train/` directory, structured for use with PyTorch's `ImageFolder`.

## Model Iterations
### Iteration 1
- A custom CNN model with 3 convolutional layers.
- Basic data augmentation techniques.
- Used batch normalization and dropout for regularization.

### Iteration 2
- Expanded to 4 convolutional layers.
- Improved batch normalization and dropout values.
- Additional data augmentation applied.

### Iteration 3
- Kept 4 convolutional layers.
- Removed data augmentation from iteration 2

### Iteration 4 (Final Model: ResNet-50)
- Transitioned to a pre-trained ResNet-50 model.
- Fine-tuned the final layers for emotion classification.
- Optimized hyperparameters for improved performance.
- Implemented learning rate scheduling and mixed precision training.

## Implementation
The main components of the project:
- `ResNet-50.py`: Fine-tuned ResNet-50 model implementation.
- `MainProgram.py`: Main script to utilize the model for emotion classification with computer vision.
- `Conf_Matrix_Eval.py`: Generates a confusion matrix to analyze performance based on "x.pth" models.
- `resnet50_emotion_model.pth`: Trained model weights.
- `confusion_matrix_ResNet50.png`: Visualization of model classification performance.
- `haarcascade_frontalface_default.xml`: Haar cascade for face detection.

## Training Details
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Learning Rate**: 0.0003 (decayed using a scheduler)
- **Batch Size**: 64
- **Epochs**: 30
- **Data Augmentation**:
  - Random horizontal flip
  - Random rotation (10 degrees)
  - Random affine transformation
  - Color jittering (brightness, contrast variation)
  - Normalization

## Results
The final model using ResNet-50 achieved a significant improvement in accuracy compared to earlier iterations. Below is the confusion matrix for the ResNet-50 model:

![Confusion Matrix](confusion_matrix_ResNet50.png)

## How to Run
1. Clone the repository:
   ```sh
   git clone https://github.com/marko-604/Emotion-Recognition.git
   cd "ResumeProjects/Computer Vision/Emotion"
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Train the model & Evaluate & Generate confusion matrix:
   ```sh
   python ResNet-50.py
   ```
4. Run the emotion recognition:
   ```sh
   python MainProgram.py
   ```

## Future Improvements
- Increase dataset size for better generalization.
- Experiment with deeper CNN architectures.
- Fine-tune other pre-trained models like EfficientNet.

## This project is for educational and research purposes only.

