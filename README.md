# Genesis-project
AI generated image detector

https://venkatasai0852-venkata-sai-genesis-project.hf.space


This project focuses on detecting whether an image is "AI-generated or real" using a deep learning–based approach.  
The system takes an image as input and outputs a prediction along with a confidence score.

# Project Overview

With the rapid advancement of generative AI models, distinguishing real images from AI-generated images has become increasingly challenging.  
This project addresses the problem by leveraging "transfer learning", "frequency-domain analysis", and an "attention-based fusion mechanism" to improve detection accuracy. The final trained model is deployed as a "public web application" using "Hugging Face Spaces", allowing real-time inference through a simple user interface.


## Model Architecture & Approach
# Transfer Learning Backbone
- EfficientNet-B3 (pretrained on ImageNet) is used as the backbone model.
- Transfer learning helps leverage rich pretrained visual features while reducing training time.
- The final classification layer is modified for binary classification:
  - Real Image
  - AI Generated Image


# Input Feature Design (RGB + FFT)
To enhance robustness and capture subtle AI-generated artifacts, the model uses "dual-domain input features":

- "RGB Features"  
  Capture spatial and visual information such as textures, edges, and color patterns.

- "FFT (Fast Fourier Transform) Features"  
  Capture frequency-domain artifacts that are often present in AI-generated images but less noticeable in the spatial domain.

Both RGB and FFT representations are combined to improve generalization and detection accuracy.

# Attention Mechanism
An "attention mechanism" is applied between the "RGB feature stream and the FFT feature stream" to:
- Emphasize discriminative features
- Suppress irrelevant or noisy information
- Improve performance on challenging and ambiguous samples

This attention-based fusion enables the model to focus on subtle patterns indicative of AI-generated content.

# Training Strategy
- The model was trained using "transfer learning".
- Initially, the pretrained "EfficientNet-B3 weights were frozen" to stabilize training.
- Total training was performed for "20 epochs".
- In the later stages of training:
  - The pretrained layers were **unfrozen**
  - Fine-tuning was applied to improve feature adaptation and overall accuracy

This staged training strategy helped achieve high accuracy while avoiding overfitting.

# Model Performance
- Training Accuracy: 98.4%
- Validation Accuracy: 97.6%

The close alignment between training and validation accuracy indicates strong generalization performance.


# Inference & Confidence Control
- Softmax probabilities are used for final classification.
- A confidence thresholding mechanism is applied:
  - Predictions with low confidence are labeled as **“Uncertain”**
- This improves reliability and reduces false confident predictions.

# Project Flow Diagram
Input Image
↓
RGB Feature Extraction ──┐
├─ Attention-based Feature Fusion
FFT Feature Extraction ──┘
↓
EfficientNet-B3 Backbone
↓
Softmax Classification
↓
Prediction + Confidence Score

# Deployment
The trained model is deployed as a "public web application" using "Gradio" on "Hugging Face Spaces".  
Users can upload images through the web interface and instantly receive predictions.

# Limitations
- Model trained on a limited set of AI image generators
- Performance may degrade on heavily compressed or edited images
- The system cannot guarantee 100% accuracy

# Future Improvements
- Cross-generator robustness testing
- Video-based AI content detection
- Integration of transformer-based architectures
- Large-scale cloud deployment

<img width="1268" height="338" alt="Screenshot 2026-01-11 212257" src="https://github.com/user-attachments/assets/3ec6679d-2c2f-498d-b373-779ff4757a9d" />

<img width="646" height="712" alt="Screenshot 2026-01-11 212337" src="https://github.com/user-attachments/assets/da5d083f-b68c-4a48-a9bf-a95d7a6ee655" />

<img width="683" height="589" alt="Screenshot 2026-01-11 212348" src="https://github.com/user-attachments/assets/540b7529-b4b8-4469-9c3e-083c6cd948a7" />


These photos include the tested examples and the metrics like Precision, F1 score, Recall and confusion matrix on validation data
