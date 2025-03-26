# Image-Captioner_Generator
## Project Overview:
The Image Caption Generator is a deep learning-based application that automatically generates descriptive captions for images. Using a combination of Convolutional Neural Networks (CNNs) for feature extraction and Recurrent Neural Networks (RNNs) with LSTMs, the model effectively learns to generate human-like descriptions for images. This project is built using TensorFlow, Keras, Streamlit, and NLP techniques.

## Features
1) Deep Learning-based Image Captioning: Utilizes a pre-trained DenseNet201 model for feature extraction and an LSTM-based decoder for text generation.
2) Custom Data Preprocessing: Tokenization, sequence padding, and custom data generators for training.
3) Interactive Web App: Built using Streamlit, allowing users to upload images and generate captions in real-time.
4) Transfer Learning: Uses a pre-trained DenseNet201 model to extract meaningful image features.
5) Efficient Training: Implements Model Checkpointing, Early Stopping, and Learning Rate Reduction for optimized training.
6) Visualization: Includes Matplotlib and Seaborn for loss visualization and evaluation.

## Directory Structure:
   Image-Caption-Generator
1) │-- 📂 models
2) │   a) │-- model.keras  (Trained Image Captioning Model)
3) │   b) │-- feature_extractor.keras (Pre-trained CNN for Feature Extraction)
4) │   c) │-- tokenizer.pkl (Saved Tokenizer for text processing)
5) │-- main.py (Streamlit Web App)
6) │-- train.py (Model Training Script)
7) │-- utils.py (Helper Functions)
8) │-- README.md (Project Documentation)

## Installation & Setup
1) Ensure you have Python 3.7+ installed and the following dependencies: requirements.txt
2) Download Pre-trained Models: Before running the application, place the trained model (model.keras), feature extractor (feature_extractor.keras), and tokenizer (tokenizer.pkl) inside the models/ directory.

## Model Training
Step 1: Data Preparation:
The dataset used is Flickr8k, consisting of images and captions. The captions are preprocessed by converting to lowercase, removing unwanted characters, and tokenizing using Tokenizer from Keras.

Step 2: Feature Extraction:
A pre-trained DenseNet201 model is used to extract meaningful image representations, replacing its classifier layer with a Global Average Pooling Layer.

Step 3: Sequence Generation:
a) Captions are converted into sequences.
b) The sequences are padded to a fixed length.
c) The image features and sequence inputs are fed into an LSTM-based Decoder.

Step 4: Training the Model: 
python train.py

### Training includes:
Using a custom Keras Sequence Data Generator, 
EarlyStopping to prevent overfitting, 
ModelCheckpoint for saving the best model.

## Running the Web App
Once the model is trained, you can run the Streamlit web app to generate captions for uploaded images.
#### streamlit run main.py

### Features:
1) Upload an image
2) Automatically generate a caption using the trained model
3) View the image with the generated caption displayed

### Dataset:
To train the model flickr 8k dataset from kaggle is being used
