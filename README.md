# 🧠 Image Denoising Using Autoencoders

An image reconstruction and denoising project using an Undercomplete Autoencoder trained on the MNIST handwritten digit dataset.

The project demonstrates how an autoencoder learns a compact latent representation of images and reconstructs the original image from this compressed representation.

The trained model is deployed as an interactive Streamlit web application where users can upload handwritten digit images and visualize the reconstructed output.


## 🔗 Project Links

### 💻 GitHub Repository

https://github.com/TPS1234795/Image-Denoising-Using-Autoencoders

### 🚀 Live Streamlit Application

https://image-denoising-using-autoencoders-b33nam8duebo2e7dnfajxh.streamlit.app/


## 📌 Project Overview

Image denoising is the process of removing unwanted noise from an image while preserving important visual information.

In this project, an Undercomplete Autoencoder is used for image reconstruction. The model compresses a 28 × 28 grayscale image into a smaller latent representation and then reconstructs the image from that representation.

Project Workflow:

Input Image
     ↓
Preprocessing
     ↓
784-dimensional Vector
     ↓
Encoder
     ↓
Latent Representation
     ↓
Decoder
     ↓
Reconstructed Image
     ↓
MSE & MAE Evaluation


## 🎯 Objectives

- Understand the working principle of autoencoders.
- Learn compact representations of image data.
- Reconstruct handwritten digit images using an Undercomplete Autoencoder.
- Evaluate reconstruction quality using MSE and MAE.
- Develop an interactive Streamlit application.
- Deploy the trained model for practical use.


## 🧠 Model Architecture

The project uses an Undercomplete Autoencoder with a bottleneck latent representation.

Input Layer
784 neurons
    ↓
Dense Layer
256 neurons
ReLU
    ↓
Dense Layer
128 neurons
ReLU
    ↓
Latent Layer
32 neurons
ReLU
    ↓
Dense Layer
128 neurons
ReLU
    ↓
Dense Layer
256 neurons
ReLU
    ↓
Output Layer
784 neurons
Sigmoid


### Model Configuration

- Model Type: Undercomplete Autoencoder
- Input Size: 784
- Original Image Size: 28 × 28
- Hidden Layer 1: 256 neurons
- Hidden Layer 2: 128 neurons
- Latent Dimension: 32
- Decoder Layer 1: 128 neurons
- Decoder Layer 2: 256 neurons
- Output Size: 784
- Hidden Activation: ReLU
- Output Activation: Sigmoid


## 📊 Dataset

The project uses the MNIST handwritten digit dataset.

MNIST consists of grayscale images of handwritten digits.

Dataset Properties:

- Image Size: 28 × 28 pixels
- Image Type: Grayscale
- Input Features after Flattening: 784
- Pixel Value Range: 0–255
- Normalized Pixel Value Range: 0–1

The images are converted into a one-dimensional vector before being provided to the autoencoder.

28 × 28 Image
      ↓
Flatten
      ↓
784-dimensional Vector


## ⚙️ Image Preprocessing

Uploaded images are processed before being passed to the model.

The preprocessing steps are:

1. Convert the image to grayscale.
2. Resize the image to 28 × 28 pixels.
3. Normalize pixel values between 0 and 1.
4. Flatten the image into a 784-dimensional vector.
5. Pass the processed image to the autoencoder.


## 📏 Evaluation Metrics

The reconstruction quality is evaluated using two metrics.


### Mean Squared Error (MSE)

MSE measures the average squared difference between the original and reconstructed image.

MSE = Mean((Original Image - Reconstructed Image)²)

A lower MSE indicates a smaller reconstruction error.


### Mean Absolute Error (MAE)

MAE measures the average absolute difference between the original and reconstructed image.

MAE = Mean(|Original Image - Reconstructed Image|)

A lower MAE indicates a smaller reconstruction error.


## 🖥️ Streamlit Application

The trained Undercomplete Autoencoder is deployed using Streamlit.

The web application provides:

- Image upload functionality
- Model information
- Original image visualization
- Reconstructed image visualization
- MSE calculation
- MAE calculation
- Reconstructed image download


### Application Workflow

Upload Image
     ↓
Image Preprocessing
     ↓
Undercomplete Autoencoder
     ↓
Image Reconstruction
     ↓
Display Original and Reconstructed Images
     ↓
Calculate MSE and MAE


## 📁 Project Structure

Image-Denoising-Using-Autoencoders/
│
├── app.py
├── requirements.txt
├── undercomplete_autoencoder.weights.h5
└── README.md


### File Description

app.py
Streamlit application for image reconstruction.

requirements.txt
Contains the Python dependencies required to run the application.

undercomplete_autoencoder.weights.h5
Contains the trained weights of the Undercomplete Autoencoder.

README.md
Project documentation.


## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pillow
- Streamlit
- MNIST Dataset


## 💻 Installation

Clone the repository using:

git clone https://github.com/TPS1234795/Image-Denoising-Using-Autoencoders.git

Navigate to the project directory:

cd Image-Denoising-Using-Autoencoders

Install the required dependencies:

pip install -r requirements.txt


## ▶️ Run the Application

Start the Streamlit application using:

python -m streamlit run app.py

The application will open in a web browser.


## 📷 How to Use the Application

1. Open the Streamlit application.
2. Upload a handwritten digit image.
3. The image is converted to grayscale.
4. The image is resized to 28 × 28 pixels.
5. Pixel values are normalized.
6. The trained autoencoder reconstructs the image.
7. The original and reconstructed images are displayed.
8. MSE and MAE reconstruction metrics are calculated.
9. The reconstructed image can be downloaded.


## 🚀 Deployment

The application is deployed using Streamlit Community Cloud.

Deployment Workflow:

GitHub Repository
       ↓
requirements.txt
       ↓
app.py
       ↓
Trained Model Weights
       ↓
Streamlit Community Cloud
       ↓
Live Web Application


### Live Application

https://image-denoising-using-autoencoders-b33nam8duebo2e7dnfajxh.streamlit.app/


## 🔮 Future Enhancements

- Use larger and more complex datasets such as CIFAR-10 and real-world image datasets.
- Test different types of noise such as salt-and-pepper, speckle, Poisson, and mixed noise.
- Explore advanced architectures such as Variational Autoencoders (VAE), U-Net, and other deep autoencoder architectures.
- Perform hyperparameter tuning for latent dimension, learning rate, batch size, and regularization strength.
- Improve computational efficiency by reducing training time and memory requirements.
- Deploy trained models for real-time image denoising using edge devices.
- Study the effect of different latent-space dimensions on reconstruction quality and information retention.
- Evaluate model robustness under different noise intensities and combinations of noise types.
- Apply the models to real-world images such as medical images, surveillance images, document images, and other practical applications.


## 👨‍💻 Author

TANIPRAVA SAHOO

B.Tech – Computer Science & Engineering
Artificial Intelligence & Machine Learning


## ⭐ Acknowledgement

This project was developed as part of an academic exploration of Autoencoders, Image Reconstruction, Image Denoising, and Deep Learning.

If you find this project useful, consider giving the repository a star on GitHub.
