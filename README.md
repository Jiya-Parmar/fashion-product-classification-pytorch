# End-to-End E-commerce Product Classifier

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Tech: PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Language: Python](https://img.shields.io/badge/Python-3.9-blue.svg)

A deep learning pipeline in PyTorch to classify e-commerce product images using transfer learning on a real-world Kaggle dataset. This project showcases an end-to-end workflow from data acquisition to model training and prediction.

---

## Table of Contents
- [Project Overview](#project-overview)
- [The Journey: Solving the Domain Gap](#the-journey-solving-the-domain-gap)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Setup and Installation](#setup-and-installation)
- [How to Use](#how-to-use)
- [Results](#results)
- [Future Improvements](#future-improvements)

---

## Project Overview
Automated product categorization is a critical task for any e-commerce platform, helping to improve user experience and streamline inventory management. This project implements a robust solution to this problem by training a powerful deep learning model to classify product images into their respective categories.

The final model is a fine-tuned ResNet18 capable of identifying products from real-world photographs with high accuracy.

---

## The Journey: Solving the Domain Gap
A key challenge in machine learning is ensuring a model trained in a lab environment can perform well on real-world data. This project directly confronts this challenge.

1.  **Initial Model (Fashion-MNIST):** An initial model was trained on the simple, clean Fashion-MNIST dataset. While it achieved over 90% accuracy on its test set, it failed to correctly classify real-world product photos.

2.  **Problem Diagnosis:** This highlighted the critical **domain gap**—a mismatch between the simple, icon-like training data and the complex, noisy real-world data.

3.  **The Solution (Transfer Learning on Real Data):** To overcome this, the project was evolved to use a rich dataset of ~44,000 real product images from Kaggle. By employing **transfer learning**, a ResNet18 model pre-trained on ImageNet was fine-tuned, allowing it to leverage its powerful feature-extraction capabilities and adapt them to this specific e-commerce task.

---

## Key Features
- **Fine-tuned ResNet18 model** for high-accuracy image classification.
- **Trained on a real-world Kaggle dataset** with thousands of product images across multiple categories.
- Demonstrates a practical solution to the common **"domain gap"** problem.
- **End-to-end pipeline** from data loading (using a custom `Dataset` class) to training and prediction.
- Optimized for **GPU training** using Google Colab.

---

## Tech Stack
- **Python 3.9+**
- **PyTorch:** The core deep learning framework.
- **Pandas:** For handling data annotations from the CSV file.
- **Pillow:** For image manipulation.
- **Kaggle API:** For programmatic dataset download.
- **Git & GitHub:** For version control and project hosting.
- **Google Colab:** For GPU-accelerated model training.

---

## Setup and Installation

To set up this project locally or in a new Colab notebook, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
    cd YOUR_REPO_NAME
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the dataset from Kaggle:**
    * First, ensure you have your `kaggle.json` API token.
    * Then, run the commands from the `Training.ipynb` notebook to download and unzip the dataset.
    ```bash
    # (Inside your Python/Colab environment)
    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json
    !kaggle datasets download -d paramaggarwal/fashion-product-images-small
    !unzip -q fashion-product-images-small.zip -d fashion-dataset
    ```
---

## How to Use

### 1. Training the Model
The model was trained using the `Training.ipynb` notebook. To replicate the training process:
- Open the notebook in Google Colab.
- Ensure the hardware accelerator is set to **GPU**.
- Follow the steps within the notebook, which cover data download, model definition, and the training loop.

### 2. Classifying a New Image
A prediction script can be created to classify new images using the saved `kaggle_fashion_model.pth` weights.

```bash
python predict.py --image_path /path/to/your/image.jpg
```

---

## Results
The fine-tuned model demonstrates strong performance on real-world product images, validating the transfer learning approach. The training process effectively adapted the pre-trained weights to the new domain, achieving high accuracy on the specific e-commerce categories.

*(Optional: Add a screenshot here of a correct prediction, e.g., your sneaker image being identified as "Shoes".)*

---

## Future Improvements
- Experiment with larger, more complex models (e.g., ResNet50, EfficientNet) for potentially higher accuracy.
- Deploy the trained model as a simple web API using Flask or FastAPI.
- Implement more advanced data augmentation techniques to further improve model robustness.
