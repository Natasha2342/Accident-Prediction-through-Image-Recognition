# Accident Prediction Using Machine Learning

## Overview

Accident prediction is a critical component of current safety management systems, aiming to reduce the incidence and severity of accidents in various areas, including transportation and industrial environments. This project leverages Convolutional Neural Networks (CNN), K-Nearest Neighbors (KNN), and Random Forest Trees to forecast accidents using image recognition algorithms. Our primary goal is to create predictive models, evaluate their performance, and determine the best method for accident prediction.
## Project Description

This study focuses on predicting accidents by employing machine learning techniques combined with image recognition. We have implemented and compared three models: CNN, KNN, and Random Forest Trees. The main objective is to determine the most effective method for predicting accidents by analyzing a diverse array of accident-related images.

### Key Highlights

- **Data Collection:** We amassed a diverse dataset of accident-related images, including traffic incidents and industrial disasters. The dataset is sourced from publicly accessible repositories such as Kaggle.

- **Data Preprocessing:** Images were preprocessed through resizing, normalization, and transformation into feature vectors suitable for machine learning algorithms.

- **Model Development:** We trained three separate models (CNN, KNN, and Random Forest Trees) using the preprocessed image data and relevant non-image features, such as location and weather conditions.

- **Model Evaluation:** The performance of each model was evaluated using metrics such as accuracy, precision, recall, and F1-score.

- **Comparative Analysis:** We conducted a comparative analysis to determine which machine learning approach is the most effective for accident prediction in different scenarios.

## Technologies Used

- **Machine Learning Algorithms:** Convolutional Neural Networks (CNN), K-Nearest Neighbors (KNN), Random Forest Trees
- **Image Recognition Techniques**
- **Programming Languages:** Python
- **Frameworks and Libraries:** TensorFlow, Keras, scikit-learn, OpenCV

## Results and Findings

Our findings indicate that each algorithm has distinct advantages and drawbacks in accident prediction tasks:

- **CNN:** Excels at capturing complex spatial patterns within images, achieving 93% accuracy.
- **KNN:** Shows promise in handling non-image data, achieving 91% accuracy.
- **Random Forest Trees:** Achieved the highest accuracy of 96%, effectively handling both numerical and categorical features.
