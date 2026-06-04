# 🍎 Cross-Fruit Ripeness Detection System

A machine learning application that predicts whether a fruit is **unripe**, **ripe**, or **overripe** from an uploaded image.

The project combines transfer learning, ensemble modelling, a FastAPI backend, and a React frontend to create an end-to-end ripeness prediction system.

---

## Project Overview

This project was developed to explore whether deep learning models can learn general ripeness characteristics across different fruit types rather than simply memorizing the appearance of specific fruits.

The system allows users to upload an image through a web interface and receive a ripeness prediction along with confidence scores for each class.

Classes:

* Unripe
* Ripe
* Overripe

---

## Features

* Image upload through a React frontend
* Real-time ripeness prediction
* FastAPI REST API backend
* Ensemble prediction using multiple CNN architectures
* Confidence score visualization
* Cross-fruit evaluation on unseen fruit categories

---

## Model Architecture

The prediction system uses a weighted ensemble of:

* MobileNetV2
* EfficientNetB0
* ResNet50

Each model was trained using transfer learning and fine-tuning techniques.

The final prediction is generated using weighted averaging of the three model outputs.

---

## Dataset

Training fruits:

* Apple
* Banana
* Orange

Cross-generalization evaluation:

* Mango
* Tomato

Images were cleaned, resized, and organized into three ripeness classes:

* Unripe
* Ripe
* Overripe

---

## Project Structure

```text
RIPENESS/
│
├── code/
│   ├── backend/
│   │   └── main.py
│   │
│   ├── frontend/
│   │   └── src/
│   │
│   └── models/
│       ├── mobilenet_model.h5
│       ├── efficientnet_model.h5
│       ├── resnet_model.h5
│       └── ensemble_weights.json
│
├── requirements.txt
└── README.md
```

---

## Tech Stack

### Machine Learning

* TensorFlow
* Keras
* NumPy
* Scikit-learn

### Backend

* FastAPI
* Uvicorn

### Frontend

* React
* Vite

### Development Environment

* Python
* Kaggle
* VS Code

---

## Notes

The trained model files are not included in this repository because large model files exceed GitHub's file size limits.

Required model files:

* mobilenet_model.h5
* efficientnet_model.h5
* resnet_model.h5
* ensemble_weights.json

These files should be placed inside:

```text
code/models/
```

before running the application.

---

## Future Improvements

* Support additional fruit categories
* Improve cross-fruit generalization performance
* Optimize model size for deployment
* Add mobile-friendly UI enhancements
* Deploy using Render and Vercel

---

## Author

Ritika

Student Project – Machine Learning, Computer Vision, and Full-Stack Deployment
