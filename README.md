# Real-Time Anomaly Detection using Autoencoder

This project implements an unsupervised machine learning model to detect anomalies in data using an autoencoder neural network. It simulates both normal and anomalous data and uses reconstruction error to identify whether a data point is normal or an attack.

---

## 📌 Overview

- Simulates synthetic data representing normal and anomalous behavior.
- Uses an autoencoder trained on normal data to detect anomalies via reconstruction error.
- Classifies data points as normal or anomalous based on an error threshold.
- Evaluates detection performance with accuracy, precision, and recall metrics.

---

## 🧠 Model Details

### ➤ Architecture

- **Input Layer** – Takes standardized features (10 dimensions).
- **Encoding Layer** – Dense layer with ReLU activation (dimension = 6).
- **Decoding Layer** – Dense layer with Sigmoid activation (same size as input).
- **Loss Function** – Mean Squared Error (MSE).
- **Optimizer** – Adam with learning rate = 0.001.

### ➤ Training

- Model is trained only on normal data.
- The reconstruction error on test data is used for anomaly detection.
- A threshold (92nd percentile of MSE) determines what is considered "anomalous".

---

## 🛠️ Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras

---

## 🧪 Dataset

### 🔹 Synthetic Data Generation

- **Normal samples**: 10,000 data points from a standard normal distribution.
- **Anomalous samples**: 1,000 data points from a normal distribution with shifted mean (loc = 5).

Each sample contains 10 numerical features.

---

## 📊 Evaluation Metrics

After threshold-based classification, the following metrics are computed:

- ✅ **Accuracy**
- 🎯 **Precision & Recall** for both:
  - **Normal Data**
  - **Attack (Anomalous) Data**

---

## 📁 File Structure

RealTimeAnomalyDetection/
├── RealTimeAnomalyDetection.py # Main Python script
├── README.md # Project description and guide
└── requirements.txt # Python dependencies

---

## 🚀 How to Run

1. **Clone the repository** or download the files.
2. **Install dependencies** using:

   ```bash
   pip install -r requirements.txt
3. **Run the Script** using:
   python RealTimeAnomalyDetection.py

---

📬 Contact
Created by Sudutta Bardhan (Feel free to reach out via LinkedIn for collaboration or questions!)
