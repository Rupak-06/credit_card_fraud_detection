# 💳 Credit Card Fraud Detection System  
**Python · Streamlit · TensorFlow (Keras) · Scikit-learn · Pandas · NumPy**

A **real-time credit card fraud detection system** using a **Deep Autoencoder-based anomaly detection model**.  
Built with **Python**, **Streamlit**, and **Keras**, this project demonstrates how unsupervised deep learning can identify fraudulent transactions with high recall accuracy in real-time applications.

---

## 📜 Table of Contents
- [About The Project](#about-the-project)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation & Setup](#installation--setup)
  - [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Model Overview](#model-overview)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 📖 About The Project
This project presents a **Deep Autoencoder-Based Anomaly Detection System** designed to identify fraudulent credit card transactions in real time.  

Instead of traditional supervised classification, the system learns normal transaction patterns from legitimate data and flags anomalies with high reconstruction errors as potential frauds.  
The project was deployed as a **Streamlit web app**, demonstrating its real-time usability for financial institutions.

### 🧠 Academic Details
- **Course:** Deep Learning Project – Final Review  
- **Institution:** School of Computer Science and Engineering, VIT-AP University  
- **Presented by:** Rupak Vivek Sai Oleti (23BCE8279)  
- **Guide:** Prof. Allapati Rajya Lakshmi  

---

## ✨ Key Features
- 🤖 **Unsupervised Deep Autoencoder:** Learns transaction patterns using only legitimate data.  
- ⚙️ **Feature Engineering:** Implements cyclical time encoding for temporal pattern recognition.  
- 🎚️ **Threshold Optimization:** Dynamically determines reconstruction error cut-off for high recall.  
- 📊 **Evaluation Metrics:** Precision, Recall, F1-Score, and Confusion Matrix.  
- 🌐 **Real-Time Web App:** Deployed using Streamlit for instant fraud detection.  
- 🧩 **Model Persistence:** Trained model (`fraud_autoencoder.h5`) and scaler (`scaler.pkl`) stored for reuse.  
- 📈 **Visualization:** Includes training loss plot and confusion matrix for interpretability.  

---

## 🚀 Getting Started
Follow the steps below to set up and run the project locally.

### 1️⃣ Prerequisites
Make sure you have the following installed:
- Python 3.8 or higher  
- pip (Python package installer)  
- git  

---

### 2️⃣ Installation & Setup

#### Clone the Repository
git clone https://github.com/your-username/credit_card_fraud_detection.git
cd credit_card_fraud_detection

### Create and Activate a Virtual Environment
# For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```


# For macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

# Install Dependencies
```bash
pip install -r requirements.txt
```

Download Dataset
Download the Credit Card Fraud Detection Dataset from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the downloaded creditcard.csv inside the /data folder.

▶️ How to Run
1️⃣ Train the Model
```bash
python src/train.py
```

This trains the deep autoencoder and saves:
- Trained model → models/fraud_autoencoder.h5
- Scaler object → models/scaler.pkl
- Training loss plot → models/training_loss_plot.png

2️⃣ Evaluate Model
```bash
python src/evaluate.py
```

Generates performance metrics and saves models/confusion_matrix.png.

3️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

4️⃣ Open the App
Visit your browser at:
```bash
http://localhost:8501
```

You can enter transaction data manually or load a sample to see live fraud detection results.

## 📁 Project Structure
```bash
CREDIT_CARD_FRAUD_DETECTION/
├─ data/
│  └─ creditcard.csv                 # Dataset
├─ models/
│  ├─ fraud_autoencoder.h5           # Trained Autoencoder
│  ├─ scaler.pkl                     # StandardScaler object
│  ├─ confusion_matrix.png           # Evaluation output
│  └─ training_loss_plot.png         # Loss visualization
├─ src/
│  ├─ data_preprocessing.py          # Data cleaning & feature engineering
│  ├─ evaluate_isolation_forest.py   # Comparison model (Isolation Forest)
│  ├─ evaluate.py                    # Evaluation script
│  ├─ model.py                       # Autoencoder architecture
│  └─ train.py                       # Training script
├─ app.py                            # Streamlit application
├─ main.py                           # Entry script
├─ requirements.txt                  # Dependencies
└─ README.md                         # Documentation
```

## 🧩 Model Overview
Architecture: Symmetrical Deep Autoencoder
- Encoder: [31 → 16 → 8 → 4]
- Decoder: [4 → 8 → 16 → 31]

Activation: ReLU

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam

Training Epochs: 50

Batch Size: 32

Threshold (MSE): 1.304159 for 90% recall

## 📊 Results
Class	        Precision	    Recall	    F1-Score	Support
Legitimate(0)	   1.00	         0.99	      1.00	     56863
Fraud(1)	       0.58	         0.90	      0.70	      98

✅ Recall (90%) ensures minimal missed frauds.
⚠️ Precision (58%) is acceptable since false positives are less critical than false negatives.
🏁 Overall Accuracy: 99%.

Visual Outputs
- training_loss_plot.png → Autoencoder loss curve over epochs
- confusion_matrix.png → Evaluation on test set

# 🤝 Contributing
Contributions are welcome and appreciated!

### Steps to contribute:
1. Fork the repository

2. Create your branch
```bash
git checkout -b feature/AmazingFeature
```

3. Commit your changes
```bash
git commit -m "Add some AmazingFeature"
```

4. Push to your branch
```bash
git push origin feature/AmazingFeature
```

5. Open a Pull Request