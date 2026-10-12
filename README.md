# 🛡️ Intelligent Network Intrusion Detection System (NIDS)

This repository contains the implementation of my graduation project (PFE). It proposes a robust Network Intrusion Detection System (NIDS) utilizing a hybrid deep learning architecture to detect cyber threats in modern IoT network environments.

## 📌 Introduction
With the rapid digitalization of critical infrastructure, safeguarding against sophisticated cyber threats has become paramount. This project presents a hybrid deep learning approach for an Intrusion Detection System focusing on the modern **CICIoT2023** dataset. 

## 🧠 Methodology & Architecture
The proposed IDS employs a structured pipeline to accurately differentiate between malicious and benign network traffic:

* **Data Balancing (SMOTE):** Synthetic Minority Over-sampling Technique was implemented to handle class imbalances within the network traffic data, significantly improving the detection of minority attack classes.
* **CNN (Feature Extraction):** Convolutional layers are employed for spatial feature extraction to identify structural patterns in network flow data.
* **LSTM (Temporal Analysis):** A Long Short-Term Memory layer captures temporal dependencies and sequential patterns in the traffic over time, crucial for detecting complex cyber attacks.

## 📊 Results & Performance
The hybrid CNN-LSTM model demonstrated superior performance. Below is a representation of the evaluation metrics (Update these numbers with your actual PFE results):

| Model Architecture | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| LSTM (Baseline)    | 0.88     | 0.87      | 0.85   | 0.86     |
| CNN (Baseline)     | 0.91     | 0.90      | 0.89   | 0.89     |
| **Hybrid CNN-LSTM**| **0.96** | **0.95**  | **0.94**| **0.95** |

## 🛠️ Technologies
* **Frameworks:** PyTorch, Scikit-Learn
* **Data Processing:** Pandas, NumPy, Imbalanced-learn (SMOTE)

## 🚀 Installation & Usage
To test the model or run it on your local machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/MFerdaouse/Hybrid-CNN-LSTM-Intrusion-Detection.git](https://github.com/MFerdaouse/Hybrid-CNN-LSTM-Intrusion-Detection.git)