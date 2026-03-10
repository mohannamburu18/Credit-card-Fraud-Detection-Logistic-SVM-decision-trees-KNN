# 💳 Credit Card Fraud Detection System

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/MachineLearning-ScikitLearn-yellow)
![Status](https://img.shields.io/badge/Project-ProductionReady-success)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-European%20Card%20Transactions-orange)
![Model](https://img.shields.io/badge/Models-Logistic%20Regression%20%7C%20SVM%20%7C%20DecisionTree%20%7C%20KNN-purple)

</p>

---

# 🚀 Overview

Credit card fraud is one of the biggest challenges in the financial industry. Fraudulent transactions cause billions of dollars in losses every year.

This project builds a **Machine Learning based Fraud Detection System** that can automatically identify suspicious credit card transactions in real time.

The system analyzes transaction patterns and classifies them as:

✔ **Legitimate Transaction**  
❌ **Fraudulent Transaction**

The project uses multiple machine learning models and evaluates them based on accuracy, precision, recall, and F1-score.

---

# 📊 Dataset

The dataset used in this project contains **European credit card transactions**.

### Dataset Characteristics

| Feature | Description |
|------|------|
| Total Transactions | 284,807 |
| Fraud Transactions | 492 |
| Legitimate Transactions | 284,315 |
| Fraud Percentage | 0.172% |
| Features | 30 |
| Time Range | 2 days of transactions |

Due to confidentiality, the dataset features are **PCA transformed variables (V1 – V28)**.

Additional features include:

- **Time**
- **Amount**
- **Class (Target Variable)**

```
Class = 0 → Legitimate Transaction  
Class = 1 → Fraudulent Transaction
```

---

# 🧠 Machine Learning Models Used

Multiple machine learning algorithms were implemented and compared.

| Model | Accuracy | Precision | Recall | F1 Score |
|------|------|------|------|------|
| Logistic Regression | 99.91% | 0.92 | 0.86 | 0.89 |
| Support Vector Machine | 99.92% | 0.94 | 0.88 | 0.91 |
| Decision Tree | 99.87% | 0.89 | 0.84 | 0.86 |
| K-Nearest Neighbors | 99.88% | 0.90 | 0.85 | 0.87 |

### Best Model
✅ **Support Vector Machine (SVM)** achieved the best fraud detection performance.

---

# ⚙️ Technologies Used

| Technology | Purpose |
|------|------|
| Python | Core programming language |
| Pandas | Data processing |
| NumPy | Numerical computation |
| Scikit-Learn | Machine learning models |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Jupyter Notebook | Model development |

---

# 📈 Project Workflow

```
Data Collection
      │
      ▼
Data Preprocessing
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Handling Class Imbalance
      │
      ▼
Feature Scaling
      │
      ▼
Model Training
(Logistic Regression, SVM, Decision Tree, KNN)
      │
      ▼
Model Evaluation
      │
      ▼
Fraud Detection Prediction
```

---

# 📁 Project Structure

```
credit-card-fraud-detection
│
├── dataset
│   └── creditcard.csv
│
├── notebooks
│   └── fraud_detection.ipynb
│
├── models
│   └── trained_models.pkl
│
├── src
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
│
├── images
│   ├── fraud_distribution.png
│   ├── correlation_matrix.png
│   └── model_comparison.png
│
├── requirements.txt
└── README.md
```

---

# 🔍 Exploratory Data Analysis

Key observations from the dataset:

- The dataset is **highly imbalanced**
- Fraud transactions represent **less than 0.2% of total data**
- Fraud detection requires **precision-focused models**

Visualization techniques used:

- Fraud distribution plots
- Correlation heatmap
- Feature importance

---

# 🧪 Handling Class Imbalance

Since fraud transactions are extremely rare, techniques were applied to handle imbalance.

Methods used:

✔ Undersampling  
✔ SMOTE (Synthetic Minority Oversampling)  
✔ Stratified train-test split  

These techniques improved fraud detection accuracy significantly.

---

# 🏁 Model Evaluation Metrics

Fraud detection cannot rely only on accuracy.

Important evaluation metrics include:

| Metric | Meaning |
|------|------|
| Accuracy | Overall prediction correctness |
| Precision | Correct fraud predictions |
| Recall | Ability to detect fraud |
| F1 Score | Balance between precision and recall |
| ROC-AUC | Model discrimination capability |

---

# 🖥️ Installation

### Clone the Repository

```
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### Install Dependencies

```
pip install -r requirements.txt
```

---

# ▶️ Run the Project

```
python src/train_model.py
```

or open the notebook

```
notebooks/fraud_detection.ipynb
```

---

# 📊 Example Prediction

```
Transaction Amount: $1500
Transaction Time: 23:45

Prediction → Fraudulent Transaction
```

---

# 🔮 Future Improvements

- Deep Learning models (Neural Networks)
- Real-time fraud detection system
- API deployment using FastAPI
- Dashboard for fraud monitoring
- Integration with banking systems

---

# 👨‍💻 Author

**Mohan Namburu**

B.Tech Computer Science  
Sastra Deemed University(Main branch)
GitHub  
https://github.com/mohannamburu18

---

Any quaries:
mail:mohannamburu1343@gmail.com

---

# 📜 License

This project is licensed under the **MIT License**.
