# 🔁 Customer Churn Prediction using ANN

A Python notebook for loading, analyzing, and building machine learning models that predict the likelihood of customers leaving a bank's credit card service, using customer demographics and account features from a real-world Kaggle dataset.

## 📌 Overview

This repository contains a Python notebook that predicts **credit card customer churn** using **machine learning** and **artificial neural networks (ANN)**.  
Utilizing the Kaggle `Churn_Modelling.csv` dataset, it walks through the full data science workflow: loading, exploring, preparing data, and laying the groundwork for churn prediction.

This project is ideal for:
- Data scientists learning churn modeling  
- Financial analysts building risk strategies  
- ML beginners practicing end-to-end projects  

## 📊 Dataset

- **Source**: [Kaggle - Credit Card Customer Churn Prediction](https://www.kaggle.com/datasets/shubhendra7/credit-card-customer-churn-prediction)
- **Filename**: `Churn_Modelling.csv`

### 🔍 Columns:
- `RowNumber`, `CustomerId`, `Surname` *(for indexing only)*
- **CreditScore**
- **Geography**
- **Gender**
- **Age**
- **Tenure**
- **Balance**
- **NumOfProducts**
- **HasCrCard**
- **IsActiveMember**
- **EstimatedSalary**
- **Exited** *(Target: 1 = Churned, 0 = Retained)*

## 🔁 Workflow

### 1. Data Loading
- Load the dataset using `pandas`

### 2. Data Exploration
- Understand distributions, correlations, and key trends

### 3. Preprocessing
- Encode categorical variables (Gender, Geography)
- Normalize numerical features
- Split dataset into train/test sets

### 4. Model Building
- Build a basic **Artificial Neural Network (ANN)** using `Keras` or `TensorFlow`
- Train on customer features to predict `Exited`

### 5. Evaluation
- Evaluate performance using accuracy, confusion matrix, etc.
- Visualize training history

## 🛠 Usage

### Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
Run the Notebook

Open the notebook in Jupyter, VS Code, or a Kaggle Notebook environment:
notebook8ad570467f.ipynb
Follow through the notebook cells to:

Load the data

Process features

Train and evaluate a basic ANN for churn prediction

🎯 Project Objective

Help banks proactively identify customers likely to churn

Provide insights to improve customer retention strategies

Serve as a baseline for deeper modeling: feature selection, advanced models, and interpretability tools (like SHAP, LIME)

📦 Requirements

Python 3.x

Libraries:
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow or keras
Jupyter Notebook or similar environment
Install dependencies:
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
✅ Results
Predicts which customers are likely to churn
Visualizations show how features impact churn
Can be extended with:
Hyperparameter tuning
Dropout, BatchNormalization
Advanced evaluation metrics (AUC, ROC, F1-score)
🙌 Credits
Dataset: Kaggle – Credit Card Customer Churn Prediction

Author: [Nitin Maurya]

📄 License

This project is open-source and available under the MIT License.

📝 Example GitHub Description

Short Description:
Artificial Neural Network-based churn prediction using customer credit card data. Helps banks identify at-risk customers. Built in Python with a clean ML pipeline in a Jupyter notebook.

Tags:
Churn, ANN, Neural Network, Classification, Banking, Kaggle, Python, Customer Retention, Deep Learning
