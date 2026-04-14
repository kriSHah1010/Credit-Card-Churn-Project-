# 💳 Credit Card Churn Analysis Report

---

## 🎯 Objective

The goal of this project is to analyze customer churn behavior and identify key factors that influence whether a customer leaves or stays.

This analysis helps businesses proactively reduce churn and improve customer retention strategies.

---

## 📊 Dataset Overview

- Total customers: **10,000**
- Features: **21**
- Target variable: **churned (1 = churn, 0 = retained)**

### Target Distribution

- Retained customers: **8,462**
- Churned customers: **1,538**
- Churn rate: **15.38%**

📌 **Insight:**  
The dataset is moderately imbalanced, with churn being a relatively rare but important event.

---

## 🧹 Data Cleaning

- No duplicate records found  
- No missing values detected  
- Standardized column names for consistency  
- Created target variable `churned` from `Attrition_Flag`  
- Removed non-informative identifier columns  

📌 **Conclusion:**  
The dataset is clean and ready for analysis and modeling without heavy preprocessing.

---

## 🔍 Exploratory Data Analysis (EDA)

### 1. Churn Distribution

- Only ~15% of customers churn  
- Majority (~85%) are retained  

📌 **Insight:**  
Churn is not frequent but has high business impact → requires targeted intervention.

---

## 📈 Feature Insights

### 👥 Customer Demographics

- Churn varies across different age groups  
- Income categories show different churn tendencies  

📌 Older or financially stable segments may behave differently than newer customers.

---

### 💳 Financial Behavior

- Credit-related features (credit limit, utilization) influence churn  
- Customers with lower usage or engagement tend to churn more  

📌 Financial engagement is a key signal of customer retention.

---

### 🔄 Customer Activity

- Transaction count and activity levels strongly relate to churn  
- Low activity customers are more likely to leave  

📌 Behavioral features are **strong predictors** of churn.

---

## 🤖 Modeling

### Models Used

- Logistic Regression (baseline)
- Random Forest (advanced model)

### Evaluation Metrics

- ROC AUC
- Confusion Matrix
- Precision / Recall / F1-score

### Results

- Random Forest outperformed Logistic Regression  
- Better at capturing complex, non-linear relationships  

📌 **Conclusion:**  
Tree-based models provide stronger predictive performance for this dataset.

---

## 🧠 Key Insights

- 📉 Low engagement customers are at higher risk of churn  
- 📊 Behavioral features outperform demographic features  
- 🔍 Churn is driven by a mix of financial and activity-based factors  

---

## 💼 Business Recommendations

- 🎯 Target low-activity customers with retention campaigns  
- 📊 Monitor transaction and engagement metrics closely  
- 💡 Offer personalized incentives to high-risk segments  
- 🔁 Improve onboarding and engagement for new customers  

---

## 🚀 Conclusion

This project demonstrates how data analysis and machine learning can be used to:

- Understand customer behavior  
- Identify churn risk factors  
- Support business decision-making  

The combination of EDA, modeling, and visualization provides a complete end-to-end data science workflow.

---
