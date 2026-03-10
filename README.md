# Customer Churn Prediction using XGBoost and SHAP

## Project Overview
Customer churn prediction helps businesses identify customers who are likely to leave their service.  
This project builds an **end-to-end machine learning pipeline** to predict churn using the **IBM Telco Customer Churn dataset**.

The project includes:

- Feature Engineering
- Preprocessing Pipelines
- Hyperparameter Tuning
- Threshold Optimization
- Model Calibration
- Model Explainability using SHAP

The goal is to **identify high-risk customers and understand the factors driving churn**.

---

# Dataset

Dataset used:

IBM Telco Customer Churn Dataset

Source:

https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

Target variable:

```
Churn
```

Converted to binary:

```
Yes → 1
No → 0
```

---

# Machine Learning Pipeline

The model is built using a **Scikit-Learn Pipeline**.

Pipeline workflow:

```
Raw Dataset
    ↓
Feature Engineering
    ↓
ColumnTransformer
    ↓
Scaling + OneHotEncoding
    ↓
XGBoost Classifier
```

Pipeline components include:

- FunctionTransformer for custom feature engineering
- ColumnTransformer for preprocessing
- StandardScaler for numeric features
- OneHotEncoder for categorical variables
- XGBoostClassifier for prediction

---

# Feature Engineering

Several custom features were created to improve model performance.

### IsFiber
Binary indicator showing whether the customer uses **Fiber optic internet**.

### Total_Services
Number of subscribed services including:

```
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
```

### ChargesPerMonth

```
ChargesPerMonth = TotalCharges / (tenure + 0.1)
```

This normalizes customer spending by tenure.

---

# Model Training

Model used:

```
XGBoost Classifier
```

Hyperparameters optimized using:

```
GridSearchCV
```

Optimization metric:

```
Recall
```

Recall is prioritized to **capture as many churners as possible**.

Hyperparameters tuned:

- n_estimators
- max_depth
- learning_rate
- subsample
- colsample_bytree

---

# Model Performance

## Confusion Matrix

![Confusion Matrix](Confusion%20Matrix.png)

Results:

| Metric | Value |
|------|------|
| True Negatives | 750 |
| False Positives | 285 |
| False Negatives | 72 |
| True Positives | 302 |

The model captures a large portion of churners while maintaining manageable false positives.

---

# Precision Recall Curve

![Precision Recall Curve](Precision%20Recall%20Curve.png)

PR-AUC Score:

```
0.661
```

Precision-Recall curves are more informative than ROC curves for **imbalanced datasets**.

---

# Threshold Optimization

Instead of using the default **0.5 classification threshold**, different thresholds were tested to analyze:

- False Negatives
- False Positives
- True Positives
- True Negatives

This allows businesses to choose a **cost-effective retention threshold**.

---

# Model Calibration

Probability calibration ensures predicted probabilities reflect **true churn likelihood**.

## Before Calibration

![Calibration Curve](Calibration%20Curve.png)

---

## After Calibration

![Calibrated Model Curve](Calibrated%20Model%20Curve.png)

Calibration method used:

```
CalibratedClassifierCV(method="sigmoid")
```

Evaluation metric:

```
Brier Score
```

Lower Brier score indicates better probability calibration.

---

# Model Interpretability with SHAP

SHAP (SHapley Additive exPlanations) was used to understand **why the model predicts churn**.

SHAP provides:

- Global feature importance
- Feature impact direction
- Individual prediction explanations

---

# SHAP Feature Importance

![SHAP Importance](Shap%20Column%20Wise.png)

Top features influencing churn:

1. Month-to-month contracts
2. Customer tenure
3. Fiber optic internet
4. No online security
5. No tech support

---

# SHAP Summary Plot

![SHAP Summary](Shap%20Values.png)

Interpretation:

- Red points → high feature values
- Blue points → low feature values
- X-axis → feature impact on churn prediction

Key findings:

- Short tenure increases churn probability
- Fiber optic users churn more frequently
- Lack of security or support services increases churn risk

---

# Local Prediction Explanation

![Feature Contribution](Features%20Contributing.png)

This SHAP waterfall plot explains **why a specific customer was predicted to churn**, showing how individual features contribute to the prediction.

---

# Technologies Used

```
Python
Pandas
NumPy
Scikit-Learn
XGBoost
SHAP
Matplotlib
Seaborn
```

Machine learning techniques applied:

- Feature Engineering
- Pipeline Architecture
- Hyperparameter Tuning
- Model Calibration
- SHAP Explainability

---

# Repository Structure

```
Customer_Churn_Prediction
│
├── Churn_Project.ipynb
├── Confusion Matrix.png
├── Precision Recall Curve.png
├── Calibration Curve.png
├── Calibrated Model Curve.png
├── Shap Column Wise.png
├── Shap Values.png
├── Features Contributing.png
└── README.md
```

---

# Future Improvements

Possible extensions:

- Customer segmentation for targeted retention
- Survival analysis for churn timing
- Deployment using FastAPI
- Streamlit dashboard
- Real-time churn prediction API

---

# Author

Machine Learning project focused on **interpretable churn prediction using XGBoost and SHAP**.

If you found this project useful, consider ⭐ starring the repository.
