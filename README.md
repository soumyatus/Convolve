## 1. Introduction
This project involves two major predictive modeling tasks:
1. **Predicting Credit Card Defaults:** Using machine learning to predict the probability of credit card defaults based on transaction history.
2. **Predicting Optimal Communication Time:** Using historical data to determine the best time slots for sending marketing emails to maximize customer engagement.

## 2. Predicting Credit Card Defaults

### 2.1 Dataset Overview
- The dataset includes **96,806** records of credit card transactions.
- Each record has a flag (`bad_flag`):
  - `1`: Defaulted
  - `0`: Not defaulted
- The objective is to develop a **machine learning model** to predict the likelihood of default for new credit card accounts.

### 2.2 Data Preprocessing
#### 2.2.1 Handling Missing Data
- **<5% missing:** Mean imputation.
- **5%-30% missing:** K-Nearest Neighbors (KNN) imputation.
- **>30% missing:** Removed due to unreliability.

#### 2.2.2 Feature Engineering
- Removed features with **low correlation (<0.1)** to the target variable (`bad_flag`).
- Removed features with **high missing percentages (>30%)**.

#### 2.2.3 Feature Scaling
- **No manual scaling was performed**, as XGBoost handles unscaled features internally.

#### 2.2.4 Handling Class Imbalance
- The dataset was **highly imbalanced** (more non-defaults than defaults).
- Used the `scale_pos_weight` parameter in XGBoost:
  - Formula: **Negative Class Count / Positive Class Count**
  - This helped adjust class imbalance without introducing synthetic data.

### 2.3 Model Selection & Training
#### 2.3.1 Models Evaluated
- **Logistic Regression:** 68.0% accuracy.
- **Random Forest:** 73.0% accuracy.
- **Support Vector Machine (SVM):** 62.0% accuracy.
- **XGBoost (Final Model):** Outperformed all models and was selected for training.

#### 2.3.2 Handling Imbalanced Data
- **SMOTE** (Synthetic Minority Over-sampling Technique) was tested but resulted in **poor F1-score** due to noisy synthetic samples.
- Instead, used XGBoost's `scale_pos_weight` for class balancing.

### 2.4 Model Evaluation
#### 2.4.1 Metrics Used
- **Accuracy**
- **Precision, Recall, and F1-Score**
- **Confusion Matrix**
- Prioritized **Recall** over Precision to reduce **false negatives**.

### 2.5 Key Insights
- **Data Imbalance:** Adjusting class weights improved predictive performance.
- **Feature Selection:** Removing noisy features improved model efficiency.
- **Model Performance:** XGBoost demonstrated superior ability in identifying defaults.

---

## 3. Predicting Optimal Communication Time

### 3.1 Dataset Overview
- The dataset includes customer engagement data to predict the best time slots for sending marketing emails.
- Goal: **Optimize email communication** to maximize open rates.

### 3.2 Data Preprocessing
#### 3.2.1 Handling Missing Values
- **>50% missing:** Removed.
- **Numerical columns:** Imputed with median.
- **Categorical columns:** Imputed with mode.

#### 3.2.2 Feature Engineering
- Created **weekly and daily time slots (28 slots per week).**
- Generated `send_slot` and `open_slot` based on timestamps.

#### 3.2.3 Data Scaling & Aggregation
- **Scaling:** Applied MinMaxScaler for numerical columns.
- **Aggregation:** Grouped data by `customer_code` to summarize engagement history.
- **Merging Datasets:** Integrated multiple data sources on `customer_code` using an outer join.

### 3.3 Model Training
#### 3.3.1 Training the XGBoost Model
- **Objective:** Multi-class classification (`multi:softprob` for probability ranking).
- **Feature Engineering:**
  - One-hot encoding for categorical variables.
  - Normalization for numerical variables.
  - Derived features like `hour_of_send`, `day_of_week`, `is_weekend`.
- **Splitting Data:** Used stratified `train_test_split` to balance class distribution.

### 3.4 Model Evaluation
- **Accuracy:** 0.077 (reflecting the probability ranking task).
- **Feature Importance:** Used XGBoost's built-in feature importance scores to interpret predictions.
- **Generating Slot Rankings:**
  - Ranked slots based on predicted probabilities for each customer.
  - Results stored in CSV format for deployment.

### 3.5 Neural Network Alternative
- Implemented a **three-layer neural network** with a custom loss function to optimize slot rankings.
- **Architecture:**
  - **Input Layer:** 128 neurons, ReLU activation.
  - **Hidden Layer:** 64 neurons, ReLU activation.
  - **Output Layer:** Softmax activation for 28 slots.
- **Custom Loss Function:** Prioritized correct slot predictions while penalizing incorrect ones.
- **Results:**
  - The neural network struggled compared to XGBoost.
  - Reasons: Complexity, sparse engagement data, and lack of hyperparameter tuning.

### 3.6 Insights & Observations
- **Peak Engagement Hours:** 1 PM & 3 PM.
- **Highest Engagement Day:** Saturday.
- **Product Preferences:** Savings & credit card activation had the highest email engagement.

---

## 4. Conclusion
### 4.1 Credit Card Default Prediction
- Successfully built an XGBoost model to predict credit card defaults.
- Addressed missing data, feature selection, and class imbalance.
- Prioritized **recall** to minimize false negatives, reducing default risk for banks.

### 4.2 Optimal Communication Time Prediction
- Built a model to rank **28 time slots** for email marketing effectiveness.
- XGBoost outperformed a neural network approach.
- Used **feature engineering, aggregation, and ranking probabilities** to optimize predictions.

---

## 5. Future Work
- **Credit Default Prediction:** Further hyperparameter tuning and explainability improvements.
- **Email Optimization:** Test additional models (e.g., LSTMs for sequential patterns).
- **A/B Testing:** Deploy predictions in a real-world email campaign and refine models based on engagement feedback.

---

### 📌 **Authors:** Team UMIAM
### 📅 **Last Updated:** 2025
"""

# Save the README as a Markdown file
with open("README.md", "w") as file:
    file.write(readme_content)

print("README.md file has been created successfully!")
