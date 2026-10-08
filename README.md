# Week 4 – Machine Learning Model Development and Evaluation

## Logistic Regression Classification Using the Breast Cancer Wisconsin Dataset

This project demonstrates a complete **machine learning model development and evaluation workflow in Python**. It was completed as part of a Week 4 machine learning task.

The project covers:

- Data loading and inspection
- Data quality checking
- Train-test splitting
- Feature scaling
- Logistic Regression model development
- Model prediction
- Classification metrics
- Confusion matrix
- ROC curve and ROC-AUC
- Error analysis
- Overfitting and underfitting discussion
- Possible model improvements

---

## 1. Project Objective

The objective of this project is to develop and evaluate a basic machine learning classification model using publicly available data.

The workflow follows these major stages:

**Data → Preprocessing → Train/Test Split → Scaling → Model Training → Prediction → Evaluation → Visualization → Analysis**

---

## 2. Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic) dataset** provided directly through Scikit-learn.

### Dataset information

| Property | Value |
|---|---:|
| Observations | 569 |
| Features | 30 |
| Target classes | 2 |
| Class 0 | Malignant |
| Class 1 | Benign |
| Training data | 80% |
| Testing data | 20% |

The dataset contains numerical measurements calculated from digitized images of breast mass cell nuclei.

No external CSV download is required because the dataset is available directly from Scikit-learn.

---

## 3. Technologies Used

- Python
- NumPy
- pandas
- Matplotlib
- Scikit-learn
- Google Colab / Jupyter Notebook

---

## 4. Machine Learning Algorithm

### Logistic Regression

Logistic Regression was selected because this is a **binary classification problem**.

Advantages of Logistic Regression:

- Simple and efficient
- Easy to understand
- Suitable for binary classification
- Produces probability estimates
- Provides a strong baseline for comparison with more complex models
- Works well when the relationship between predictors and the log-odds of the target is approximately linear

---

## 5. Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset using Scikit-learn.
2. Created pandas DataFrames for the features and target.
3. Checked for missing values.
4. Checked for duplicate rows.
5. Split the dataset into training and testing sets.
6. Used stratified sampling to preserve class proportions.
7. Standardized the numerical features using `StandardScaler`.

### Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Feature Scaling

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 6. Model Training

The Logistic Regression model was trained using the scaled training data.

```python
model = LogisticRegression(
    max_iter=5000,
    random_state=42
)

model.fit(X_train_scaled, y_train)
```

Predictions were generated using:

```python
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
```

---

## 7. Model Performance

The model achieved the following results on the held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | **98.25%** |
| Precision | **98.61%** |
| Recall | **98.61%** |
| F1-Score | **98.61%** |
| ROC-AUC | **99.54%** |

These results indicate that the model performed very well on the selected test set.

---

## 8. Confusion Matrix

The confusion matrix shows the number of correct and incorrect predictions for each class.

![Confusion Matrix](assets/confusion_matrix.png)

The model correctly classified:

- **41 malignant observations**
- **71 benign observations**

There were only **2 misclassified observations** in the test set.

---

## 9. ROC Curve

The ROC curve evaluates the model's ability to distinguish between the two classes at different classification thresholds.

![ROC Curve](assets/roc_curve.png)

The model achieved an **ROC-AUC of 0.9954**, indicating excellent class-separation performance on the test set.

---

## 10. Performance Metrics Comparison

![Performance Metrics](assets/performance_metrics.png)

The chart compares Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

All five metrics are above 0.98, showing consistent performance across the selected evaluation measures.

---

## 11. Error Analysis

Although the model achieved high performance, classification errors are still possible.

Potential sources of error include:

- Overlap between the distributions of the two classes
- Measurement variability
- Limited sample size
- Correlation or redundancy among features
- The linear assumptions of Logistic Regression
- Differences between this dataset and future unseen data

In a medical classification context, false negatives can be particularly important because a malignant case classified as benign could potentially delay further investigation.

---

## 12. Overfitting and Underfitting

### Overfitting

Overfitting occurs when a model learns the training data too closely and performs worse on unseen data.

A possible sign would be:

**Very high training performance + significantly lower test performance**

### Underfitting

Underfitting occurs when a model is too simple to capture important patterns in the data.

Logistic Regression is a relatively simple model. If the underlying relationship is strongly nonlinear, algorithms such as Random Forest, Support Vector Machine, or Gradient Boosting may perform better.

A stronger evaluation would use cross-validation instead of relying on only one train-test split.

---

## 13. Possible Improvements

Future versions of the project could include:

1. Compare Logistic Regression with Decision Tree.
2. Compare with Random Forest.
3. Compare with Support Vector Machine.
4. Compare with XGBoost or another boosting model.
5. Use k-fold cross-validation.
6. Perform hyperparameter tuning using `GridSearchCV`.
7. Perform feature selection.
8. Try dimensionality reduction such as PCA.
9. Tune the classification threshold.
10. Evaluate the final model on an independent external dataset.

---

## 14. Project Structure

```text
Week_4_ML_Logistic_Regression_GitHub/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── week4_logistic_regression.py
├── model_evaluation_results.csv
├── Week_4_Machine_Learning_Model_Development_and_Evaluation.docx
│
└── assets/
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── performance_metrics.png
```

---

## 15. How to Run the Project

### Step 1 – Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2 – Open the project

```bash
cd Week_4_ML_Logistic_Regression_GitHub
```

### Step 3 – Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 – Run the Python script

```bash
python week4_logistic_regression.py
```

The script will load the dataset, preprocess the data, train the model, display evaluation metrics, and generate the required visualizations.

### Google Colab

The Python file can also be copied into Google Colab and executed cell by cell.

No external dataset upload is required.

---

## 16. Key Learning Outcomes

Through this project, the following machine learning concepts were demonstrated:

- Supervised learning
- Binary classification
- Train-test splitting
- Stratified sampling
- Feature scaling
- Logistic Regression
- Model prediction
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC curve
- ROC-AUC
- Error analysis
- Overfitting
- Underfitting
- Model improvement strategies

---

## 17. Conclusion

This project successfully demonstrates the complete development and evaluation of a basic machine learning classification model.

Logistic Regression achieved **98.25% accuracy** and **99.54% ROC-AUC** on the held-out test set. The confusion matrix and ROC curve provide visual evidence of the model's strong classification performance.

However, a high score on one dataset does not guarantee equivalent performance in a real-world environment. Further experimentation with cross-validation, hyperparameter tuning, feature selection, and alternative algorithms would provide a more robust comparison.

---

## Author

**Apoorv Saxena**

BCA – Manipal University Jaipur

Project: Week 4 – Machine Learning Model Development and Evaluation
