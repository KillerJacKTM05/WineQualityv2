# Wine Quality Analysis

This project analyzes the red wine quality dataset to understand the relationships between physicochemical features and wine quality, perform feature selection, and build regression and classification models to predict quality scores or bins.

## Dataset Overview

- **Source**: Red wine quality dataset
- **Target**: `quality` score (ranging from 3 to 8)

### Quality Bin Distribution

| Quality Bin | Count |
|-------------|-------|
| Low         | 63    |
| Medium      | 1319  |
| High        | 217   |

### The dataset is not well-distributed.
Since 3 to 8 is a range for this dataset, a highly skewed distribution exists around quality 5-6. This situation challenges the classification and regression models.

## Feature Importance (ANOVA)

ANOVA F-test was used for feature selection. Below are the top features ranked by F-score:

| Feature                | F-Score     | p-Value         |
|------------------------|-------------|------------------|
| Alcohol                | 158.78      | 1.29e-63         |
| Volatile Acidity       | 102.40      | 1.44e-42         |
| Citric Acid            | 44.84       | 1.13e-19         |
| Sulphates              | 36.52       | 3.10e-16         |
| Total Sulfur Dioxide   | 21.96       | 3.92e-10         |
| Density                | 18.77       | 8.80e-09         |
| Fixed Acidity          | 13.17       | 2.13e-06         |
| pH                     | 9.41        | 8.65e-05         |
| Free Sulfur Dioxide    | 9.30        | 9.64e-05         |
| Chlorides              | 8.26        | 2.70e-04         |
| Residual Sugar         | 2.32        | 9.83e-02         |

---

## 🧪 Feature Pair Evaluation (Distance Consistency)

Feature pairs with high distance consistency (useful for class separation):

| Feature 1           | Feature 2             | Distance Consistency |
|---------------------|-----------------------|----------------------|
| Alcohol             | Volatile Acidity      | 0.8355               |
| Citric Acid         | Sulphates             | 0.8343               |
| Alcohol             | Citric Acid           | 0.8330               |
| Alcohol             | Sulphates             | 0.8324               |
| Alcohol             | Total Sulfur Dioxide  | 0.8268               |
| Sulphates           | Total Sulfur Dioxide  | 0.8199               |
| Citric Acid         | Total Sulfur Dioxide  | 0.8193               |
| Volatile Acidity    | Total Sulfur Dioxide  | 0.8186               |
| Volatile Acidity    | Sulphates             | 0.8124               |
| Volatile Acidity    | Citric Acid           | 0.8055               |

---

##  Regression Results

### Linear Regression

- **Using ANOVA-selected Features**  
  - MSE: 0.415  
  - R² Score: 0.357

- **Using All Features**  
  - MSE: 0.406  
  - R² Score: 0.370

### Random Forest Regression

- **Using ANOVA-selected Features**  
  - MSE: 0.338  
  - R² Score: 0.476

- **Using All Features**  
  - MSE: 0.322  
  - R² Score: 0.501

---

## Classification Results

**Model**: Random Forest Classifier (with 3 Quality Bins: low, medium, high)

| Metric      | High  | Low   | Medium |
|-------------|-------|-------|--------|
| Precision   | 0.55  | 0.20  | 0.91   |
| Recall      | 0.72  | 0.15  | 0.88   |
| F1-Score    | 0.63  | 0.17  | 0.89   |
| Support     | 43    | 13    | 264    |

**Overall Metrics**:
- Accuracy: **0.82**
- Macro Avg F1: **0.56**
- Weighted Avg F1: **0.83**

### Confusion Matrix
| Metric      | High  | Low   | Medium |
|-------------|-------|-------|--------|
|Actual High  | 31  | 0  | 12   |
|Actual Low   | 0  | 2  | 11   |
|Actual Medium | 25    | 8   | 231    |
