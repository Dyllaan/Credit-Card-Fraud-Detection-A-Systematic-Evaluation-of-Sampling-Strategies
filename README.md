# Credit Card Fraud Detection: A Systematic Evaluation of Sampling Strategies

A rigorous machine learning pipeline for credit card fraud detection, achieving **0.9020 AUPRC** through systematic testing of sampling strategies with gradient boosting models.

## Overview

This project systematically evaluates multiple oversampling techniques (SMOTE, ADASYN, Borderline SMOTE, SMOTE-ENN) combined with XGBoost and LightGBM for fraud detection on a highly imbalanced dataset (0.17% fraud, 577:1 ratio).

**Key Finding:** SMOTE with 2% oversampling ratio achieves the best performance (0.9020 AUPRC), though the improvement over baseline XGBoost (+0.70%) demonstrates that modern gradient boosting frameworks handle class imbalance effectively without synthetic sampling.

## Results Summary

| Model | AUPRC | Precision | Recall |
|-------|-------|-----------|--------|
| **SMOTE 0.02 + XGBoost** | **0.9020** | 0.876 | 0.867 |
| Borderline SMOTE 0.005 + XGBoost | 0.9010 | 0.869 | 0.878 |
| No Sampling + XGBoost | 0.8957 | 0.878 | 0.878 |
| SMOTE 0.03 + XGBoost | 0.9008 | 0.896 | 0.878 |
| No Sampling + LightGBM | 0.8898 | 0.813 | 0.888 |

**Key Insights:**
- SMOTE with 2-3% ratio optimal for XGBoost
- Borderline SMOTE competitive with standard SMOTE
- ADASYN underperforms due to PCA-transformed feature space
- SMOTE-ENN significantly reduces precision
- XGBoost consistently outperforms LightGBM

## Dataset

**Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Specifications:**
- 284,807 transactions
- 30 features (28 PCA-transformed + Time + Amount)
- 492 frauds (0.17% of all transactions)
- Highly imbalanced: 577:1 ratio

### Prerequisites
- Python 3.1.2+ (Others may work but are untested)
- Dependencies listed in requirements.txt
- `creditcard.csv` from Kaggle

## Usage

Execute cells sequentially. The notebook includes:
1. Data loading and train/test split
2. Feature engineering
3. Baseline model evaluation
4. SMOTE experimentation
5. ADASYN experimentation
6. Borderline SMOTE experimentation
7. SMOTE-ENN experimentation
8. Results visualisation

### Expected Runtime

- **Full notebook execution:** ~6-8 hours on RTX 4060
- **Bayesian optimisation:** 50 trials × 5 folds per model
- **Caching:** Subsequent runs with same parameters are near-instant

## Methodology

### Data Splitting
- 80/20 train-test split with stratification
- Validation set (20% of training) for feature engineering and hyperparameter tuning
- Held-out test set for final evaluation

### Feature Engineering
- Statistical aggregations (mean, std, min, max)
- Time-based transaction patterns
- Amount-based features (normalised and binned)
- Interaction features from high-importance variables

### Sampling Strategies

**SMOTE (Synthetic Minority Over-sampling Technique)**
- Ratios tested: 0.5%, 1%, 2%, 3%
- Generates synthetic fraud cases via interpolation

**ADASYN (Adaptive Synthetic Sampling)**
- Ratios tested: 0.5%, 1%, 2%, 3%
- Focuses on harder-to-learn regions

**Borderline SMOTE**
- Ratios tested: 0.5%, 1%, 2%
- Only generates samples near decision boundary

**SMOTE-ENN (SMOTE + Edited Nearest Neighbours)**
- Ratios tested: 0.5%, 1%, 2%
- Combines oversampling with cleaning

### Models

**XGBoost**
- Gradient boosting with `scale_pos_weight` for imbalance
- Bayesian optimisation for hyperparameters through Optuna

**LightGBM**
- Alternative gradient boosting framework
- Comparison baseline for XGBoost

### 5. Evaluation Metrics

**Primary Metric: AUPRC (Area Under Precision-Recall Curve)**
- More informative than ROC-AUC for imbalanced datasets
- Focuses on minority class performance
- Captures precision-recall trade-off

**Secondary Metrics:**
- Precision: Minimise false positives (customer satisfaction)
- Recall: Maximise fraud detection (loss prevention)

## Key Results and Analysis

### Why SMOTE Works (Marginally)
- Creates denser representation of fraud cases in feature space
- Helps XGBoost refine decision boundaries
- Optimal ratio (2-3%) balances improvement vs overfitting

### Why ADASYN Underperforms
1. PCA-transformed features make distance metrics unreliable
2. Adaptive density-based sampling may amplify noisy borderline cases
3. Dataset already well-separated (baseline performs well)

### Why SMOTE-ENN Fails
- Removes borderline samples, including legitimate fraud cases
- Particularly harmful for 577:1 imbalance ratio
- Precision drops from 0.876 to 0.67-0.79

### Production Recommendation For This Dataset
Deploy **SMOTE 0.02 + XGBoost** for:
- Best overall AUPRC (0.9020)
- Balanced precision (0.876) and recall (0.867)
- Robust and stable performance

## Project Context

This work builds upon my previous undergraduate research in fraud detection using multi-layer perceptrons in MATLAB (https://github.com/Dyllaan/MLP-ANN). The current project demonstrates substantially improved methodology:
- Systematic hyperparameter optimisation (Bayesian search)
- Rigorous train/validation/test splitting
- Comprehensive sampling strategy evaluation
- Modern gradient boosting frameworks

## Future Work

- Neural network architectures like TabNet (preliminary experiments showed higher computational cost with inferior performance on this tabular data)
- Ensemble methods combining multiple sampling strategies
- Cost-sensitive learning approaches
- Real-time deployment considerations
- Interpretability analysis (SHAP values)
