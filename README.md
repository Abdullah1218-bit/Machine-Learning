📁 Machine Learning Repository Overview
This repository contains comprehensive hands-on work across the entire machine learning pipeline — from raw data preprocessing to model selection and performance improvement. Each folder covers specific topics, algorithms, and techniques used in practical data science workflows.

🔧 1. Preprocessing & Data Handling
📂 Data Scaling
Standardizing feature values is essential to ensure fair model performance.

StandardScaler
Scales data to have a mean = 0 and standard deviation = 1. Works best with normally distributed features.

MinMaxScaler
Scales features to a given range, usually [0, 1]. Useful when you want to preserve the shape of the distribution.

RobustScaler
Removes the median and scales the data using the interquartile range (IQR). Robust to outliers.

MaxAbsScaler
Scales data by the maximum absolute value, maintaining sparsity in datasets.

📂 Data Transformation
Transforms the distribution of features to reduce skewness and improve linearity.

Log Transformation
Used to handle right-skewed data by compressing large values.

Square Root Transformation
Reduces skewness of moderate skewed distributions.

Box-Cox Transformation
Applies a power transformation to stabilize variance and make data more normal-like. Only works on positive values.

Yeo-Johnson Transformation
Similar to Box-Cox, but supports both positive and negative values.

📂 Imputers
Used to fill in missing values in datasets.

SimpleImputer
Strategies include:

mean – replaces missing values with column mean

median – replaces with column median

most_frequent – for categorical data

constant – fills with a fixed value

KNNImputer
Fills missing values using K-Nearest Neighbors, considering similarity between rows.

IterativeImputer
A multivariate imputer that models each feature with missing values as a function of other features.

📂 Outliers
Detecting and treating outliers is important for model robustness.

Z-Score Method
Marks values with z-score > 3 or < -3 as outliers.

IQR Method
Identifies outliers as values below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR.

Boxplots & Distribution Plots
Used to visualize and confirm outlier presence.

📂 Feature Encoding
Transforms categorical data into numerical format for model compatibility.

Label Encoding
Converts each category to an integer label (e.g., red=0, green=1, blue=2). Suitable for ordinal data.

OneHotEncoding
Creates binary columns for each category. Best for nominal (non-ordered) categories.

OrdinalEncoder
Manually assigns numeric order to categories (e.g., Low=1, Medium=2, High=3).

Target Encoding (not yet mentioned but useful)
Encodes categories based on the mean of the target variable.

⚙️ 2. Modeling & Pipelines
📂 Pipeline
Automates the full ML process: preprocessing → modeling → prediction.

Used Pipeline and ColumnTransformer from Scikit-learn to streamline workflows.

Enables reproducibility and clean code.

📂 Polynomial Regression
Extends linear models by generating polynomial features 
to capture non-linear relationships.

📂 Regularization
Used to prevent overfitting in linear models.

Ridge Regression (L2)
Adds penalty equal to the square of coefficients. Shrinks coefficients but doesn’t eliminate them.

Lasso Regression (L1)
Adds penalty equal to the absolute value of coefficients. Can shrink some coefficients to zero (feature selection).

ElasticNet
Combines L1 and L2 penalties for a balanced regularization.


🤖 Supervised Machine Learning Algorithms
This section showcases a wide range of machine learning models implemented and evaluated using clean pipelines and preprocessing. Each algorithm is trained, tuned, and validated using best practices such as scaling, encoding, cross-validation, and hyperparameter tuning.

🔍 Linear Models
Linear Regression
Simple baseline for predicting continuous numeric values.

Ridge Regression (L2 Regularization)
Penalizes large coefficients to reduce overfitting.

Lasso Regression (L1 Regularization)
Encourages sparsity by eliminating irrelevant features.

ElasticNet
Combination of L1 and L2 regularization.

🌲 Tree-Based Models
📌 Basic Models
Decision Tree Classifier / Regressor
Splits data into branches based on feature values. Easy to interpret but prone to overfitting.

🧩 Bagging Algorithms
Bagging (Bootstrap Aggregating) is an ensemble technique that reduces variance and overfitting by training multiple models on random subsets of the data and aggregating their results.

Bagging Classifier / Regressor
Wraps any base estimator (e.g., Decision Tree, KNN). Uses majority voting (for classification) or averaging (for regression).
Helps stabilize high-variance models.

Random Forest (Classifier / Regressor)
Builds an ensemble of decision trees trained on bootstrapped samples. Each split considers a random subset of features.
Powerful and widely used for structured data. Offers feature importance.

Extra Trees (Extremely Randomized Trees)
Similar to Random Forest, but adds more randomness by choosing split thresholds randomly instead of selecting the best.
Often faster and less prone to overfitting.

🚀 Boosting Algorithms
Boosting models train weak learners sequentially, where each new model focuses on correcting the errors of the previous one.

Gradient Boosting (Scikit-learn)
Traditional boosting using decision trees. Slower but highly customizable.

XGBoost (Extreme Gradient Boosting)
Highly optimized and regularized gradient boosting framework. Handles missing values, supports early stopping and GPU training.

LightGBM (Light Gradient Boosting Machine)
Faster and memory-efficient gradient boosting. Grows trees leaf-wise. Handles categorical features natively.

CatBoost
Handles categorical variables automatically and requires less preprocessing. Robust to overfitting and highly accurate.

🧠 Other Models
K-Nearest Neighbors (KNN)
Instance-based learner. Predicts based on proximity. Sensitive to feature scaling.

Support Vector Machine (SVM)
Finds the best decision boundary (hyperplane). Works well in high-dimensional spaces.

Naive Bayes
Probabilistic classifier assuming independence among features. Effective for text classification.

Logistic Regression
Linear model used for binary and multi-class classification. Simple and interpretable.

📈 Model Evaluation Techniques Used
Classification Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

Regression Metrics: R² Score, Mean Absolute Error (MAE), Mean Squared Error (MSE), RMSE

Cross-Validation: K-Fold, Stratified K-Fold

Hyperparameter Tuning: GridSearchCV, RandomizedSearchCV

📌 Visualization Tools Used
Matplotlib, Seaborn, Plotly

Used to visualize:

Model predictions

Feature importance

Error distribution

Confusion matrices

Residual plots


