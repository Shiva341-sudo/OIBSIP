# -*- coding: utf-8 -*-
"""
Iris Flower Classification with Visualization

This script performs a complete machine learning workflow for iris flower classification:
- Loads the Iris dataset and performs exploratory data analysis (EDA)
- Visualizes class distribution, relationships between features, and correlations
- Trains a Decision Tree classifier and evaluates performance
- Generates confusion matrix and feature importance visualizations

Dataset: Iris.csv (150 samples, 4 features, 3 species classes)
Target: Species classification (Setosa, Versicolor, Virginica)

Key Steps:
1. Data Loading & Exploration: Display dataset info, statistics, and visualizations
2. Data Preparation: Encode target labels and normalize features
3. Train-Test Split: 80-20 split with stratification
4. Model Training: Decision Tree classifier
5. Evaluation: Accuracy, classification report, confusion matrix
6. Feature Analysis: Identify most important features

Author: Shiva341-sudo
"""

# Iris Flower Classification with Visualization
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("/content/Iris.csv")

# Remove Id column
df = df.drop("Id", axis=1)


# Exploratory Data Analysis

print(df.head())
print(df.info())
print(df.describe())
# 1. Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Species', data=df)
plt.title("Class Distribution")
plt.show()

# 2. Pair Plot

sns.pairplot(df, hue='Species')
plt.show()

# 3. Correlation Heatmap

plt.figure(figsize=(8,6))
sns.heatmap(
    df.drop('Species', axis=1).corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("Feature Correlation Heatmap")
plt.show()

# Data Preparation

X = df.drop("Species", axis=1)
y = df["Species"]
le = LabelEncoder()
y = le.fit_transform(y)

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y   # ← add stratify=y
)

# Model Training

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction

y_pred = model.predict(X_test)

# Evaluation

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=le.classes_
)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Feature Importance

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)
plt.figure(figsize=(8,5))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance
)
plt.title("Feature Importance")
plt.show()
