import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. LOAD THE DATASET
# ==========================================
file_name = '/Users/ranjith/Desktop/disease preditions/heart failure .csv'
df = pd.read_csv(file_name)

print('--- Dataset Info ---')
print(f'Dataset Shape: {df.shape}')
print(df.head())

# ==========================================
# 2. DATA PREPROCESSING & ENCODING
# ==========================================
# Target outcome column
target_column = 'HeartDisease'

# Separate features (X) and target outcome (y)
X = df.drop(columns=[target_column])
y = df[target_column]

# Convert categorical text features into numeric dummy/indicator variables
X = pd.get_dummies(X, drop_first=True)

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 3. FEATURE SCALING
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 4. RANDOM FOREST MODEL TRAINING
# ==========================================
rf_classifier = RandomForestClassifier(
    n_estimators=100, max_depth=10, random_state=42
)
rf_classifier.fit(X_train_scaled, y_train)

# ==========================================
# 5. PREDICTING AND EVALUATING OUTCOMES
# ==========================================
y_pred = rf_classifier.predict(X_test_scaled)

print('\n--- Model Outcome Prediction Performance ---')
print(f'Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%\n')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# ==========================================
# 6. VISUALIZATIONS
# ==========================================
# Confusion Matrix Heatmap
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Random Forest - Outcome Prediction Confusion Matrix')
plt.xlabel('Predicted Outcome')
plt.ylabel('Actual Outcome')
plt.tight_layout()
plt.show()

# Feature Importance Plot
feature_importances = pd.Series(
    rf_classifier.feature_importances_, index=X.columns
).sort_values(ascending=False)
plt.figure(figsize=(9, 5))
feature_importances.head(10).plot(kind='barh', color='teal')
plt.title('Top 10 Features Driving Heart Failure Outcome Predictions')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()