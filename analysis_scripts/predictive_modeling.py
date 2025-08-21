import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import shap
import matplotlib.pyplot as plt
import os

# Configuration
OUTPUT_DIR = "analysis_outputs/model_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and preprocess data
def load_and_preprocess_data():
    """Load the dataset and perform initial preprocessing."""
    df = pd.read_csv('data/MachineLearningRating_v3.txt', sep='|', low_memory=False)
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
    return df

def prepare_features(df):
    """Prepare features and target for modeling."""
    features = ['TotalPremium', 'RegistrationYear', 'Province', 'VehicleType']
    target = 'TotalClaims'
    df_model = df[features + [target]].dropna()
    X = pd.get_dummies(df_model.drop(target, axis=1), columns=['Province', 'VehicleType'])
    y = df_model[target]
    return X, y

# Load data
df = load_and_preprocess_data()
X, y = prepare_features(df)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split for modeling: Train {X_train.shape}, Test {X_test.shape}")

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
print(f"Linear Regression Mean Squared Error: {mse_lr:.4f}")

# Cross-validation for Linear Regression
cv_scores_lr = cross_val_score(lr_model, X, y, cv=5, scoring='neg_mean_squared_error')
print(f"Linear Regression CV MSE (mean): {-cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std() * 2:.4f})")

# Feature importance for Linear Regression
feature_importance_lr = pd.DataFrame({'Feature': X.columns, 'Coefficient': lr_model.coef_})
print("Linear Regression Feature Importance:\n", feature_importance_lr.sort_values(by='Coefficient', ascending=False))

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"Random Forest Mean Squared Error: {mse_rf:.4f}")

# Cross-validation for Random Forest
cv_scores_rf = cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_squared_error')
print(f"Random Forest CV MSE (mean): {-cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std() * 2:.4f})")

# Feature importance for Random Forest
feature_importance_rf = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
print("Random Forest Feature Importance:\n", feature_importance_rf.sort_values(by='Importance', ascending=False))

# Interpretability with SHAP
explainer = shap.TreeExplainer(rf_model)  # Use TreeExplainer for Random Forest
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("SHAP Feature Importance")
plt.savefig(os.path.join(OUTPUT_DIR, "shap_feature_importance.png"))
plt.close()

# Save model outputs
with open(os.path.join(OUTPUT_DIR, "model_evaluation.txt"), "w") as f:
    f.write(f"Linear Regression MSE: {mse_lr:.4f}\nRandom Forest MSE: {mse_rf:.4f}\n")
feature_importance_lr.to_csv(os.path.join(OUTPUT_DIR, "lr_feature_importance.csv"), index=False)
feature_importance_rf.to_csv(os.path.join(OUTPUT_DIR, "rf_feature_importance.csv"), index=False)