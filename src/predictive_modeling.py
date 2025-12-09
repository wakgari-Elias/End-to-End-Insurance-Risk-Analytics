# src/predictive_modeling.py
# Task 4: Risk-Based Premium Engine (Production Ready)
# AlphaCare Insurance Solutions

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor
import shap
import matplotlib.pyplot as plt
import os

# Create plots folder
os.makedirs("../plots", exist_ok=True)

def load_and_prepare_data(path="../data/MachineLearningRating_v3.csv"):
    print("Loading data...")
    df = pd.read_csv(path, low_memory=False)
    
    # Convert numeric
    df['TotalPremium'] = pd.to_numeric(df['TotalPremium'], errors='coerce')
    df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce')
    
    # Feature engineering
    df['VehicleAge'] = 2025 - df['RegistrationYear']
    df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)
    
    # Correct column names
    cat_cols = ['Province', 'Gender', 'VehicleType', 'Make', 'CoverType']
    num_cols = ['SumInsured', 'CalculatedPremiumPerTerm', 'VehicleAge', 'cubiccapacity', 'kilowatts']
    
    features = cat_cols + num_cols
    X = pd.get_dummies(df[features], columns=cat_cols, drop_first=True).fillna(0)
    
    print(f"Final feature count: {X.shape[1]}")
    return df, X

def train_probability_model(X, y):
    print("\nTraining Claim Probability Model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = XGBClassifier(n_estimators=300, max_depth=6, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"Claim Probability AUC: {auc:.4f}")
    
    return model, X_test

def train_severity_model(df_claims):
    print("\nTraining Claim Severity Model (only policies with claims)...")
    cat_cols = ['Province', 'Gender', 'VehicleType', 'Make', 'CoverType']
    num_cols = ['SumInsured', 'CalculatedPremiumPerTerm', 'VehicleAge', 'cubiccapacity', 'kilowatts']
    features = cat_cols + num_cols
    
    X_sev = pd.get_dummies(df_claims[features], columns=cat_cols, drop_first=True).fillna(0)
    y_sev = df_claims['TotalClaims']
    
    X_train, X_test, y_train, y_test = train_test_split(X_sev, y_sev, test_size=0.2, random_state=42)
    
    model = XGBRegressor(n_estimators=500, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    
    print(f"Claim Severity → RMSE: R{rmse:,.0f} | R²: {r2:.3f}")
    
    return model, X_test

def generate_shap_plot(sev_model, X_sample):
    print("\nGenerating SHAP plot...")
    explainer = shap.TreeExplainer(sev_model)
    shap_values = explainer.shap_values(X_sample.sample(500, random_state=42))
    
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample.sample(500, random_state=42), 
                       plot_type="bar", max_display=10, show=False)
    plt.title("Top 10 Features Driving Claim Severity (SHAP)")
    plt.tight_layout()
    plt.savefig("../plots/shap_task4.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("SHAP plot saved!")

def calculate_risk_premium(prob_model, sev_model, X_test, X_test_aligned):
    print("\nCalculating Risk-Based Premium...")
    sample = X_test.copy()
    sample['P_Claim'] = prob_model.predict_proba(sample)[:, 1]
    sample_aligned = sample.reindex(columns=X_test_aligned.columns, fill_value=0)
    sample['Predicted_Severity'] = sev_model.predict(sample_aligned)
    sample['Risk_Based_Premium'] = sample['P_Claim'] * sample['Predicted_Severity'] * 1.3
    
    print("First 10 Risk-Based Premiums:")
    print(sample[['P_Claim', 'Predicted_Severity', 'Risk_Based_Premium']].head(10).round(2))
    return sample

# MAIN EXECUTION
if __name__ == "__main__":
    print("="*100)
    print("TASK 4: RISK-BASED PREMIUM ENGINE")
    print("="*100)
    
    df, X = load_and_prepare_data()
    
    # Probability model
    prob_model, X_test = train_probability_model(X, df['HasClaim'])
    
    # Severity model
    df_claims = df[df['TotalClaims'] > 0]
    sev_model, X_test_sev = train_severity_model(df_claims)
    
    # SHAP
    generate_shap_plot(sev_model, X_test_sev)
    
    # Risk-Based Premium
    risk_premium_df = calculate_risk_premium(prob_model, sev_model, X_test, X_test_sev)
    
    print("\nTASK-4 COMPLETE — READY FOR PRODUCTION")
    print("Next: Deploy as API")