"""Core EDA functions extracted from Task-1_EDA.ipynb.

Functions:
- load_data: load CSV into DataFrame
- summarize: print/return descriptive statistics and variability
- check_missing: return missing value counts
- compute_lossratio: add LossRatio column
- univariate_plots: generate histograms for numerical columns
- correlation_and_heatmap: compute correlation matrix and save heatmap
- temporal_trends: aggregate by month and return aggregated df
"""

# eda.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid")

def data_summary(df):
    """Print structure, missing values, descriptive stats"""
    print("Shape:", df.shape)
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDescriptive Statistics:\n", df.describe())

def univariate_analysis(df, numeric_cols, categorical_cols):
    """Histograms for numerical and bar plots for categorical"""
    for col in numeric_cols:
        plt.figure(figsize=(8,4))
        sns.histplot(df[col].dropna(), bins=50, kde=True)
        plt.title(f"{col} Distribution")
        plt.show()
    
    for col in categorical_cols:
        plt.figure(figsize=(8,4))
        sns.countplot(x=col, data=df)
        plt.title(f"Count of {col}")
        plt.xticks(rotation=45)
        plt.show()

def bivariate_analysis(df):
    """Basic bivariate and correlation plots"""
    # Scatter
    plt.figure(figsize=(8,5))
    sns.scatterplot(x="TotalPremium", y="TotalClaims", data=df, alpha=0.3)
    plt.title("TotalPremium vs TotalClaims")
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(df[["TotalPremium","TotalClaims","CustomValueEstimate","LossRatio"]].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()
