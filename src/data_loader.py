# data_loader.py
import pandas as pd

def load_data(path):
    """
    Load CSV and preprocess columns
    """
    df = pd.read_csv(path, low_memory=False)
    
    # Convert date
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
    
    # Create LossRatio
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium'].replace(0, 1)
    
    return df
