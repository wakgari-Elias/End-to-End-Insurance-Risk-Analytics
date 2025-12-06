"""Utility functions for quick aggregations used in EDA."""
import pandas as pd

def top_k_by_metric(df, group_col, metric_col, k=10, agg='mean'):
    """Return top-k groups by aggregated metric_col."""
    if agg == 'mean':
        grouped = df.groupby(group_col)[metric_col].mean()
    elif agg == 'sum':
        grouped = df.groupby(group_col)[metric_col].sum()
    else:
        grouped = df.groupby(group_col)[metric_col].agg(agg)
    return grouped.sort_values(ascending=False).head(k)

def lossratio_by_group(df, group_col):
    """Return average LossRatio by group_col (if LossRatio exists)."""
    if 'LossRatio' not in df.columns:
        raise KeyError('LossRatio column not in df. Call compute_lossratio first.')
    return df.groupby(group_col)['LossRatio'].mean().sort_values()
# utils.py
def calc_loss_ratio(total_claims, total_premium):
    """
    Calculate LossRatio safely
    """
    total_premium = total_premium if total_premium != 0 else 1
    return total_claims / total_premium
