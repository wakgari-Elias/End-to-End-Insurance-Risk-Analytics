"""Visualization helpers for EDA package."""
# visualizations.py
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def save_fig(plt_obj, path):
    """Save a matplotlib.pyplot object to `path`. Accepts Path or str."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt_obj.savefig(str(p), bbox_inches='tight', dpi=150)
    except Exception:
        # fallback: try to use current figure
        plt_obj.gcf().savefig(str(p), bbox_inches='tight', dpi=150)




sns.set(style="whitegrid")

def loss_ratio_by_province(df):
    prov_loss = df.groupby("Province")["LossRatio"].mean().sort_values()
    plt.figure(figsize=(12,7))
    prov_loss.plot(kind="barh")
    plt.title("Average Loss Ratio by Province")
    plt.xlabel("Loss Ratio")
    plt.ylabel("Province")
    plt.tight_layout()
    plt.show()

def avg_claim_by_vehicle_make(df, top_n=15):
    make_claims = df.groupby("Make")["TotalClaims"].mean().sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(14,6))
    make_claims.plot(kind="bar", color="skyblue")
    plt.title(f"Top {top_n} Vehicle Makes by Average Claim Amount")
    plt.ylabel("Average Claim Amount")
    plt.xlabel("Vehicle Make")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def premium_vs_claims_postal(df, top_n=15):
    top_postal = df.groupby("PostalCode")["TotalClaims"].sum().sort_values(ascending=False).head(top_n).index
    df_top = df[df["PostalCode"].isin(top_postal)]
    
    plt.figure(figsize=(11,7))
    sns.scatterplot(
        data=df_top,
        x="TotalPremium",
        y="TotalClaims",
        hue="PostalCode",
        alpha=0.7,
        palette="tab10"
    )
    plt.title(f"Premium vs Claims for Top {top_n} PostalCodes")
    plt.xlabel("Total Premium")
    plt.ylabel("Total Claims")
    plt.legend(title="PostalCode", bbox_to_anchor=(1.02,1), loc='upper left')
    plt.tight_layout()
    plt.show()
