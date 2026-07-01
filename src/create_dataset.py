"""
Creates a realistic telecom customer churn dataset.
Deliberately imbalanced — ~27% churn rate, matching real-world telecom data.
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)

n_samples = 500
n_churn = int(n_samples * 0.27)
n_stay = n_samples - n_churn

def make_customers(n, churned):
    if churned:
        # churned customers: shorter tenure, higher monthly charges,
        # more support calls, month-to-month contracts
        tenure_months   = np.random.randint(1, 24, n)
        monthly_charges = np.random.uniform(70, 120, n)
        support_calls   = np.random.randint(3, 10, n)
        contract_month  = np.random.choice([1, 0, 0], n)  # mostly month-to-month
        has_tech_support = np.random.choice([0, 0, 1], n)
        total_charges   = tenure_months * monthly_charges * np.random.uniform(0.9, 1.1, n)
    else:
        # loyal customers: longer tenure, lower charges, fewer calls
        tenure_months   = np.random.randint(12, 72, n)
        monthly_charges = np.random.uniform(20, 90, n)
        support_calls   = np.random.randint(0, 4, n)
        contract_month  = np.random.choice([0, 0, 1], n)  # mostly longer contracts
        has_tech_support = np.random.choice([0, 1, 1], n)
        total_charges   = tenure_months * monthly_charges * np.random.uniform(0.9, 1.1, n)

    return pd.DataFrame({
        "tenure_months":    tenure_months,
        "monthly_charges":  np.round(monthly_charges, 2),
        "total_charges":    np.round(total_charges, 2),
        "support_calls":    support_calls,
        "contract_month_to_month": contract_month,
        "has_tech_support": has_tech_support,
        "senior_citizen":   np.random.choice([0, 1], n, p=[0.84, 0.16]),
        "paperless_billing": np.random.choice([0, 1], n, p=[0.4, 0.6]),
        "churn": [1 if churned else 0] * n
    })

df_churn = make_customers(n_churn, churned=True)
df_stay  = make_customers(n_stay, churned=False)

df = pd.concat([df_churn, df_stay], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/churn.csv", index=False)

print(f"Dataset created: {len(df)} customers")
print(f"\nChurn distribution:")
print(df["churn"].value_counts())
print(f"Churn rate: {df['churn'].mean():.1%}")