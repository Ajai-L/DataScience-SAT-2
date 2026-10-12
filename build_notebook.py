"""
build_notebook.py
Generates the comprehensive Jupyter Notebook for Round 2:
round2_eda_feature_engineering.ipynb
"""

import json

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

# Title & Context
add_md("""# PaySphere FinTech Pvt. Ltd. — Assessment Code: U23AD491
## Round 2: Exploratory Data Analysis (EDA) & Feature Engineering (30 Marks)
### Use Case 13: Digital Payment Insight
**Organization:** PaySphere FinTech Pvt. Ltd. | **Sector:** FinTech  
**Core Business Question:** *Which transaction patterns differ from normal customer behaviour and deserve operational review?*

---
### Notebook Structure:
1. **Environment Setup & Data Ingestion**
2. **Data Audit & Hygiene (Deduplication, Standardization, Imputation, Outlier Isolation)**
3. **Relational Table Integration (Left Joins & Target Labeling)**
4. **Analytical EDA Visualizations (Addressing All 6 Rubric Questions)**
5. **FinTech Feature Engineering (Temporal, Behavioral, Interactions, Encodings)**
6. **Multicollinearity & Feature Selection (Random Forest & Mutual Information)**
7. **Dimensionality Reduction (PCA 2D Projection)**
8. **Export for Round 3 Excel Analysis (`cleaned_paysphere_transactions.csv`)**
""")

# Setup Cell
add_code("""import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA

# Configure publication-grade styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.dpi'] = 150

os.makedirs('eda_plots', exist_ok=True)
print("Environment initialized successfully.")""")

# Step 1: Data Ingestion & Audit
add_md("""## 1. Data Ingestion & Baseline Data Quality Audit
Let's inspect the 5 operational relational tables:
- `transactions.csv` (Base transaction logs)
- `customers.csv` (Customer demographic & segment profile)
- `devices.csv` (Client device hardware & new-device flag)
- `merchants.csv` (Merchant categories & geography)
- `alerts.csv` (Operational risk team review logs)""")

add_code("""df_txns = pd.read_csv("transactions.csv")
df_cust = pd.read_csv("customers.csv")
df_dev = pd.read_csv("devices.csv")
df_merch = pd.read_csv("merchants.csv")
df_alerts = pd.read_csv("alerts.csv")

print("--- BASELINE SHAPES ---")
print(f"Transactions: {df_txns.shape}")
print(f"Customers:    {df_cust.shape}")
print(f"Devices:      {df_dev.shape}")
print(f"Merchants:    {df_merch.shape}")
print(f"Alerts:       {df_alerts.shape}")""")

add_code("""# Audit Duplicates and Missing Values
tables = {'Transactions': df_txns, 'Customers': df_cust, 'Devices': df_dev, 'Merchants': df_merch, 'Alerts': df_alerts}
for name, df in tables.items():
    nulls = df.isnull().sum()
    null_cols = nulls[nulls > 0].to_dict()
    print(f"[{name.upper()}] Duplicates: {df.duplicated().sum()} | Missing: {null_cols if null_cols else 'None'}")""")

# Step 2: Cleaning & Hygiene
add_md("""## 2. Data Cleaning & Table Integration
Adhering to the data quality conditions:
1. **Deduplication:** Remove duplicated transaction IDs (keeping the first authentic event).
2. **Channel Standardization:** Strip inconsistent whitespaces, trim lowercase/uppercase variations.
3. **Imputation:** Lookup missing `merchant_category` from `merchants.csv`.
4. **Device Imputation:** Impute missing device types as `'Unknown'` and missing `new_device_flag` as `0`.
5. **Outlier Flagging:** Isolate extreme amounts via segment-wise IQR flags without deleting rows.""")

add_code("""# 1. Deduplication
init_len = len(df_txns)
df_txns_clean = df_txns.drop_duplicates(subset=['transaction_id'], keep='first').copy()
print(f"Dropped {init_len - len(df_txns_clean)} duplicate transactions. Remaining: {len(df_txns_clean)}")

# 2. Channel Standardization
def clean_channel(ch):
    if pd.isna(ch):
        return 'Unknown'
    c = str(ch).strip().lower()
    c = re.sub(r'[\\s_]+', ' ', c)
    if 'upi' in c: return 'UPI'
    if 'netbanking' in c or 'net banking' in c: return 'NetBanking'
    if 'credit' in c: return 'Credit Card'
    if 'debit' in c: return 'Debit Card'
    if 'wallet' in c: return 'Wallet'
    return c.title()

df_txns_clean['channel'] = df_txns_clean['channel'].apply(clean_channel)
print("Standardized Channels:", df_txns_clean['channel'].unique().tolist())

# 3. Merchant Category Imputation
merch_map = dict(zip(df_merch['merchant_id'], df_merch['merchant_category']))
missing_before = df_txns_clean['merchant_category'].isnull().sum()
df_txns_clean['merchant_category'] = df_txns_clean['merchant_category'].fillna(
    df_txns_clean['merchant_id'].map(merch_map)
).fillna('Unknown')
print(f"Imputed {missing_before - df_txns_clean['merchant_category'].isnull().sum()} missing merchant categories.")

# 4. Device Table Imputation
df_dev_clean = df_dev.copy()
df_dev_clean['device_type'] = df_dev_clean['device_type'].fillna('Unknown')
df_dev_clean['new_device_flag'] = df_dev_clean['new_device_flag'].fillna(0).astype(int)
print("Devices table cleansed.")""")

add_code("""# Relational Table Integration (Left Joins)
df_merged = df_txns_clean.merge(
    df_cust[['customer_id', 'customer_segment', 'region', 'tenure_months']].rename(columns={'region': 'customer_region'}),
    on='customer_id', how='left'
).merge(
    df_dev_clean[['device_id', 'device_type', 'new_device_flag']],
    on='device_id', how='left'
).merge(
    df_merch[['merchant_id', 'region']].rename(columns={'region': 'merchant_region'}),
    on='merchant_id', how='left'
).merge(
    df_alerts[['transaction_id', 'review_status', 'reason_category']],
    on='transaction_id', how='left'
)

df_merged['is_alerted'] = df_merged['review_status'].notnull().astype(int)
df_merged['review_status'] = df_merged['review_status'].fillna('No Alert')
df_merged['reason_category'] = df_merged['reason_category'].fillna('None')

# Segment-wise IQR Outlier Flag
iqr_bounds = df_merged.groupby('customer_segment')['amount'].apply(
    lambda s: s.quantile(0.75) + 1.5 * (s.quantile(0.75) - s.quantile(0.25))
).to_dict()
df_merged['segment_upper_bound'] = df_merged['customer_segment'].map(iqr_bounds)
df_merged['is_iqr_outlier'] = (df_merged['amount'] > df_merged['segment_upper_bound']).astype(int)

print(f"Merged Dataset Shape: {df_merged.shape}")
print(f"Alerted Transactions: {df_merged['is_alerted'].sum()} ({df_merged['is_alerted'].mean()*100:.2f}%)")""")

# Step 3: Analytical EDA Visualizations
add_md("""## 3. Analytical EDA Visualizations
Addressing the 6 specific business questions from the assessment rubric:
1. **Normal amount ranges by customer segment**
2. **Hourly transaction volume spikes**
3. **New devices vs. Known devices**
4. **Merchant categories with high alert volumes**
5. **Regional transaction spikes and cross-region anomalies**
6. **Patterns leading to manual review**""")

add_code("""# Temporal Feature Extraction for EDA
df_merged['time'] = pd.to_datetime(df_merged['time'])
df_merged['hour'] = df_merged['time'].dt.hour
df_merged['day_of_week'] = df_merged['time'].dt.day_name()
df_merged['day_of_week_num'] = df_merged['time'].dt.dayofweek
df_merged['is_weekend'] = df_merged['day_of_week_num'].isin([5, 6]).astype(int)
df_merged['is_night_txn'] = df_merged['hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)""")

add_code("""# Q1: Normal amount ranges by customer segment
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.boxplot(
    data=df_merged, x='customer_segment', y='amount', hue='customer_segment', legend=False,
    order=['Standard', 'SMB', 'Premium', 'Corporate'], palette='Set2', showmeans=True, ax=axes[0]
)
axes[0].set_yscale('log')
axes[0].set_title("Amount Distribution by Customer Segment (Log Scale)", fontweight='bold')
axes[0].set_ylabel("Amount ($) [Log Scale]")

sns.violinplot(
    data=df_merged, x='customer_segment', y='amount', hue='customer_segment', legend=False,
    order=['Standard', 'SMB', 'Premium', 'Corporate'], palette='Pastel1', inner='quartile', cut=0, ax=axes[1]
)
axes[1].set_yscale('log')
axes[1].set_title("Density & Spread by Segment (Violin Plot)", fontweight='bold')
axes[1].set_ylabel("Amount ($) [Log Scale]")

plt.tight_layout()
plt.savefig('eda_plots/q1_amount_ranges_by_segment.png', dpi=300)
plt.show()""")

add_code("""# Q2: Hourly transaction volume spikes & late-night alert surge
hourly = df_merged.groupby('hour').agg(
    total_vol=('transaction_id', 'count'),
    alert_rate=('is_alerted', 'mean')
).reset_index()

fig, ax1 = plt.subplots(figsize=(14, 6))
ax1.bar(
    hourly['hour'], hourly['total_vol'],
    color=['#e74c3c' if h in [0, 1, 2, 3, 4, 5] else '#3498db' for h in hourly['hour']],
    alpha=0.75, edgecolor='black', width=0.7
)
ax1.set_xlabel("Hour of Day (0-23)", fontweight='bold')
ax1.set_ylabel("Total Transactions", fontweight='bold')
ax1.set_xticks(range(24))

ax2 = ax1.twinx()
ax2.plot(hourly['hour'], hourly['alert_rate']*100, color='#c0392b', marker='o', lw=2.5)
ax2.set_ylabel("Alert Rate (%)", color='#c0392b', fontweight='bold')
ax2.grid(False)

plt.title("Hourly Transaction Volume & Alert Rate (Late Night 00:00 - 05:00 Surge)", fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plots/q2_hourly_volume_spikes.png', dpi=300)
plt.show()""")

add_code("""# Q3: New devices vs. Known devices
dev_comp = df_merged.groupby('new_device_flag').agg(
    txn_count=('transaction_id', 'count'),
    mean_amt=('amount', 'mean'),
    alert_rate=('is_alerted', 'mean')
).reset_index()
dev_comp['label'] = dev_comp['new_device_flag'].map({0: 'Known Device (0)', 1: 'New Device (1)'})
dev_comp['alert_pct'] = dev_comp['alert_rate'] * 100

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
sns.barplot(data=dev_comp, x='label', y='txn_count', hue='label', legend=False, palette=['#2980b9', '#e67e22'], ax=axes[0], edgecolor='black')
axes[0].set_title("Total Volume", fontweight='bold')

sns.barplot(data=dev_comp, x='label', y='mean_amt', hue='label', legend=False, palette=['#2980b9', '#e67e22'], ax=axes[1], edgecolor='black')
axes[1].set_title("Mean Amount ($)", fontweight='bold')

sns.barplot(data=dev_comp, x='label', y='alert_pct', hue='label', legend=False, palette=['#27ae60', '#c0392b'], ax=axes[2], edgecolor='black')
axes[2].set_title("Alert Rate (%)", fontweight='bold')

plt.suptitle("New Devices vs. Known Devices: Volume, Spend & Risk Profile", fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('eda_plots/q3_new_vs_known_devices.png', dpi=300, bbox_inches='tight')
plt.show()""")

add_code("""# Q4: Merchant categories with high alert volumes
cat_alerts = df_merged.groupby('merchant_category').agg(
    alert_rate=('is_alerted', 'mean')
).reset_index()
cat_alerts['alert_pct'] = cat_alerts['alert_rate'] * 100
cat_alerts = cat_alerts.sort_values(by='alert_pct', ascending=True)

plt.figure(figsize=(12, 6))
bars = plt.barh(cat_alerts['merchant_category'], cat_alerts['alert_pct'], color=sns.color_palette("rocket", len(cat_alerts)), edgecolor='black')
plt.title("Merchant Categories Ranked by Alert Rate (%)", fontweight='bold')
plt.xlabel("Alert Rate (%)", fontweight='bold')
for bar in bars:
    plt.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.2f}%", va='center', fontweight='bold')
plt.xlim(0, max(cat_alerts['alert_pct']) * 1.18)
plt.tight_layout()
plt.savefig('eda_plots/q4_merchant_categories_alerts.png', dpi=300)
plt.show()""")

add_code("""# Q5: Regional transaction spikes and anomalies
reg_cross = pd.crosstab(df_merged['customer_region'], df_merged['merchant_region'], normalize='all') * 100
reg_alerts = df_merged.groupby(['customer_region', 'merchant_region'])['is_alerted'].mean().unstack() * 100

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(reg_cross, annot=True, fmt=".1f", cmap="Blues", ax=axes[0], linewidths=0.5)
axes[0].set_title("Transaction Share by Region Flow (% of All Txns)", fontweight='bold')

sns.heatmap(reg_alerts, annot=True, fmt=".2f", cmap="YlOrRd", ax=axes[1], linewidths=0.5)
axes[1].set_title("Alert Trigger Rate by Region Flow (%)", fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plots/q5_regional_spikes_and_cross_region_anomalies.png', dpi=300)
plt.show()""")

add_code("""# Q6: Patterns leading to manual review
alerts_only = df_merged[df_merged['is_alerted'] == 1]
review_crosstab = pd.crosstab(alerts_only['reason_category'], alerts_only['review_status'], normalize='index') * 100

plt.figure(figsize=(12, 6))
sns.heatmap(review_crosstab, annot=True, fmt=".1f", cmap="Purples", linewidths=0.8)
plt.title("Alert Reason vs. Review Resolution Status (%)", fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plots/q6_patterns_before_manual_review.png', dpi=300)
plt.show()""")

# Step 4: Feature Engineering
add_md("""## 4. Feature Engineering
Constructing domain-specific FinTech features:
- **Temporal Features:** `hour`, `day_of_week`, `is_weekend`, `is_night_txn`.
- **Behavioral Ratios:** `amount_to_segment_median_ratio`, `high_amount_flag` (> 95th percentile).
- **Interaction Flags:** `new_device_high_amount` (`new_device_flag == 1 & high_amount_flag == 1`), `cross_region_flag` (customer region != merchant region).
- **Categorical Encodings:** One-hot for channel and device type, ordinal for customer segment, frequency encoding for merchant category.""")

add_code("""# Behavioral Ratios
seg_medians = df_merged.groupby('customer_segment')['amount'].median().to_dict()
df_merged['segment_median_amount'] = df_merged['customer_segment'].map(seg_medians)
df_merged['amount_to_segment_median_ratio'] = df_merged['amount'] / df_merged['segment_median_amount']

seg_p95 = df_merged.groupby('customer_segment')['amount'].quantile(0.95).to_dict()
df_merged['segment_p95_amount'] = df_merged['customer_segment'].map(seg_p95)
df_merged['high_amount_flag'] = (df_merged['amount'] > df_merged['segment_p95_amount']).astype(int)

# Interaction Flags
df_merged['new_device_high_amount'] = (
    (df_merged['new_device_flag'] == 1) & (df_merged['high_amount_flag'] == 1)
).astype(int)

df_merged['cross_region_flag'] = (
    df_merged['customer_region'] != df_merged['merchant_region']
).astype(int)

# Categorical Encodings
ch_dummies = pd.get_dummies(df_merged['channel'], prefix='ch', dtype=int)
dev_dummies = pd.get_dummies(df_merged['device_type'], prefix='dev', dtype=int)
encoded_df = pd.concat([df_merged.copy(), ch_dummies, dev_dummies], axis=1)

encoded_df['customer_segment_ord'] = encoded_df['customer_segment'].map({'Standard': 1, 'SMB': 2, 'Premium': 3, 'Corporate': 4})
cat_freq = df_merged['merchant_category'].value_counts(normalize=True).to_dict()
encoded_df['merchant_category_freq'] = encoded_df['merchant_category'].map(cat_freq)

print("Engineered features created successfully.")""")

# Step 5: Feature Selection & PCA
add_md("""## 5. Feature Selection & Dimensionality Reduction
1. **Multicollinearity Matrix:** Pearson correlation heatmap.
2. **Feature Importance:** Random Forest & Mutual Information ranking top predictors of `is_alerted`.
3. **Dimensionality Reduction:** 2-Component PCA standardizing continuous and encoded features.""")

add_code("""# Correlation Heatmap
numeric_features = [
    'amount', 'tenure_months', 'hour', 'day_of_week_num', 'is_weekend', 'is_night_txn',
    'new_device_flag', 'amount_to_segment_median_ratio', 'high_amount_flag',
    'new_device_high_amount', 'cross_region_flag', 'is_iqr_outlier',
    'customer_segment_ord', 'merchant_category_freq', 'is_alerted'
]

corr = encoded_df[numeric_features].corr()
plt.figure(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-0.5, vmax=1.0)
plt.title("Correlation Matrix of Numeric & Engineered Risk Features", fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plots/step5_correlation_matrix.png', dpi=300)
plt.show()""")

add_code("""# Feature Selection via Random Forest and Mutual Information
feature_cols = [
    'amount', 'tenure_months', 'hour', 'is_weekend', 'is_night_txn',
    'new_device_flag', 'amount_to_segment_median_ratio', 'high_amount_flag',
    'new_device_high_amount', 'cross_region_flag', 'is_iqr_outlier',
    'customer_segment_ord', 'merchant_category_freq'
] + [c for c in encoded_df.columns if c.startswith('ch_') or c.startswith('dev_')]

X = encoded_df[feature_cols]
y = encoded_df['is_alerted']

rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, class_weight='balanced')
rf.fit(X, y)
rf_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

mi = mutual_info_classif(X, y, random_state=42)
mi_series = pd.Series(mi, index=feature_cols)

top10_df = pd.DataFrame({'Feature': rf_imp.index[:10], 'RF_Importance': rf_imp.values[:10], 'Mutual_Info': mi_series[rf_imp.index[:10]].values})
print(top10_df)

plt.figure(figsize=(12, 7))
sns.barplot(data=top10_df, x='RF_Importance', y='Feature', hue='Feature', legend=False, palette='viridis', edgecolor='black')
plt.title("Top 10 Most Predictive Features for Operational Alerts", fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plots/step5_top10_feature_importances.png', dpi=300)
plt.show()""")

add_code("""# PCA (2 Principal Components)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
encoded_df['PC1'] = X_pca[:, 0]
encoded_df['PC2'] = X_pca[:, 1]

print(f"PCA Variance: PC1 = {pca.explained_variance_ratio_[0]*100:.2f}%, PC2 = {pca.explained_variance_ratio_[1]*100:.2f}%")

plt.figure(figsize=(12, 8))
sns.scatterplot(data=encoded_df, x='PC1', y='PC2', hue='is_alerted', palette={0: '#3498db', 1: '#e74c3c'}, alpha=0.6, s=40)
plt.title("PCA 2D Projection: Normal vs. Alerted Transactions", fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plots/step5_pca_2d_projection.png', dpi=300)
plt.show()""")

# Step 6: Deliverable Export
add_md("""## 6. Final Deliverable Export
Exporting the enriched, cleaned master dataset to `cleaned_paysphere_transactions.csv` ready for Round 3 Excel analysis (Pivot Tables and Pivot Charts).""")

add_code("""export_cols = [
    'transaction_id', 'customer_id', 'merchant_id', 'device_id',
    'time', 'hour', 'day_of_week', 'is_weekend', 'is_night_txn',
    'amount', 'customer_segment', 'customer_region', 'tenure_months',
    'merchant_category', 'merchant_region', 'channel',
    'device_type', 'new_device_flag',
    'amount_to_segment_median_ratio', 'segment_median_amount', 'segment_p95_amount',
    'high_amount_flag', 'new_device_high_amount', 'cross_region_flag',
    'is_iqr_outlier', 'is_alerted', 'review_status', 'reason_category',
    'PC1', 'PC2'
]

df_export = encoded_df[export_cols].copy()
df_export.to_csv("cleaned_paysphere_transactions.csv", index=False)
print(f"Successfully exported cleaned_paysphere_transactions.csv ({df_export.shape[0]} rows, {df_export.shape[1]} cols).")""")

notebook = {
    "cells": cells,
    "metadata": {
        "language_info": {"name": "python", "version": "3.14.4"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("round2_eda_feature_engineering.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Created round2_eda_feature_engineering.ipynb successfully!")
