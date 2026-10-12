"""
run_round2_analysis.py
================================================================================
PaySphere FinTech Pvt. Ltd. | Assessment Code: U23AD491
Use Case 13: Digital Payment Insight
Round 2: Exploratory Data Analysis & Feature Engineering (30 Marks)
================================================================================
Steps Covered:
1. Data Ingestion & Baseline Quality Auditing
2. Cleaning, Standardizing, Imputation & Relational Merging
3. Analytical EDA addressing all 6 Rubric Questions (saved to ./eda_plots/)
4. Domain-specific FinTech Feature Engineering
5. Multicollinearity, Feature Selection (Random Forest / Mutual Info), and PCA
6. Export of cleaned_paysphere_transactions.csv and reporting metrics
================================================================================
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA

# Set style for publication-grade visualizations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300

os.makedirs('eda_plots', exist_ok=True)

print("="*80)
print("ROUND 2: DATA CLEANING, EDA & FEATURE ENGINEERING PIPELINE")
print("PaySphere FinTech Pvt. Ltd. | Use Case 13: Digital Payment Insight")
print("="*80)

# ==============================================================================
# STEP 2: DATA AUDIT, CLEANING & TABLE INTEGRATION
# ==============================================================================
print("\n>>> STEP 2.1: Loading Datasets and Auditing Baseline Data Quality...")

df_txns = pd.read_csv("transactions.csv")
df_cust = pd.read_csv("customers.csv")
df_dev = pd.read_csv("devices.csv")
df_merch = pd.read_csv("merchants.csv")
df_alerts = pd.read_csv("alerts.csv")

tables = {
    'transactions': df_txns,
    'customers': df_cust,
    'devices': df_dev,
    'merchants': df_merch,
    'alerts': df_alerts
}

audit_summary = []
for name, df in tables.items():
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0].to_dict()
    dup_count = df.duplicated().sum()
    audit_summary.append({
        'Table': name,
        'Rows': df.shape[0],
        'Columns': df.shape[1],
        'Duplicates': dup_count,
        'Nulls by Column': null_cols if null_cols else 'None'
    })
    print(f"[{name.upper()}] Shape: {df.shape} | Duplicates: {dup_count} | Nulls: {null_cols if null_cols else 'None'}")

# --- Data Cleaning ---
print("\n>>> STEP 2.2: Executing Data Hygiene Operations...")

# 1. Deduplicate transactions
init_txn_count = len(df_txns)
df_txns_clean = df_txns.drop_duplicates(subset=['transaction_id'], keep='first').copy()
dropped_dups = init_txn_count - len(df_txns_clean)
print(f" [Hygiene] Deduplication: Dropped {dropped_dups} duplicate transaction records. Clean shape: {df_txns_clean.shape}")

# 2. Standardize channel labels
# Trim whitespace, replace underscores/spaces, normalize casing
def clean_channel(ch):
    if pd.isna(ch):
        return 'Unknown'
    ch_str = str(ch).strip().lower()
    ch_str = re.sub(r'[\s_]+', ' ', ch_str)
    if 'upi' in ch_str:
        return 'UPI'
    elif 'netbanking' in ch_str or 'net banking' in ch_str:
        return 'NetBanking'
    elif 'credit' in ch_str:
        return 'Credit Card'
    elif 'debit' in ch_str:
        return 'Debit Card'
    elif 'wallet' in ch_str:
        return 'Wallet'
    else:
        return ch_str.title()

df_txns_clean['channel_raw'] = df_txns_clean['channel']
df_txns_clean['channel'] = df_txns_clean['channel'].apply(clean_channel)
print(f" [Hygiene] Standardized Channel Labels. Unique channels: {df_txns_clean['channel'].unique().tolist()}")

# 3. Impute missing merchant_category using merchants.csv lookup
merch_cat_lookup = dict(zip(df_merch['merchant_id'], df_merch['merchant_category']))
missing_before = df_txns_clean['merchant_category'].isnull().sum()
df_txns_clean['merchant_category'] = df_txns_clean['merchant_category'].fillna(
    df_txns_clean['merchant_id'].map(merch_cat_lookup)
)
# Any remaining unresolved cases mapped to 'Unknown'
df_txns_clean['merchant_category'] = df_txns_clean['merchant_category'].fillna('Unknown')
missing_after = df_txns_clean['merchant_category'].isnull().sum()
print(f" [Hygiene] Merchant Category Imputation: Imputed {missing_before - missing_after} missing categories using merchants lookup table.")

# 4. Clean devices.csv
df_dev_clean = df_dev.copy()
df_dev_clean['device_type'] = df_dev_clean['device_type'].fillna('Unknown')
# new_device_flag mode / 0 imputation (assume 0 if unknown)
mode_flag = df_dev_clean['new_device_flag'].dropna().mode()[0]
df_dev_clean['new_device_flag'] = df_dev_clean['new_device_flag'].fillna(0).astype(int)
print(f" [Hygiene] Devices Table: Imputed null device_type with 'Unknown' and missing new_device_flag with 0.")

# 5. Outlier Detection on Amount (IQR Method per segment to isolate anomalies without deletion)
print(" [Hygiene] Computing segment-wise IQR outlier flags...")
# 6. Relational Table Merging onto transactions base table
print("\n>>> STEP 2.3: Merging Relational Datasets onto Transactions...")
# Merge with customers
df_merged = df_txns_clean.merge(
    df_cust[['customer_id', 'customer_segment', 'region', 'tenure_months']].rename(columns={'region': 'customer_region'}),
    on='customer_id',
    how='left'
)

# Merge with devices
df_merged = df_merged.merge(
    df_dev_clean[['device_id', 'device_type', 'new_device_flag']],
    on='device_id',
    how='left'
)
df_merged['device_type'] = df_merged['device_type'].fillna('Unknown')
df_merged['new_device_flag'] = df_merged['new_device_flag'].fillna(0).astype(int)

# Merge with merchants
df_merged = df_merged.merge(
    df_merch[['merchant_id', 'region']].rename(columns={'region': 'merchant_region'}),
    on='merchant_id',
    how='left'
)

# Merge with alerts
df_merged = df_merged.merge(
    df_alerts[['transaction_id', 'review_status', 'reason_category']],
    on='transaction_id',
    how='left'
)

# Create binary target: is_alerted
df_merged['is_alerted'] = df_merged['review_status'].notnull().astype(int)
df_merged['review_status'] = df_merged['review_status'].fillna('No Alert')
df_merged['reason_category'] = df_merged['reason_category'].fillna('None')

# Compute Segment-wise IQR Outlier Flag cleanly
iqr_bounds = df_merged.groupby('customer_segment')['amount'].apply(
    lambda s: s.quantile(0.75) + 1.5 * (s.quantile(0.75) - s.quantile(0.25))
).to_dict()
df_merged['segment_upper_bound'] = df_merged['customer_segment'].map(iqr_bounds)
df_merged['is_iqr_outlier'] = (df_merged['amount'] > df_merged['segment_upper_bound']).astype(int)

print(f"Merged Dataset Shape: {df_merged.shape}")
print(f"Total Transactions: {len(df_merged)} | Alerted Transactions: {df_merged['is_alerted'].sum()} ({df_merged['is_alerted'].mean()*100:.2f}%)")

# ==============================================================================
# STEP 3: ANALYTICAL EDA VISUALIZATIONS (Addressing 6 Core Business Questions)
# ==============================================================================
print("\n>>> STEP 3: Creating Publication-Quality Visualizations for 6 Business Questions...")

df_merged['time'] = pd.to_datetime(df_merged['time'])
df_merged['hour'] = df_merged['time'].dt.hour
df_merged['day_of_week'] = df_merged['time'].dt.day_name()
df_merged['day_of_week_num'] = df_merged['time'].dt.dayofweek
df_merged['is_weekend'] = df_merged['day_of_week_num'].isin([5, 6]).astype(int)
df_merged['is_night_txn'] = df_merged['hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)

# --- Question 1: What are normal amount ranges by segment? ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Boxplot on Log Scale
sns.boxplot(
    data=df_merged,
    x='customer_segment',
    y='amount',
    hue='customer_segment',
    legend=False,
    order=['Standard', 'SMB', 'Premium', 'Corporate'],
    palette='Set2',
    showmeans=True,
    meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"red", "markersize":"7"},
    ax=axes[0]
)
axes[0].set_yscale('log')
axes[0].set_title("Transaction Amount Distribution by Customer Segment (Log Scale)", fontweight='bold')
axes[0].set_xlabel("Customer Segment")
axes[0].set_ylabel("Transaction Amount ($) [Log Scale]")

# Violin plot with median markers
sns.violinplot(
    data=df_merged,
    x='customer_segment',
    y='amount',
    hue='customer_segment',
    legend=False,
    order=['Standard', 'SMB', 'Premium', 'Corporate'],
    palette='Pastel1',
    inner='quartile',
    cut=0,
    ax=axes[1]
)
axes[1].set_yscale('log')
axes[1].set_title("Density & Spread of Amounts by Segment (Violin Plot)", fontweight='bold')
axes[1].set_xlabel("Customer Segment")
axes[1].set_ylabel("Transaction Amount ($) [Log Scale]")

plt.tight_layout()
q1_path = 'eda_plots/q1_amount_ranges_by_segment.png'
plt.savefig(q1_path, dpi=300)
plt.close()
print(f" [Saved] Question 1 Plot: {q1_path}")

# --- Question 2: Which hours show unusual volume? ---
hourly_stats = df_merged.groupby('hour').agg(
    total_volume=('transaction_id', 'count'),
    alert_volume=('is_alerted', 'sum'),
    alert_rate=('is_alerted', 'mean'),
    avg_amount=('amount', 'mean')
).reset_index()

fig, ax1 = plt.subplots(figsize=(14, 6))
bars = ax1.bar(
    hourly_stats['hour'],
    hourly_stats['total_volume'],
    color=['#e74c3c' if h in [0, 1, 2, 3, 4, 5] else '#3498db' for h in hourly_stats['hour']],
    alpha=0.75,
    edgecolor='black',
    width=0.7,
    label='Total Volume (Red = Night Hours 0-5 AM)'
)

ax1.set_xlabel("Hour of the Day (0-23)", fontweight='bold')
ax1.set_ylabel("Total Transaction Count", color='#2c3e50', fontweight='bold')
ax1.set_xticks(range(24))
ax1.grid(axis='y', linestyle='--', alpha=0.6)

# Overlay alert rate line on twin axis
ax2 = ax1.twinx()
line = ax2.plot(
    hourly_stats['hour'],
    hourly_stats['alert_rate'] * 100,
    color='#c0392b',
    linewidth=2.5,
    marker='o',
    markersize=6,
    label='Alert Rate (%)'
)
ax2.set_ylabel("Alert Rate (%)", color='#c0392b', fontweight='bold')
ax2.grid(False)

# Annotate late-night anomaly spike
peak_night_hr = hourly_stats[hourly_stats['hour'].isin([2, 3])].sort_values(by='alert_rate', ascending=False).iloc[0]
ax2.annotate(
    f"Night Alert Spike: {peak_night_hr['alert_rate']*100:.1f}%\nat {int(peak_night_hr['hour'])}:00 hrs",
    xy=(peak_night_hr['hour'], peak_night_hr['alert_rate']*100),
    xytext=(peak_night_hr['hour'] + 1.5, peak_night_hr['alert_rate']*100 + 4),
    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
    fontweight='bold',
    bbox=dict(boxstyle="round,pad=0.3", fc="#fadbd8", ec="#e74c3c", lw=1.5)
)

plt.title("Hourly Transaction Volume & Alert Rate Surge (Highlighting 00:00 - 05:00 hrs Anomaly)", fontweight='bold', pad=15)
fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
plt.tight_layout()
q2_path = 'eda_plots/q2_hourly_volume_spikes.png'
plt.savefig(q2_path, dpi=300)
plt.close()
print(f" [Saved] Question 2 Plot: {q2_path}")

# --- Question 3: How do new devices differ from known devices? ---
device_comp = df_merged.groupby('new_device_flag').agg(
    txn_count=('transaction_id', 'count'),
    mean_amount=('amount', 'mean'),
    median_amount=('amount', 'median'),
    alert_count=('is_alerted', 'sum'),
    alert_rate=('is_alerted', 'mean')
).reset_index()
device_comp['device_label'] = device_comp['new_device_flag'].map({0: 'Known Device (0)', 1: 'New Device (1)'})

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Total Volume
sns.barplot(data=device_comp, x='device_label', y='txn_count', hue='device_label', legend=False, palette=['#2980b9', '#e67e22'], ax=axes[0], edgecolor='black')
axes[0].set_title("Total Transaction Volume", fontweight='bold')
axes[0].set_ylabel("Transaction Count")
axes[0].set_xlabel("")
for p in axes[0].patches:
    axes[0].annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                     ha='center', va='center', color='white', fontweight='bold', fontsize=11)

# 2. Average Spend
sns.barplot(data=device_comp, x='device_label', y='mean_amount', hue='device_label', legend=False, palette=['#2980b9', '#e67e22'], ax=axes[1], edgecolor='black')
axes[1].set_title("Average Spend ($)", fontweight='bold')
axes[1].set_ylabel("Mean Amount ($)")
axes[1].set_xlabel("")
for p in axes[1].patches:
    axes[1].annotate(f"${p.get_height():,.1f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                     ha='center', va='center', color='white', fontweight='bold', fontsize=11)

# 3. Alert Rate (%)
device_comp['alert_pct'] = device_comp['alert_rate'] * 100
sns.barplot(data=device_comp, x='device_label', y='alert_pct', hue='device_label', legend=False, palette=['#27ae60', '#c0392b'], ax=axes[2], edgecolor='black')
axes[2].set_title("Alert Trigger Rate (%)", fontweight='bold')
axes[2].set_ylabel("Alert Rate (%)")
axes[2].set_xlabel("")
for p in axes[2].patches:
    axes[2].annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                     ha='center', va='center', color='white', fontweight='bold', fontsize=11)

plt.suptitle("New Devices vs. Known Devices: Volume, Spend & Risk Profiling", fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
q3_path = 'eda_plots/q3_new_vs_known_devices.png'
plt.savefig(q3_path, dpi=300, bbox_inches='tight')
plt.close()
print(f" [Saved] Question 3 Plot: {q3_path}")

# --- Question 4: Which merchant categories show more alerts? ---
cat_alerts = df_merged.groupby('merchant_category').agg(
    total_txns=('transaction_id', 'count'),
    alerts=('is_alerted', 'sum'),
    alert_rate=('is_alerted', 'mean'),
    total_amt=('amount', 'sum')
).reset_index()
cat_alerts['alert_pct'] = cat_alerts['alert_rate'] * 100
cat_alerts = cat_alerts.sort_values(by='alert_pct', ascending=True)

plt.figure(figsize=(12, 6))
bars = plt.barh(
    cat_alerts['merchant_category'],
    cat_alerts['alert_pct'],
    color=sns.color_palette("rocket", len(cat_alerts)),
    edgecolor='black'
)
plt.title("Merchant Categories Ranked by Alert Rate (%)", fontweight='bold', fontsize=13, pad=12)
plt.xlabel("Alert Rate (%)", fontweight='bold')
plt.ylabel("Merchant Category", fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.6)

for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.3, bar.get_y() + bar.get_height()/2,
             f"{width:.2f}%", ha='left', va='center', fontweight='bold', fontsize=10)

plt.xlim(0, max(cat_alerts['alert_pct']) * 1.18)
plt.tight_layout()
q4_path = 'eda_plots/q4_merchant_categories_alerts.png'
plt.savefig(q4_path, dpi=300)
plt.close()
print(f" [Saved] Question 4 Plot: {q4_path}")

# --- Question 5: Which regions have unusual spikes? ---
# Comparing customer region vs merchant region flow
reg_cross = pd.crosstab(
    df_merged['customer_region'],
    df_merged['merchant_region'],
    values=df_merged['amount'],
    aggfunc='count',
    normalize='all'
) * 100

reg_alerts = df_merged.groupby(['customer_region', 'merchant_region']).agg(
    alert_rate=('is_alerted', 'mean')
).unstack() * 100

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(reg_cross, annot=True, fmt=".1f", cmap="Blues", cbar=True, ax=axes[0], linewidths=0.5)
axes[0].set_title("Cross-Regional Transaction Share (% of All Txns)", fontweight='bold')
axes[0].set_xlabel("Merchant Region", fontweight='bold')
axes[0].set_ylabel("Customer Region", fontweight='bold')

sns.heatmap(reg_alerts['alert_rate'], annot=True, fmt=".2f", cmap="YlOrRd", cbar=True, ax=axes[1], linewidths=0.5)
axes[1].set_title("Cross-Regional Alert Trigger Rate (%)", fontweight='bold')
axes[1].set_xlabel("Merchant Region", fontweight='bold')
axes[1].set_ylabel("Customer Region", fontweight='bold')

plt.tight_layout()
q5_path = 'eda_plots/q5_regional_spikes_and_cross_region_anomalies.png'
plt.savefig(q5_path, dpi=300)
plt.close()
print(f" [Saved] Question 5 Plot: {q5_path}")

# --- Question 6: What patterns appear before manual review? ---
# Subsetting alerted transactions to inspect review_status and reason_category
alerts_only = df_merged[df_merged['is_alerted'] == 1].copy()

# Cross-tab of Reason Category vs Review Status
status_reason_ct = pd.crosstab(
    alerts_only['reason_category'],
    alerts_only['review_status'],
    normalize='index'
) * 100

plt.figure(figsize=(12, 6))
sns.heatmap(status_reason_ct, annot=True, fmt=".1f", cmap="Purples", cbar=True, linewidths=0.8)
plt.title("Alert Reason Category vs. Review Resolution Status (% Breakdown)", fontweight='bold', fontsize=13, pad=12)
plt.xlabel("Review Status", fontweight='bold')
plt.ylabel("Reason Category", fontweight='bold')
plt.tight_layout()
q6_path = 'eda_plots/q6_patterns_before_manual_review.png'
plt.savefig(q6_path, dpi=300)
plt.close()
print(f" [Saved] Question 6 Plot: {q6_path}")

# ==============================================================================
# STEP 4: FEATURE ENGINEERING
# ==============================================================================
print("\n>>> STEP 4: Constructing Domain-Specific FinTech Features...")

# 1. Behavioral Ratios:
# amount_to_segment_median_ratio: amount divided by customer segment median
seg_medians = df_merged.groupby('customer_segment')['amount'].median().to_dict()
df_merged['segment_median_amount'] = df_merged['customer_segment'].map(seg_medians)
df_merged['amount_to_segment_median_ratio'] = df_merged['amount'] / df_merged['segment_median_amount']

# high_amount_flag: amount exceeding 95th percentile of that segment
seg_p95 = df_merged.groupby('customer_segment')['amount'].quantile(0.95).to_dict()
df_merged['segment_p95_amount'] = df_merged['customer_segment'].map(seg_p95)
df_merged['high_amount_flag'] = (df_merged['amount'] > df_merged['segment_p95_amount']).astype(int)

# 2. Interaction Flags:
# new_device_high_amount: new_device_flag == 1 and high_amount_flag == 1
df_merged['new_device_high_amount'] = (
    (df_merged['new_device_flag'] == 1) & (df_merged['high_amount_flag'] == 1)
).astype(int)

# cross_region_flag: customer_region != merchant_region
df_merged['cross_region_flag'] = (
    df_merged['customer_region'] != df_merged['merchant_region']
).astype(int)

# 3. Categorical Encodings:
# Low-cardinality nominal one-hot encoding: channel, device_type
ch_dummies = pd.get_dummies(df_merged['channel'], prefix='ch', dtype=int)
dev_dummies = pd.get_dummies(df_merged['device_type'], prefix='dev', dtype=int)
encoded_df = pd.concat([df_merged.copy(), ch_dummies, dev_dummies], axis=1)

# Ordinal / Frequency Encodings
# Customer segment ordinal mapping based on tier/purchasing power
segment_order = {'Standard': 1, 'SMB': 2, 'Premium': 3, 'Corporate': 4}
encoded_df['customer_segment_ord'] = encoded_df['customer_segment'].map(segment_order)

# Merchant category frequency encoding
cat_freq = df_merged['merchant_category'].value_counts(normalize=True).to_dict()
encoded_df['merchant_category_freq'] = encoded_df['merchant_category'].map(cat_freq)

# Customer region & merchant region frequency encodings
reg_freq = df_merged['customer_region'].value_counts(normalize=True).to_dict()
encoded_df['customer_region_freq'] = encoded_df['customer_region'].map(reg_freq)

print(" [Engineered] Created Features:")
print("  - Temporal: hour, day_of_week_num, is_weekend, is_night_txn")
print("  - Ratios: amount_to_segment_median_ratio, high_amount_flag")
print("  - Interaction: new_device_high_amount, cross_region_flag")
print("  - Encodings: One-hot channels, one-hot devices, ordinal segment, freq category")

# ==============================================================================
# STEP 5: FEATURE SELECTION & DIMENSIONALITY REDUCTION
# ==============================================================================
print("\n>>> STEP 5.1: Multicollinearity Assessment & Correlation Matrix...")

numeric_features = [
    'amount', 'tenure_months', 'hour', 'day_of_week_num', 'is_weekend', 'is_night_txn',
    'new_device_flag', 'amount_to_segment_median_ratio', 'high_amount_flag',
    'new_device_high_amount', 'cross_region_flag', 'is_iqr_outlier',
    'customer_segment_ord', 'merchant_category_freq', 'is_alerted'
]

corr_matrix = encoded_df[numeric_features].corr()

plt.figure(figsize=(14, 11))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    vmin=-0.5,
    vmax=1.0,
    linewidths=0.5,
    cbar_kws={'label': 'Pearson Correlation'}
)
plt.title("Correlation Matrix of Numeric & Engineered Risk Features", fontweight='bold', fontsize=13, pad=12)
plt.tight_layout()
corr_path = 'eda_plots/step5_correlation_matrix.png'
plt.savefig(corr_path, dpi=300)
plt.close()
print(f" [Saved] Correlation Heatmap: {corr_path}")

print("\n>>> STEP 5.2: Feature Importance Evaluation (Random Forest & Mutual Information)...")
# Predictor candidate columns (excluding target and raw string IDs)
feature_cols = [
    'amount', 'tenure_months', 'hour', 'is_weekend', 'is_night_txn',
    'new_device_flag', 'amount_to_segment_median_ratio', 'high_amount_flag',
    'new_device_high_amount', 'cross_region_flag', 'is_iqr_outlier',
    'customer_segment_ord', 'merchant_category_freq'
] + [c for c in encoded_df.columns if c.startswith('ch_') or c.startswith('dev_')]

X = encoded_df[feature_cols].copy()
y = encoded_df['is_alerted'].copy()

# Fit Random Forest Classifier
rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, class_weight='balanced')
rf.fit(X, y)
rf_importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

# Mutual Information
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_series = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)

feat_imp_df = pd.DataFrame({
    'Feature': rf_importances.index,
    'RF_Importance': rf_importances.values,
    'Mutual_Info': [mi_series[f] for f in rf_importances.index]
})

print("\n--- TOP 10 MOST PREDICTIVE FEATURES (RANDOM FOREST) ---")
print(feat_imp_df.head(10).to_string(index=False))

# Plot Feature Importances
plt.figure(figsize=(12, 7))
sns.barplot(
    data=feat_imp_df.head(10),
    x='RF_Importance',
    y='Feature',
    hue='Feature',
    legend=False,
    palette='viridis',
    edgecolor='black'
)
plt.title("Top 10 Most Predictive Features for Operational Alerts (Random Forest Gini Importance)", fontweight='bold', fontsize=13, pad=12)
plt.xlabel("Mean Decrease in Impurity (Feature Importance)", fontweight='bold')
plt.ylabel("Engineered Feature", fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
feat_imp_path = 'eda_plots/step5_top10_feature_importances.png'
plt.savefig(feat_imp_path, dpi=300)
plt.close()
print(f" [Saved] Top 10 Feature Importance Plot: {feat_imp_path}")

print("\n>>> STEP 5.3: Dimensionality Reduction via PCA (2 Components)...")
# Select continuous and encoded features for PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

encoded_df['PC1'] = X_pca[:, 0]
encoded_df['PC2'] = X_pca[:, 1]

print(f"PCA Explained Variance Ratio: PC1 = {pca.explained_variance_ratio_[0]*100:.2f}%, PC2 = {pca.explained_variance_ratio_[1]*100:.2f}% (Total = {np.sum(pca.explained_variance_ratio_)*100:.2f}%)")

# 2D Scatter Plot of Normal vs. Alerted Transactions
plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(
    data=encoded_df,
    x='PC1',
    y='PC2',
    hue='is_alerted',
    palette={0: '#3498db', 1: '#e74c3c'},
    alpha=0.6,
    s=40,
    edgecolor='none'
)
plt.title(f"Principal Component Analysis (PCA): Normal vs. Alerted Transactions\nPC1 ({pca.explained_variance_ratio_[0]*100:.1f}%) & PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontweight='bold', fontsize=13, pad=12)
plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)", fontweight='bold')
plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)", fontweight='bold')
plt.legend(title="Transaction Status", labels=['Normal (0)', 'Alerted (1)'], loc='upper right')
plt.tight_layout()
pca_path = 'eda_plots/step5_pca_2d_projection.png'
plt.savefig(pca_path, dpi=300)
plt.close()
print(f" [Saved] PCA 2D Scatter Plot: {pca_path}")

# ==============================================================================
# STEP 6: DELIVERABLES EXPORT
# ==============================================================================
print("\n>>> STEP 6: Exporting Cleaned Master Dataset for Excel Round 3...")

# Prepare user-friendly master table ready for Excel Pivot Tables
export_cols = [
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
export_filename = "cleaned_paysphere_transactions.csv"
df_export.to_csv(export_filename, index=False)
print(f"Successfully saved {export_filename} with {df_export.shape[0]} rows and {df_export.shape[1]} columns!")

print("\n" + "="*80)
print("ROUND 2 PIPELINE EXECUTION COMPLETE!")
print("="*80)
