"""
generate_datasets.py
PaySphere FinTech Pvt. Ltd. (Assessment Code: U23AD491)
Use Case 13: Digital Payment Insight
Dataset Generation adhering strictly to schema and intentional data quality conditions.
"""

import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

def generate_paysphere_data(n_transactions=12000, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    
    # ---------------------------------------------------------
    # 1. Customers Table (customers.csv)
    # customer_id, customer_segment, region, tenure_months
    # ---------------------------------------------------------
    n_customers = 1500
    customer_ids = [f"CUST_{i:05d}" for i in range(1, n_customers + 1)]
    segments = ['Standard', 'Premium', 'Corporate', 'SMB']
    segment_probs = [0.55, 0.20, 0.10, 0.15]
    regions = ['North', 'South', 'East', 'West', 'Central']
    region_probs = [0.28, 0.24, 0.16, 0.22, 0.10]
    
    cust_segments = np.random.choice(segments, size=n_customers, p=segment_probs)
    cust_regions = np.random.choice(regions, size=n_customers, p=region_probs)
    # Tenure in months: 1 to 72 months
    cust_tenure = np.random.randint(1, 73, size=n_customers)
    
    df_customers = pd.DataFrame({
        'customer_id': customer_ids,
        'customer_segment': cust_segments,
        'region': cust_regions,
        'tenure_months': cust_tenure
    })
    
    # ---------------------------------------------------------
    # 2. Merchants Table (merchants.csv)
    # merchant_id, merchant_category, region
    # ---------------------------------------------------------
    n_merchants = 350
    merchant_ids = [f"MERCH_{i:04d}" for i in range(1, n_merchants + 1)]
    categories = [
        'Grocery', 'Dining', 'Electronics', 'Travel & Hospitality',
        'Jewellery', 'Gaming & Digital Goods', 'Utilities', 'Healthcare'
    ]
    merch_cat_probs = [0.25, 0.20, 0.12, 0.10, 0.05, 0.08, 0.12, 0.08]
    
    merch_categories = np.random.choice(categories, size=n_merchants, p=merch_cat_probs)
    merch_regions = np.random.choice(regions, size=n_merchants, p=region_probs)
    
    df_merchants = pd.DataFrame({
        'merchant_id': merchant_ids,
        'merchant_category': merch_categories,
        'region': merch_regions
    })
    
    # ---------------------------------------------------------
    # 3. Devices Table (devices.csv)
    # device_id, device_type, new_device_flag
    # Incorporate defect: Incomplete device history (null device types, missing flags)
    # ---------------------------------------------------------
    n_devices = 2800
    device_ids = [f"DEV_{i:05d}" for i in range(1, n_devices + 1)]
    device_types = ['Android', 'iOS', 'Web_Browser', 'POS_Terminal']
    device_type_probs = [0.52, 0.28, 0.14, 0.06]
    
    dev_types = np.random.choice(device_types, size=n_devices, p=device_type_probs).astype(object)
    # new_device_flag: 1 for newly registered device (last 14 days), 0 for known
    new_flags = np.random.choice([0, 1], size=n_devices, p=[0.78, 0.22]).astype(object)
    
    # Inject missing device types (~6%) and missing new_device_flags (~4%)
    missing_type_idx = np.random.choice(n_devices, size=int(n_devices * 0.06), replace=False)
    dev_types[missing_type_idx] = np.nan
    
    missing_flag_idx = np.random.choice(n_devices, size=int(n_devices * 0.04), replace=False)
    new_flags[missing_flag_idx] = np.nan
    
    df_devices = pd.DataFrame({
        'device_id': device_ids,
        'device_type': dev_types,
        'new_device_flag': new_flags
    })
    
    # ---------------------------------------------------------
    # 4. Transactions Table (transactions.csv)
    # transaction_id, customer_id, merchant_id, device_id, time, amount, merchant_category, channel
    # Incorporate defects:
    # - Duplicate transaction IDs
    # - Inconsistent channel labels (casing, whitespaces)
    # - Missing merchant categories in transactions
    # - Extreme amounts (fat-tailed/positive skew + clear fraud-like outliers)
    # ---------------------------------------------------------
    # Generate base unique transactions
    n_unique_txns = n_transactions - 180 # 180 duplicates will be added
    txn_ids = [f"TXN_{i:07d}" for i in range(1, n_unique_txns + 1)]
    
    tx_cust_ids = np.random.choice(customer_ids, size=n_unique_txns)
    tx_merch_ids = np.random.choice(merchant_ids, size=n_unique_txns)
    tx_dev_ids = np.random.choice(device_ids, size=n_unique_txns)
    
    # Timestamp generation over 45 days
    base_date = datetime(2026, 8, 1, 0, 0, 0)
    
    # Hours with circadian cycle: standard daytime volume + late-night spike anomalies (2 AM - 4 AM)
    # 0 to 23 hours
    hour_weights = [
        0.015, 0.010, 0.025, 0.030, 0.015, 0.010, # 0-5 night/early (unusual burst at 2-3 AM from automated/fraud bots)
        0.020, 0.035, 0.055, 0.070, 0.075, 0.080, # 6-11 morning
        0.085, 0.080, 0.070, 0.065, 0.070, 0.075, # 12-17 afternoon
        0.085, 0.080, 0.065, 0.045, 0.030, 0.020  # 18-23 evening
    ]
    hour_weights = np.array(hour_weights) / np.sum(hour_weights)
    
    days_offset = np.random.randint(0, 45, size=n_unique_txns)
    hours = np.random.choice(24, size=n_unique_txns, p=hour_weights)
    minutes = np.random.randint(0, 60, size=n_unique_txns)
    seconds = np.random.randint(0, 60, size=n_unique_txns)
    
    tx_times = [
        (base_date + timedelta(days=int(d), hours=int(h), minutes=int(m), seconds=int(s))).strftime("%Y-%m-%d %H:%M:%S")
        for d, h, m, s in zip(days_offset, hours, minutes, seconds)
    ]
    
    # Merchant category lookup map
    merch_map = dict(zip(df_merchants['merchant_id'], df_merchants['merchant_category']))
    tx_merch_cats = [merch_map[mid] for mid in tx_merch_ids]
    
    # Customer segment lookup map
    cust_seg_map = dict(zip(df_customers['customer_id'], df_customers['customer_segment']))
    tx_segs = [cust_seg_map[cid] for cid in tx_cust_ids]
    
    # Amount generation: Segment-based lognormal distribution + extreme outliers
    # Standard: median ~450, IQR 200-1200
    # Premium: median ~1800, IQR 800-4500
    # SMB: median ~3200, IQR 1500-8000
    # Corporate: median ~14500, IQR 6000-35000
    amounts = np.zeros(n_unique_txns)
    for i, seg in enumerate(tx_segs):
        if seg == 'Standard':
            amt = np.random.lognormal(mean=6.1, sigma=0.85) # median ~445
        elif seg == 'Premium':
            amt = np.random.lognormal(mean=7.5, sigma=0.80) # median ~1800
        elif seg == 'SMB':
            amt = np.random.lognormal(mean=8.0, sigma=0.85) # median ~2980
        else: # Corporate
            amt = np.random.lognormal(mean=9.5, sigma=0.90) # median ~13350
        amounts[i] = round(amt, 2)
        
    # Inject extreme amounts (fat-tailed outliers / fraud-like amounts)
    # E.g. Standard users making 35,000 to 120,000 transactions, or Corporate making 350,000+
    extreme_outlier_idx = np.random.choice(n_unique_txns, size=150, replace=False)
    for idx in extreme_outlier_idx:
        seg = tx_segs[idx]
        if seg == 'Standard':
            amounts[idx] = round(np.random.uniform(25000, 95000), 2)
        elif seg == 'Premium':
            amounts[idx] = round(np.random.uniform(60000, 180000), 2)
        else:
            amounts[idx] = round(np.random.uniform(150000, 450000), 2)
            
    # Channels with intentional messy casing and whitespaces
    # Base channels: 'UPI', 'NetBanking', 'Credit Card', 'Debit Card', 'Wallet'
    channel_variants = {
        'UPI': ['UPI', 'upi', ' UPI ', 'Upi', '  upi'],
        'NetBanking': ['NetBanking', 'NETBANKING', 'netbanking', ' NetBanking ', 'Netbanking '],
        'Credit Card': ['Credit Card', 'CREDIT CARD', 'credit card', ' Credit Card ', 'credit_card'],
        'Debit Card': ['Debit Card', 'DEBIT CARD', 'debit card', ' Debit Card '],
        'Wallet': ['Wallet', 'WALLET', 'wallet', ' Wallet ']
    }
    channel_keys = ['UPI', 'NetBanking', 'Credit Card', 'Debit Card', 'Wallet']
    channel_key_probs = [0.46, 0.20, 0.18, 0.10, 0.06]
    
    assigned_channels = np.random.choice(channel_keys, size=n_unique_txns, p=channel_key_probs)
    messy_channels = [random.choice(channel_variants[k]) for k in assigned_channels]
    
    # Defect: Missing merchant categories in transactions (~7.5% missing)
    tx_merch_cats_with_nulls = list(tx_merch_cats)
    missing_cat_idx = np.random.choice(n_unique_txns, size=int(n_unique_txns * 0.075), replace=False)
    for idx in missing_cat_idx:
        tx_merch_cats_with_nulls[idx] = np.nan
        
    df_txns = pd.DataFrame({
        'transaction_id': txn_ids,
        'customer_id': tx_cust_ids,
        'merchant_id': tx_merch_ids,
        'device_id': tx_dev_ids,
        'time': tx_times,
        'amount': amounts,
        'merchant_category': tx_merch_cats_with_nulls,
        'channel': messy_channels
    })
    
    # Defect: Duplicate transaction IDs (180 duplicates appended)
    dup_rows = df_txns.sample(n=180, random_state=seed).copy()
    df_txns = pd.concat([df_txns, dup_rows], ignore_index=True)
    # Shuffle slightly so duplicates aren't just at the end
    df_txns = df_txns.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    # ---------------------------------------------------------
    # 5. Alerts Table (alerts.csv)
    # transaction_id, review_status, reason_category
    # Alerts occur disproportionately on:
    # - New devices
    # - Extreme amounts
    # - Late-night hours (0 to 5)
    # - High-risk categories: Jewellery, Gaming, Electronics
    # - Cross-region mismatch
    # ---------------------------------------------------------
    dev_new_map = dict(zip(df_devices['device_id'], df_devices['new_device_flag']))
    cust_reg_map = dict(zip(df_customers['customer_id'], df_customers['region']))
    merch_reg_map = dict(zip(df_merchants['merchant_id'], df_merchants['region']))
    
    alert_candidates = []
    
    # Unique transactions without duplicates for alert generation
    unique_txns_df = df_txns.drop_duplicates(subset=['transaction_id'])
    
    for _, row in unique_txns_df.iterrows():
        tid = row['transaction_id']
        cid = row['customer_id']
        mid = row['merchant_id']
        did = row['device_id']
        amt = row['amount']
        cat = merch_map.get(mid, 'Unknown')
        
        # parse hour
        hr = int(row['time'].split(' ')[1].split(':')[0])
        is_night = (hr in [0, 1, 2, 3, 4, 5])
        is_new_dev = (dev_new_map.get(did, 0) == 1)
        is_cross_region = (cust_reg_map.get(cid) != merch_reg_map.get(mid))
        is_extreme_amt = (amt > 15000)
        
        # Base risk score
        risk_score = 0.015 # baseline 1.5%
        if is_new_dev:
            risk_score += 0.08
        if is_night:
            risk_score += 0.07
        if is_extreme_amt:
            risk_score += 0.35
        if cat in ['Jewellery', 'Gaming & Digital Goods']:
            risk_score += 0.06
        if is_cross_region and is_new_dev:
            risk_score += 0.12
            
        # Probability threshold
        if np.random.rand() < min(risk_score, 0.85):
            # Select dominant reason
            if is_extreme_amt:
                reason = 'High Amount Anomaly'
            elif is_new_dev and is_cross_region:
                reason = 'Cross-Region Surge'
            elif is_new_dev:
                reason = 'New Device Spike'
            elif is_night:
                reason = 'Unusual Hour Velocity'
            elif cat in ['Jewellery', 'Gaming & Digital Goods']:
                reason = 'High-Risk Merchant Category'
            else:
                reason = 'Velocity Threshold Exceeded'
                
            # Assign review status
            # 'Approved', 'Rejected', 'Under Review', 'Escalated'
            if reason == 'High Amount Anomaly':
                status = np.random.choice(['Rejected', 'Escalated', 'Approved', 'Under Review'], p=[0.40, 0.30, 0.15, 0.15])
            elif reason in ['New Device Spike', 'Cross-Region Surge']:
                status = np.random.choice(['Under Review', 'Approved', 'Rejected', 'Escalated'], p=[0.35, 0.35, 0.20, 0.10])
            else:
                status = np.random.choice(['Approved', 'Under Review', 'Rejected', 'Escalated'], p=[0.45, 0.30, 0.15, 0.10])
                
            alert_candidates.append({
                'transaction_id': tid,
                'review_status': status,
                'reason_category': reason
            })
            
    df_alerts = pd.DataFrame(alert_candidates)
    
    return df_transactions_raw(df_txns), df_customers, df_devices, df_merchants, df_alerts

def df_transactions_raw(df):
    return df

if __name__ == '__main__':
    df_txns, df_cust, df_dev, df_merch, df_alerts = generate_paysphere_data()
    
    df_txns.to_csv("transactions.csv", index=False)
    df_cust.to_csv("customers.csv", index=False)
    df_dev.to_csv("devices.csv", index=False)
    df_merch.to_csv("merchants.csv", index=False)
    df_alerts.to_csv("alerts.csv", index=False)
    
    print("Synthetic datasets generated successfully!")
    print(f"Transactions: {df_txns.shape}")
    print(f"Customers:    {df_cust.shape}")
    print(f"Devices:      {df_dev.shape}")
    print(f"Merchants:    {df_merch.shape}")
    print(f"Alerts:       {df_alerts.shape}")
