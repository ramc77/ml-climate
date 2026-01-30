#!/usr/bin/env python3
"""
================================================================================
SCRIPT 02: REPROCESSING
================================================================================
Based on: arxiv 2508.07062 (Leakage-proof preprocessing)

By: Dr. Ram Chand (BNBWU)
Project: Sindh Climate Hotspot Detection (SHEC Funded) for the year 2024-2026
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================

PROCESSED_DIR = Path('processed_data')
DATA_DIR = Path('data')
RESULTS_DIR = Path('analysis_results')

for d in [PROCESSED_DIR, RESULTS_DIR]:
    d.mkdir(exist_ok=True)


DISTRICT_CATEGORIES = {
    'Coastal': ['Karachi', 'Thatta', 'Badin', 'Sujawal'],
    'Urban_Major': ['Karachi', 'Hyderabad', 'Sukkur', 'Larkana'],
    'Desert': ['Tharparkar', 'Umerkot', 'Sanghar'],
    'Northern_Hot': ['Jacobabad', 'Kashmore', 'Shikarpur', 'Ghotki', 'Sukkur'],
    'Agricultural': ['Nawabshah', 'Khairpur', 'Sanghar', 'Mirpurkhas', 'Mirpur Khas'],
    'River_Delta': ['Thatta', 'Sujawal', 'Badin', 'Tando Muhammad Khan'],
}


NAME_MAPPING = {
    'Mirpur Khas': 'Mirpurkhas',
    'Mirpur khas': 'Mirpurkhas',
    'mirpur khas': 'Mirpurkhas',
}


BASELINE_START = 1981
BASELINE_END = 2010

# =============================================================================
# DATA LOADING AND CLEANING
# =============================================================================

def load_and_clean_data():
    """Load and clean all data sources."""
    print("\n" + "=" * 70)
    print("LOADING AND CLEANING DATA (V4)")
    print("=" * 70)
    
    
    complete_path = PROCESSED_DIR / 'yearly_data_complete_22x44.csv'
    basic_path = PROCESSED_DIR / 'yearly_data.csv'

    if complete_path.exists():
        yearly = pd.read_csv(complete_path)
        print(f"  Loaded yearly_data_complete_22x44.csv: {len(yearly)} rows")
    else:
        yearly = pd.read_csv(basic_path)
        print(f"  Loaded yearly_data.csv: {len(yearly)} rows")
    
   
    yearly['district'] = yearly['district'].replace(NAME_MAPPING)
    print(f"  Standardized district names")
    
    
    if 'Mirpurkhas' in yearly['district'].values:
        print(f"  ✓ Mirpurkhas found in data")
    else:
        print(f"  ✗ WARNING: Mirpurkhas not found!")
    
    return yearly

def add_karachi_specific_features(df):
    """Add Karachi-specific features."""
    print("\n  Adding Karachi-specific features...")
    
    
    df['is_coastal'] = df['district'].isin(DISTRICT_CATEGORIES['Coastal']).astype(int)
    
   
    df['is_major_urban'] = df['district'].isin(DISTRICT_CATEGORIES['Urban_Major']).astype(int)
    
   
    df['is_desert'] = df['district'].isin(DISTRICT_CATEGORIES['Desert']).astype(int)
    
   
    df['is_northern_hot'] = df['district'].isin(DISTRICT_CATEGORIES['Northern_Hot']).astype(int)
    
   
    df['is_karachi'] = (df['district'] == 'Karachi').astype(int)
    
    
    coastal_districts = DISTRICT_CATEGORIES['Coastal']
    df['coastal_proximity'] = df['district'].apply(
        lambda x: 1.0 if x == 'Karachi' else 
                  0.8 if x in ['Thatta', 'Badin'] else
                  0.6 if x in ['Sujawal', 'Tando Muhammad Khan'] else
                  0.3 if x in ['Hyderabad', 'Jamshoro'] else 0.0
    )
    
    
    if 'pop_density' in df.columns:
        df['urban_heat_proxy'] = df['pop_density'] * df['is_major_urban']
    
    print(f"    Added: is_coastal, is_major_urban, is_desert, is_northern_hot")
    print(f"    Added: is_karachi, coastal_proximity, urban_heat_proxy")
    
    return df

def add_district_cluster_features(df):
    """Add district clustering features."""
    print("\n  Adding district cluster features...")
    
   
    lat_mapping = {
        'Kashmore': 28.43, 'Ghotki': 28.00, 'Jacobabad': 28.28, 'Shikarpur': 27.96,
        'Sukkur': 27.71, 'Larkana': 27.56, 'Khairpur': 27.53, 'Nawabshah': 26.25,
        'Dadu': 26.73, 'Sanghar': 26.05, 'Umerkot': 25.36, 'Mirpurkhas': 25.53,
        'Hyderabad': 25.40, 'Jamshoro': 25.43, 'Matiari': 25.59, 
        'Tando Allahyar': 25.46, 'Tando Muhammad Khan': 25.12,
        'Karachi': 24.86, 'Thatta': 24.75, 'Badin': 24.63, 'Tharparkar': 24.74,
        'Sujawal': 24.56
    }
    
    df['latitude'] = df['district'].map(lat_mapping)
    
   
    lon_mapping = {
        'Karachi': 67.00, 'Thatta': 67.92, 'Dadu': 67.78, 'Jamshoro': 68.28,
        'Hyderabad': 68.36, 'Matiari': 68.45, 'Nawabshah': 68.40, 'Larkana': 68.21,
        'Jacobabad': 68.44, 'Shikarpur': 68.64, 'Sukkur': 68.86, 'Khairpur': 68.76,
        'Ghotki': 69.32, 'Kashmore': 69.58, 'Sanghar': 68.95, 'Mirpurkhas': 69.01,
        'Umerkot': 69.74, 'Tharparkar': 70.13, 'Badin': 68.84, 'Sujawal': 68.02,
        'Tando Allahyar': 68.72, 'Tando Muhammad Khan': 68.53
    }
    
    df['longitude'] = df['district'].map(lon_mapping)
    
    
    df['lat_temp_adjust'] = (df['latitude'] - 26.0) * 0.5  # Higher lat = hotter in Sindh
    
    print(f"    Added: latitude, longitude, lat_temp_adjust")
    
    return df

def calculate_improved_anomalies(df):
    """Calculate temperature anomalies with district-specific baselines."""
    print("\n  Calculating improved anomalies...")
    
   
    baseline_mask = (df['year'] >= BASELINE_START) & (df['year'] <= BASELINE_END)
    
    district_baselines = df[baseline_mask].groupby('district')['T2M_mean'].mean()
    df['district_baseline'] = df['district'].map(district_baselines)
    
   
    df['temp_anomaly'] = df['T2M_mean'] - df['district_baseline']
    
   
    global_baseline = df[baseline_mask]['T2M_mean'].mean()
    df['temp_anomaly_global'] = df['T2M_mean'] - global_baseline
    
    print(f"    District baselines calculated for {len(district_baselines)} districts")
    print(f"    Global baseline: {global_baseline:.2f}°C")
    
    return df

def create_temporal_features(df):
    """Create advanced temporal features."""
    print("\n  Creating temporal features...")
    
  
    df['decade'] = ((df['year'] - 1980) // 10).astype(int)
    
   
    df['years_since_baseline'] = np.maximum(0, df['year'] - BASELINE_END)
    
   
    df['year_squared'] = (df['year'] - 2000) ** 2 / 100
    
   
    df['period_1980s'] = ((df['year'] >= 1981) & (df['year'] < 1990)).astype(int)
    df['period_1990s'] = ((df['year'] >= 1990) & (df['year'] < 2000)).astype(int)
    df['period_2000s'] = ((df['year'] >= 2000) & (df['year'] < 2010)).astype(int)
    df['period_2010s'] = ((df['year'] >= 2010) & (df['year'] < 2020)).astype(int)
    df['period_2020s'] = (df['year'] >= 2020).astype(int)
    
    print(f"    Added: decade, years_since_baseline, year_squared, period indicators")
    
    return df

def create_interaction_features(df):
    """Create interaction features."""
    print("\n  Creating interaction features...")
    
    
    df['coastal_year'] = df['is_coastal'] * df['years_since_baseline']
    
   
    if 'population' in df.columns:
        df['urban_pop_interaction'] = df['is_major_urban'] * np.log1p(df['population'])
    
   
    if 'ndvi_mean' in df.columns:
        df['ndvi_temp_feedback'] = df['ndvi_mean'] * df.get('lat_temp_adjust', 0)
    
   
    df['northern_year'] = df['is_northern_hot'] * df['years_since_baseline']
    
    print(f"    Added: coastal_year, urban_pop_interaction, ndvi_temp_feedback, northern_year")
    
    return df

# =============================================================================
# TRAIN/TEST
# =============================================================================

def create_improved_splits(df):
    """Create improved train/validation/test splits."""
    print("\n" + "=" * 70)
    print("CREATING IMPROVED DATA SPLITS")
    print("=" * 70)
    
    
    train = df[df['year'] <= 2015].copy()
    val = df[(df['year'] >= 2016) & (df['year'] <= 2019)].copy()
    test = df[df['year'] >= 2020].copy()
    
    print(f"\n  Train: {train['year'].min()}-{train['year'].max()} ({len(train)} samples)")
    print(f"  Validation: {val['year'].min()}-{val['year'].max()} ({len(val)} samples)")
    print(f"  Test: {test['year'].min()}-{test['year'].max()} ({len(test)} samples)")
    

    print(f"\n  Target distribution:")
    print(f"    Train anomaly: {train['temp_anomaly'].mean():.3f} ± {train['temp_anomaly'].std():.3f}")
    print(f"    Val anomaly:   {val['temp_anomaly'].mean():.3f} ± {val['temp_anomaly'].std():.3f}")
    print(f"    Test anomaly:  {test['temp_anomaly'].mean():.3f} ± {test['temp_anomaly'].std():.3f}")
    
    return train, val, test

def prepare_features(train, val, test):
    """Prepare features with proper scaling."""
    print("\n  Preparing features...")
    
    numeric_features = [
        'year', 'years_since_baseline', 'year_squared',
        'latitude', 'longitude', 'lat_temp_adjust',
        'coastal_proximity', 'urban_heat_proxy',
        'coastal_year', 'northern_year',
    ]
    
    lag_features = [c for c in train.columns if 'lag' in c.lower() and 'norm' not in c.lower()]
    numeric_features.extend(lag_features)
    
    original_features = ['population', 'pop_density', 'urban_pct', 'co2_emissions_kt',
                        'national_co2_mt', 'ndvi_mean', 'PRECTOTCORR_sum', 'PRECTOTCORR_mean']
    numeric_features.extend([f for f in original_features if f in train.columns])


    new_human_activity = ['nightlights', 'nightlights_change', 'co2_per_capita_kg',
                          'cumulative_floods']
    numeric_features.extend([f for f in new_human_activity if f in train.columns])
    
   
    numeric_features = [f for f in numeric_features if f in train.columns]
    
    binary_features = ['is_coastal', 'is_major_urban', 'is_desert', 'is_northern_hot',
                       'is_karachi', 'period_1980s', 'period_1990s', 'period_2000s',
                       'period_2010s', 'period_2020s', 'flood_occurred']
    binary_features = [f for f in binary_features if f in train.columns]
    
    print(f"    Numeric features: {len(numeric_features)}")
    print(f"    Binary features: {len(binary_features)}")
    
  
    scaler = StandardScaler()
    
    train_scaled = train.copy()
    val_scaled = val.copy()
    test_scaled = test.copy()
    
   
    scaler.fit(train[numeric_features])
    
   
    for col in numeric_features:
        col_idx = numeric_features.index(col)
        train_scaled[f'{col}_scaled'] = scaler.transform(train[numeric_features])[:, col_idx]
        val_scaled[f'{col}_scaled'] = scaler.transform(val[numeric_features])[:, col_idx]
        test_scaled[f'{col}_scaled'] = scaler.transform(test[numeric_features])[:, col_idx]
    
   
    feature_cols = [f'{c}_scaled' for c in numeric_features] + binary_features
    
    print(f"    Total features: {len(feature_cols)}")
    
    return train_scaled, val_scaled, test_scaled, feature_cols, scaler

# =============================================================================
# PREPROCESSING
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("PREPROCESSING ...")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    df = load_and_clean_data()
    
    df = add_karachi_specific_features(df)
    df = add_district_cluster_features(df)
    df = calculate_improved_anomalies(df)
    df = create_temporal_features(df)
    df = create_interaction_features(df)
    
    train, val, test = create_improved_splits(df)
    
    train_scaled, val_scaled, test_scaled, feature_cols, scaler = prepare_features(train, val, test)
    
    print("\n" + "=" * 70)
    print("SAVING PROCESSED DATA")
    print("=" * 70)
    
    train_scaled.to_csv(PROCESSED_DIR / 'train.csv', index=False)
    val_scaled.to_csv(PROCESSED_DIR / 'val.csv', index=False)
    test_scaled.to_csv(PROCESSED_DIR / 'test.csv', index=False)
    
    pd.DataFrame({'feature': feature_cols}).to_csv(PROCESSED_DIR / 'features.csv', index=False)
    
    df.to_csv(PROCESSED_DIR / 'full_processed.csv', index=False)
    
    print(f"  ✓ train.csv: {len(train_scaled)} rows")
    print(f"  ✓ val.csv: {len(val_scaled)} rows")
    print(f"  ✓ test.csv: {len(test_scaled)} rows")
    print(f"  ✓ features.csv: {len(feature_cols)} features")
    print(f"  ✓ full_processed.csv: {len(df)} rows")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("PREPROCESSING SUMMARY")
    print("=" * 70)
    print(f"""
  Data Coverage:
    Years: {df['year'].min()}-{df['year'].max()} ({df['year'].nunique()} years)
    Districts: {df['district'].nunique()} districts
    Total observations: {len(df)}
    
  Karachi-Specific Features Added:
    is_coastal, is_major_urban, is_karachi
    coastal_proximity, urban_heat_proxy
    coastal_year (interaction)
    
  District Clustering:
    Coastal: {len(DISTRICT_CATEGORIES['Coastal'])} districts
    Urban Major: {len(DISTRICT_CATEGORIES['Urban_Major'])} districts
    Desert: {len(DISTRICT_CATEGORIES['Desert'])} districts
    Northern Hot: {len(DISTRICT_CATEGORIES['Northern_Hot'])} districts
    
  Split Strategy:
    Train: 1981-2015 (before distribution shift)
    Validation: 2016-2019 (bridge period)
    Test: 2020-2024 (COVID-affected period)
    """)
    
    print("=" * 70)
    print("✅ PREPROCESSING COMPLETE")
    print("=" * 70)
    
    return df, train_scaled, val_scaled, test_scaled, feature_cols

if __name__ == "__main__":
    main()
