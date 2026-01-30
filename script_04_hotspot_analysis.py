#!/usr/bin/env python3
"""
================================================================================
SCRIPT 04: HOTSPOT ANALYSIS
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


PROCESSED_DIR = Path('processed_data')
RESULTS_DIR = Path('analysis_results')

PARIS_1_5_THRESHOLD = 1.5
PARIS_2_0_THRESHOLD = 2.0
BASELINE_END = 2010
CURRENT_YEAR = 2024

SINDH_DISTRICTS = [
    "Karachi", "Hyderabad", "Sukkur", "Larkana", "Nawabshah",
    "Mirpurkhas", "Thatta", "Badin", "Tharparkar", "Umerkot",
    "Sanghar", "Khairpur", "Ghotki", "Jacobabad", "Shikarpur",
    "Kashmore", "Dadu", "Jamshoro", "Matiari", "Tando Allahyar",
    "Tando Muhammad Khan", "Sujawal"
]

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load all required data."""
    print("\n" + "=" * 70)
    print("LOADING DATA FOR HOTSPOT ANALYSIS")
    print("=" * 70)
    
    data = {}
    
    data['lodo'] = pd.read_csv(RESULTS_DIR / 'lodo_cv.csv')
    data['ml_results'] = pd.read_csv(RESULTS_DIR / 'ml_results.csv')
    data['predictions'] = pd.read_csv(RESULTS_DIR / 'predictions.csv')
    data['attribution'] = pd.read_csv(RESULTS_DIR / 'attribution_importance.csv')
    
    data['full'] = pd.read_csv(PROCESSED_DIR / 'full_processed.csv')
    
    for name, df in data.items():
        print(f"  ✓ {name}: {len(df)} rows")
    
    return data


def calculate_climate_trends(data):
    """Calculate climate trends for each district."""
    print("\n" + "=" * 70)
    print("CALCULATING CLIMATE TRENDS")
    print("=" * 70)
    
    full = data['full']
    
    trends = []
    for district in full['district'].unique():
        d_data = full[full['district'] == district].sort_values('year')
        
        if len(d_data) < 10:
            continue
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            d_data['year'], d_data['T2M_mean']
        )
        
        baseline = d_data[d_data['year'] <= BASELINE_END]['T2M_mean'].mean()
        recent = d_data[d_data['year'] >= 2015]['T2M_mean'].mean()
        
        trends.append({
            'district': district,
            'baseline_temp': baseline,
            'recent_temp': recent,
            'warming_since_baseline': recent - baseline,
            'trend_per_decade': slope * 10,
            'trend_p_value': p_value,
            'trend_r2': r_value ** 2,
            'trend_significant': p_value < 0.05,
            'n_years': len(d_data),
            'temp_mean': d_data['T2M_mean'].mean(),
            'temp_std': d_data['T2M_mean'].std(),
            'temp_min': d_data['T2M_mean'].min(),
            'temp_max': d_data['T2M_mean'].max(),
        })
    
    trends_df = pd.DataFrame(trends)
    
    print(f"\n  Districts analyzed: {len(trends_df)}")
    print(f"  Mean warming: {trends_df['warming_since_baseline'].mean():.3f}°C")
    print(f"  Trend range: {trends_df['trend_per_decade'].min():.3f} to {trends_df['trend_per_decade'].max():.3f}°C/decade")
    
    return trends_df

# =============================================================================
# HOTSPOT SCORE
# =============================================================================

def calculate_hotspot_scores(data, trends_df):
    """Calculate hotspot scores."""
    print("\n" + "=" * 70)
    print("CALCULATING HOTSPOT SCORES")
    print("=" * 70)
    
    lodo = data['lodo']
    
   
    hotspot = trends_df.merge(lodo[['district', 'r2', 'rmse']], on='district', how='left')
    hotspot = hotspot.rename(columns={'r2': 'model_r2', 'rmse': 'model_rmse'})
    
   
    hotspot['hazard_index'] = (
        hotspot['warming_since_baseline'].rank(pct=True) * 0.4 +
        hotspot['trend_per_decade'].rank(pct=True) * 0.3 +
        hotspot['temp_max'].rank(pct=True) * 0.3
    )
    
   
    hotspot['exposure_index'] = hotspot['model_rmse'].rank(pct=True)
    
   
    hotspot['vulnerability_index'] = 1 - hotspot['model_r2'].rank(pct=True)
    
    # Combined hotspot score (IPCC framework: Risk = Hazard × Exposure × Vulnerability)
    hotspot['hotspot_score'] = (
        hotspot['hazard_index'] * 0.4 +
        hotspot['exposure_index'] * 0.3 +
        hotspot['vulnerability_index'] * 0.3
    )
    
    hotspot['hotspot_score'] = (hotspot['hotspot_score'] - hotspot['hotspot_score'].min()) / \
                               (hotspot['hotspot_score'].max() - hotspot['hotspot_score'].min())
    
    def categorize(score):
        if score >= 0.8:
            return 'Very High'
        elif score >= 0.6:
            return 'High'
        elif score >= 0.4:
            return 'Moderate'
        else:
            return 'Low'
    
    hotspot['risk_category'] = hotspot['hotspot_score'].apply(categorize)
    
    
    hotspot['rank'] = hotspot['hotspot_score'].rank(ascending=False).astype(int)
    
   
    hotspot = hotspot.sort_values('hotspot_score', ascending=False)
    
    print(f"\n  Top 5 Hotspots:")
    for _, row in hotspot.head(5).iterrows():
        print(f"    {row['rank']}. {row['district']}: {row['hotspot_score']:.3f} ({row['risk_category']})")
    
    return hotspot

# PARIS AGREEMENT 
# =============================================================================

def calculate_paris_compliance(hotspot):
    """Calculate Paris Agreement."""
    print("\n" + "=" * 70)
    print("CALCULATING PARIS AGREEMENT")
    print("=" * 70)
    
    paris = hotspot.copy()
    
    paris['warming_rate'] = paris['trend_per_decade'] / 10
    
    remaining_to_1_5 = PARIS_1_5_THRESHOLD - paris['warming_since_baseline']
    paris['years_to_1_5C'] = np.where(
        paris['warming_rate'] > 0,
        remaining_to_1_5 / paris['warming_rate'],
        np.inf
    )
    paris['years_to_1_5C'] = paris['years_to_1_5C'].replace([np.inf, -np.inf], 999)
    
    remaining_to_2_0 = PARIS_2_0_THRESHOLD - paris['warming_since_baseline']
    paris['years_to_2_0C'] = np.where(
        paris['warming_rate'] > 0,
        remaining_to_2_0 / paris['warming_rate'],
        np.inf
    )
    paris['years_to_2_0C'] = paris['years_to_2_0C'].replace([np.inf, -np.inf], 999)
    
    def get_status(row):
        if row['warming_since_baseline'] >= PARIS_2_0_THRESHOLD:
            return 'Exceeded 2.0°C'
        elif row['warming_since_baseline'] >= PARIS_1_5_THRESHOLD:
            return 'Exceeded 1.5°C'
        elif row['years_to_1_5C'] < 10:
            return 'Critical (<10 years)'
        elif row['years_to_1_5C'] < 30:
            return 'At Risk (<30 years)'
        else:
            return 'On Track'
    
    paris['compliance_status'] = paris.apply(get_status, axis=1)
    
    for year in [2030, 2040, 2050]:
        years_ahead = year - CURRENT_YEAR
        paris[f'projected_{year}'] = paris['warming_since_baseline'] + paris['warming_rate'] * years_ahead
    
    print(f"\n  Compliance Status Distribution:")
    for status in paris['compliance_status'].unique():
        count = (paris['compliance_status'] == status).sum()
        print(f"    {status}: {count} districts")
    
    return paris


def calculate_ipcc_risk(hotspot):
    """Calculate IPCC risk framework metrics."""
    print("\n" + "=" * 70)
    print("CALCULATING IPCC RISK FRAMEWORK")
    print("=" * 70)
    
    ipcc = hotspot.copy()
    
    def get_confidence(r2):
        if r2 >= 0.95:
            return 'Very High'
        elif r2 >= 0.85:
            return 'High'
        elif r2 >= 0.70:
            return 'Medium'
        else:
            return 'Low'
    
    ipcc['model_confidence'] = ipcc['model_r2'].apply(get_confidence)
    
    def get_ipcc_level(score):
        if score >= 0.8:
            return 'Very High Risk'
        elif score >= 0.6:
            return 'High Risk'
        elif score >= 0.4:
            return 'Medium Risk'
        elif score >= 0.2:
            return 'Low Risk'
        else:
            return 'Very Low Risk'
    
    ipcc['ipcc_level'] = ipcc['hotspot_score'].apply(get_ipcc_level)
    

    ipcc['ipcc_score'] = ipcc['hotspot_score'] * ipcc['model_r2']
    
    print(f"\n  IPCC Risk Distribution:")
    for level in ipcc['ipcc_level'].unique():
        count = (ipcc['ipcc_level'] == level).sum()
        print(f"    {level}: {count} districts")
    
    return ipcc

def calculate_projections(hotspot):
    """Calculate climate projections to 2080."""
    print("\n" + "=" * 70)
    print("CALCULATING CLIMATE PROJECTIONS")
    print("=" * 70)
    
    projections = []
    
    for _, row in hotspot.iterrows():
        district = row['district']
        baseline_temp = row['baseline_temp']
        current_anomaly = row['warming_since_baseline']
        trend = row['trend_per_decade'] / 10  # per year
        
        for year in [2030, 2040, 2050, 2060, 2070, 2080]:
            years_ahead = year - CURRENT_YEAR
            projected_anomaly = current_anomaly + trend * years_ahead
            
            uncertainty = row['model_rmse'] * np.sqrt(years_ahead / 10)
            
            projections.append({
                'district': district,
                'year': year,
                'baseline_temp': baseline_temp,
                'projected_anomaly': projected_anomaly,
                'projected_temp': baseline_temp + projected_anomaly,
                'uncertainty_lower': projected_anomaly - 1.96 * uncertainty,
                'uncertainty_upper': projected_anomaly + 1.96 * uncertainty,
                'exceeds_1_5C': projected_anomaly >= PARIS_1_5_THRESHOLD,
                'exceeds_2_0C': projected_anomaly >= PARIS_2_0_THRESHOLD,
            })
    
    proj_df = pd.DataFrame(projections)
    
    print(f"\n  Projections calculated for {proj_df['district'].nunique()} districts")
    print(f"  Years: {proj_df['year'].min()}-{proj_df['year'].max()}")
    
    print(f"\n  Districts exceeding thresholds by year:")
    for year in [2030, 2050, 2080]:
        year_data = proj_df[proj_df['year'] == year]
        n_1_5 = year_data['exceeds_1_5C'].sum()
        n_2_0 = year_data['exceeds_2_0C'].sum()
        print(f"    {year}: {n_1_5} exceed 1.5°C, {n_2_0} exceed 2.0°C")
    
    return proj_df


def calculate_comprehensive_ranking(hotspot, paris, ipcc):
    """Calculate comprehensive ranking for policy prioritization."""
    print("\n" + "=" * 70)
    print("CALCULATING COMPREHENSIVE RANKING")
    print("=" * 70)
    
    ranking = hotspot[['district', 'hotspot_score', 'hazard_index', 
                       'exposure_index', 'vulnerability_index', 'model_r2', 'rank']].copy()
    
    
    paris_cols = paris[['district', 'years_to_1_5C', 'compliance_status']]
    ranking = ranking.merge(paris_cols, on='district')
    
    
    ipcc_cols = ipcc[['district', 'ipcc_level', 'model_confidence']]
    ranking = ranking.merge(ipcc_cols, on='district')
    
   
    def get_priority(row):
        score = row['hotspot_score']
        years = row['years_to_1_5C']
        
        if score >= 0.8 or years < 20:
            return 'Critical'
        elif score >= 0.6 or years < 50:
            return 'High'
        elif score >= 0.4 or years < 100:
            return 'Medium'
        else:
            return 'Low'
    
    ranking['priority_level'] = ranking.apply(get_priority, axis=1)
    
   
    ranking = ranking.sort_values('hotspot_score', ascending=False)
    ranking['priority_rank'] = range(1, len(ranking) + 1)
    
    print(f"\n  Priority Distribution:")
    for level in ['Critical', 'High', 'Medium', 'Low']:
        count = (ranking['priority_level'] == level).sum()
        print(f"    {level}: {count} districts")
    
    return ranking


def save_results(trends, hotspot, paris, ipcc, projections, ranking):
    """Save all analysis results."""
    print("\n" + "=" * 70)
    print("SAVING ANALYSIS RESULTS")
    print("=" * 70)
    
    trends.to_csv(RESULTS_DIR / 'climate_trends.csv', index=False)
    hotspot.to_csv(RESULTS_DIR / 'hotspot_analysis.csv', index=False)
    paris.to_csv(RESULTS_DIR / 'paris_compliance.csv', index=False)
    ipcc.to_csv(RESULTS_DIR / 'ipcc_risk.csv', index=False)
    projections.to_csv(RESULTS_DIR / 'climate_projections.csv', index=False)
    ranking.to_csv(RESULTS_DIR / 'comprehensive_ranking.csv', index=False)
    
    print(f"  ✓ climate_trends.csv")
    print(f"  ✓ hotspot_analysis.csv")
    print(f"  ✓ paris_compliance.csv")
    print(f"  ✓ ipcc_risk.csv")
    print(f"  ✓ climate_projections.csv")
    print(f"  ✓ comprehensive_ranking.csv")

def main():
    print("\n" + "=" * 70)
    print("HOTSPOT ANALYSIS ...")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    data = load_data()
    
    trends = calculate_climate_trends(data)
    hotspot = calculate_hotspot_scores(data, trends)
    paris = calculate_paris_compliance(hotspot)
    ipcc = calculate_ipcc_risk(hotspot)
    projections = calculate_projections(hotspot)
    ranking = calculate_comprehensive_ranking(hotspot, paris, ipcc)
    

    save_results(trends, hotspot, paris, ipcc, projections, ranking)
 
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"""
  Data Coverage:
    Districts: {len(ranking)}
    Years: {data['full']['year'].min()}-{data['full']['year'].max()}
    
  Model Performance (LODO-CV):
    Mean R²: {data['lodo']['r2'].mean():.4f}
    Median R²: {data['lodo']['r2'].median():.4f}
    Districts R² > 0.9: {(data['lodo']['r2'] > 0.9).sum()}/{len(data['lodo'])}
    
  Top 5 Climate Hotspots:
    1. {ranking.iloc[0]['district']} (Score: {ranking.iloc[0]['hotspot_score']:.3f})
    2. {ranking.iloc[1]['district']} (Score: {ranking.iloc[1]['hotspot_score']:.3f})
    3. {ranking.iloc[2]['district']} (Score: {ranking.iloc[2]['hotspot_score']:.3f})
    4. {ranking.iloc[3]['district']} (Score: {ranking.iloc[3]['hotspot_score']:.3f})
    5. {ranking.iloc[4]['district']} (Score: {ranking.iloc[4]['hotspot_score']:.3f})
    
  Paris Agreement Status:
    Critical: {(ranking['priority_level'] == 'Critical').sum()} districts
    High Priority: {(ranking['priority_level'] == 'High').sum()} districts
    """)
    
    print("=" * 70)
    print("✅ HOTSPOT ANALYSIS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
